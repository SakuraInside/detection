"""Отладочный рендер: видит ли MOG2/трекер статичные предметы.

Рисует на кадрах:
  зелёный  — сырые регионы MOG2 (ObjectCandidates.process, ДО трекера)
  жёлтый   — подтверждённые треки трекера (hits>=CONFIRM, не stable)
  красный  — стабильные треки (stable=True, пиксельно-верифицированы)
  синий    — люди (ByteTrack)

Сохраняет каждый STEP-й кадр в debug_frames/. Запуск:
  python -m pyinfer._debug_render "data/<file>.mkv" [max_frames] [step]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .bytetrack import BYTETracker
from .candidates import (CONFIRM_HITS, IouTracker, ObjectCandidates)
from .config import load_config
from .worker import (_box_on_person, _merge_regions, _suppress_near_persons,
                     _drop_oversized, _drop_in_ignore_rect)
from .yolo_onnx import YoloOnnx


class _P:
    __slots__ = ("track_id", "confidence", "bbox")

    def __init__(self, tid, conf, bbox):
        self.track_id, self.confidence, self.bbox = tid, conf, bbox


def main():
    video = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "debug_frames"
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()

    cfg = load_config(root)
    yolo_conf = max(0.05, min(0.30, cfg.tracker.low_thresh))
    model = YoloOnnx(cfg.model_path, imgsz=cfg.imgsz, conf=yolo_conf,
                     iou=cfg.iou, person_class=cfg.person_class, threads=cfg.onnx_threads)
    tracker = BYTETracker(cfg.tracker)
    cand = ObjectCandidates(cfg.analyzer.frame_diff_min_region_area_px)
    obj_tracker = IouTracker(cfg.analyzer.tracker_iou_match_threshold,
                             cfg.analyzer.tracker_max_missed_frames)

    cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = saved = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        dets, yolo_objs, weak = model.detect_all(
            frame, cfg.object_detect_classes, cfg.object_detect_conf, cfg.person_suppress_conf)
        stracks = tracker.update(dets)
        persons = [_P(int(s.track_id), float(s.score),
                      tuple(float(x) for x in s.tlbr)) for s in stracks]
        h, w = frame.shape[:2]
        suppress_persons = list(persons)
        for t in tracker.predicted_lost(cfg.analyzer.suppress_lost_person_frames):
            suppress_persons.append(_P(int(t.track_id), float(t.score),
                                       tuple(float(x) for x in t.tlbr)))
        weak_boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in weak]
        for wb in weak_boxes:
            suppress_persons.append(_P(-1, 0.1, wb))
        regions = cand.process(frame, suppress_persons)
        yolo_boxes = [(o[0], o[1], o[2], o[3]) for o in yolo_objs
                      if not _box_on_person((o[0], o[1], o[2], o[3]), persons, 0.70)]
        regions = _merge_regions(yolo_boxes, regions, 0.40)
        regions = _suppress_near_persons(regions, suppress_persons)
        regions = _drop_oversized(regions, cfg.analyzer.max_object_area_ratio, w, h)
        regions = _drop_in_ignore_rect(regions, cfg.analyzer.ignore_norm_rect, w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objs = obj_tracker.update(regions, persons, gray)

        if n % step != 0:
            continue
        vis = frame.copy()
        # сырые регионы MOG2 — зелёные
        for r in regions:
            cv2.rectangle(vis, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)
        # внутренние треки: жёлтый confirmed / красный stable
        for t in obj_tracker._tracks:
            if getattr(t, "baseline", False):
                color = (255, 0, 255)  # magenta = baseline (мебель, в FSM не идёт)
                tag = "BASE"
            elif t.stable:
                color = (0, 0, 255)  # red = stable (идёт в FSM)
                tag = "S"
            elif t.hits >= CONFIRM_HITS:
                color = (0, 200, 255)  # yellow = confirmed
                tag = ""
            else:
                color = (120, 120, 120)
                tag = ""
            cv2.rectangle(vis, (int(t.bbox[0]), int(t.bbox[1])),
                          (int(t.bbox[2]), int(t.bbox[3])), color, 2)
            cv2.putText(vis, f"id{t.track_id} h{t.hits}{tag}",
                        (int(t.bbox[0]), int(t.bbox[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for p in persons:
            cv2.rectangle(vis, (int(p.bbox[0]), int(p.bbox[1])),
                          (int(p.bbox[2]), int(p.bbox[3])), (255, 100, 0), 1)
        ts = n / fps
        cv2.putText(vis, f"f{n} t={ts:.1f}s regions={len(regions)} stable={sum(1 for t in obj_tracker._tracks if t.stable)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        path = out_dir / f"f{n:06d}.jpg"
        cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        saved += 1
    cap.release()
    print(f"saved {saved} frames to {out_dir} (processed {n})")


if __name__ == "__main__":
    main()
