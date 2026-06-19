"""Оффлайн-прогон pyinfer-пайплайна по реальному видео — проверка шума детектора.

Запуск:  .venv\\Scripts\\python -m pyinfer._offline_check "data/<file>.mkv" [max_frames]

Считает по кадрам: число регионов MOG2, подтверждённых/стабильных объект-треков,
людей, и все события scene_fsm. Цель — убедиться, что нет лавины ложных
object_unattended / шумовых кандидатов на пустой сцене.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .bytetrack import BYTETracker
from .candidates import IouTracker, ObjectCandidates
from .config import load_config
from .scene_fsm import SceneAnalyzer
from .worker import _build_model


class _P:
    __slots__ = ("track_id", "confidence", "bbox")

    def __init__(self, tid, conf, bbox):
        self.track_id = tid
        self.confidence = conf
        self.bbox = bbox


def main():
    if len(sys.argv) < 2:
        print("usage: python -m pyinfer._offline_check <video> [max_frames]")
        raise SystemExit(2)
    video = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 900

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root)
    yolo_conf = max(0.05, min(0.30, cfg.tracker.low_thresh))
    model = _build_model(cfg, yolo_conf)

    tracker = BYTETracker(cfg.tracker)
    candidates = ObjectCandidates(cfg.analyzer.frame_diff_min_region_area_px)
    obj_tracker = IouTracker(cfg.analyzer.tracker_iou_match_threshold,
                             cfg.analyzer.tracker_max_missed_frames)
    analyzer = SceneAnalyzer(cfg.analyzer)

    cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"cannot open {video}")
        raise SystemExit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ev_counter = Counter()
    region_hist = []
    confirmed_hist = []
    persons_hist = []
    stable_hist = []
    ever_stable_ids = set()
    first_unattended_pts = None
    _life = {}  # track_id -> [first_stable_frame, last_stable_frame, last_bbox]
    fw = fh = 1
    t_start = time.perf_counter()
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        pts_ms = n / fps * 1000.0
        h, w = frame.shape[:2]
        fw, fh = w, h

        dets, _yolo_objs, weak = model.detect_all(
            frame, cfg.object_detect_classes, cfg.object_detect_conf, cfg.person_suppress_conf)
        stracks = tracker.update(dets)
        persons = [_P(int(s.track_id), float(s.score),
                      tuple(float(x) for x in s.tlbr)) for s in stracks]

        # потерянные треки + слабые люди — только в подавление (зеркалим worker.process)
        suppress_persons = list(persons)
        for t in tracker.predicted_lost(cfg.analyzer.suppress_lost_person_frames):
            suppress_persons.append(_P(int(t.track_id), float(t.score),
                                       tuple(float(x) for x in t.tlbr)))
        for b in weak:
            suppress_persons.append(_P(-1, float(b[4]),
                                       (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))

        regions = candidates.process(frame, suppress_persons)
        from .worker import (_box_on_person, _merge_regions, _suppress_near_persons,
                             _drop_in_ignore_rect, _drop_oversized)
        yolo_boxes = [(o[0], o[1], o[2], o[3]) for o in _yolo_objs
                      if not _box_on_person((o[0], o[1], o[2], o[3]), persons, 0.70)]
        regions = _merge_regions(yolo_boxes, regions, 0.40)
        # зеркалим прод-пайплайн (worker.process): подавление + размерный порог + ignore-rect
        regions = _suppress_near_persons(regions, suppress_persons)
        regions = _drop_oversized(regions, cfg.analyzer.max_object_area_ratio, w, h)
        regions = _drop_in_ignore_rect(regions, cfg.analyzer.ignore_norm_rect, w, h)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = obj_tracker.update(regions, persons, gray_full)

        ts = n / fps  # детерминированное «время» из позиции в видео
        events = analyzer.ingest(ts, pts_ms, cfg.camera_id, objects, persons, w, h)
        for e in events:
            ev_counter[e["type"]] += 1
            if e["type"] in ("object_unattended", "object_removed", "object_missing"):
                bx = e["bbox"]
                ncx = (bx[0] + bx[2]) / 2 / max(1, w)
                ncy = (bx[1] + bx[3]) / 2 / max(1, h)
                print(f"  >>> {e['type']} @ {pts_ms/1000:6.1f}s  id={e['track_id']} "
                      f"центр=({ncx:.2f},{ncy:.2f})  note={e.get('note','')}")
            if e["type"] == "object_unattended" and first_unattended_pts is None:
                first_unattended_pts = pts_ms / 1000.0

        region_hist.append(len(regions))
        confirmed_hist.append(len(objects))
        persons_hist.append(len(persons))
        cur_stable = [t for t in obj_tracker._tracks if t.stable and not getattr(t, "baseline", False)]
        stable_hist.append(len(cur_stable))
        for t in cur_stable:
            ever_stable_ids.add(t.track_id)
            d = _life.setdefault(t.track_id, [n, n, t.bbox, False])
            d[1] = n  # последний кадр, когда стабилен
            d[2] = t.bbox
            d[3] = d[3] or getattr(t, "owner_seen", False)  # владелец был привязан

        if n % 100 == 0:
            stable = sum(1 for t in obj_tracker._tracks if t.stable)
            print(f"frame {n:4d} pts={pts_ms/1000:6.1f}s "
                  f"regions={len(regions):2d} obj_tracks_visible={len(objects):2d} "
                  f"stable={stable:2d} persons={len(persons):2d} events={dict(ev_counter)}")

    dt = time.perf_counter() - t_start
    cap.release()

    def stats(name, arr):
        if not arr:
            print(f"  {name}: (нет данных)")
            return
        a = np.array(arr)
        print(f"  {name}: avg={a.mean():.2f} max={a.max()} p95={np.percentile(a,95):.0f}")

    print("\n==== ИТОГ ====")
    print(f"кадров обработано: {n}  за {dt:.1f}s  ({n/dt:.1f} fps инференс)")
    stats("MOG2 регионов/кадр", region_hist)
    stats("видимых объект-треков/кадр (после фильтра)", confirmed_hist)
    stats("стабильных треков/кадр", stable_hist)
    stats("людей/кадр", persons_hist)
    print(f"всего РАЗНЫХ треков, дошедших до stable: {len(ever_stable_ids)}")
    top = sorted(_life.values(), key=lambda d: d[1] - d[0], reverse=True)[:12]
    print("долгоживущие стабильные треки (старт->длительность @ норм.центр):")
    for first, last, bbox, owner in top:
        life_s = (last - first) / fps
        cx = (bbox[0] + bbox[2]) / 2 / fw
        cy = (bbox[1] + bbox[3]) / 2 / fh
        bench = "  <-- ЛАВКА" if (cx < 0.55 and cy > 0.5) else ""
        own = "  [OWNER]" if owner else ""
        print(f"   старт={first/fps:6.1f}s  жил {life_s:5.1f}s  центр=({cx:.2f},{cy:.2f}){own}{bench}")
    if first_unattended_pts is not None:
        print(f"первый object_unattended на: {first_unattended_pts:.1f}s видео")
    print(f"события: {dict(ev_counter) if ev_counter else 'НЕТ'}")
    n_unattended = ev_counter.get("object_unattended", 0)
    print(f"\nложно-тревожный индикатор: object_unattended={n_unattended} "
          f"({n_unattended / max(1, n) * fps * 60:.1f}/мин видео)")


if __name__ == "__main__":
    main()
