"""TCP-воркер инференса (протокол infer_worker, см. runtime-core/src/bin/infer_worker.rs).

  Клиент(gateway) → воркер:  {"width":W,"height":H,"pts_ms":T}\n + u32-LE + BGR
  Воркер → клиент:           {"type":"event","payload":{…}}\n          (0..N)
                             {"type":"frame_result","payload":{…}}\n   (1 на кадр)

Запуск:  python -m pyinfer.worker --listen 127.0.0.1:9910
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .bytetrack import BYTETracker
from .candidates import IouTracker, ObjectCandidates
from .config import Config, load_config
from .scene_fsm import SceneAnalyzer
from .yolo_onnx import YoloOnnx


def log(msg: str):
    print(f"[pyinfer] {msg}", file=sys.stderr, flush=True)


def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-6)


def _box_on_person(box, persons, contain_thr: float) -> bool:
    """True, если ≥ contain_thr площади box внутри какого-либо человека (его держат)."""
    barea = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    for p in persons:
        pb = p.bbox
        ix1 = max(box[0], pb[0]); iy1 = max(box[1], pb[1])
        ix2 = min(box[2], pb[2]); iy2 = min(box[3], pb[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter / barea >= contain_thr:
            return True
    return False


def _merge_regions(primary, secondary, iou_thr: float) -> list:
    """primary (YOLO) + те из secondary (MOG2), что не дублируют primary по IoU."""
    out = list(primary)
    for s in secondary:
        if all(_iou(s, p) < iou_thr for p in primary):
            out.append(s)
    return out


def _drop_oversized(regions, max_area_ratio, fw, fh):
    """Убрать объект-регионы крупнее `max_area_ratio` площади кадра.

    «Оставленная вещь» (сумка/рюкзак/чемодан) мала в кадре. Большой регион — это
    ближний план или силуэт крупного СИДЯЩЕГО человека, которого YOLO не взял как
    person. Без класса отличаем по размеру: люди-силуэты крупные, предметы — нет.
    """
    if max_area_ratio <= 0 or fw <= 0 or fh <= 0:
        return list(regions)
    cap = max_area_ratio * float(fw) * float(fh)
    return [r for r in regions
            if (r[2] - r[0]) * (r[3] - r[1]) <= cap]


def _drop_in_ignore_rect(regions, ignore_rect, fw, fh):
    """Убрать объект-кандидаты, чей центроид попал в зону игнорирования
    (норм. координаты [x1,y1,x2,y2]).

    Зеркалит native/src/frame_filter.cpp: люди НЕ трогаются (их ведёт ByteTrack),
    отсекаются только class-agnostic объект-регионы. Главный давитель шума на
    «мёртвых» участках сцены (витрина/монитор/эскалатор/блики в зоне rect).
    """
    if not ignore_rect or fw <= 0 or fh <= 0:
        return list(regions)
    x1, y1, x2, y2 = ignore_rect
    out = []
    for r in regions:
        nx = 0.5 * (r[0] + r[2]) / fw
        ny = 0.5 * (r[1] + r[3]) / fh
        if x1 <= nx <= x2 and y1 <= ny <= y2:
            continue
        out.append(r)
    return out


def _suppress_near_persons(regions, persons, margin=0.10, contain=0.55):
    """Убрать кандидатов, которые реально НА человеке (его держат/часть тела).

    Лёгкое подавление (не широкая зона!): давим регион, только если ≥`contain`
    его площади внутри слегка расширенного (на `margin`) бокса человека. Так
    предмет РЯДОМ с человеком (на лавке у ног, сбоку) сохраняется и виден — это
    важно для recall: новый предмет на лавке не должен глушиться, если поблизости
    кто-то есть.
    """
    if not persons:
        return list(regions)
    zones = []
    for p in persons:
        x1, y1, x2, y2 = p.bbox
        pw, ph = (x2 - x1), (y2 - y1)
        zones.append((x1 - margin * pw, y1 - margin * ph, x2 + margin * pw, y2 + margin * ph))
    out = []
    for r in regions:
        rarea = max(1.0, (r[2] - r[0]) * (r[3] - r[1]))
        drop = False
        for z in zones:
            ix = max(0.0, min(r[2], z[2]) - max(r[0], z[0]))
            iy = max(0.0, min(r[3], z[3]) - max(r[1], z[1]))
            if ix * iy / rarea >= contain:
                drop = True
                break
        if not drop:
            out.append(r)
    return out


@dataclass
class PersonDet:
    track_id: int
    confidence: float
    bbox: tuple  # (x1,y1,x2,y2)


class Session:
    """Состояние одного TCP-подключения (один видеопоток)."""

    def __init__(self, model: YoloOnnx, cfg: Config):
        self.model = model
        self.cfg = cfg
        self.frame_seq = 0
        self.last_pts = None
        self._make_state()

    def _make_state(self):
        self.tracker = BYTETracker(self.cfg.tracker)
        self.candidates = ObjectCandidates(self.cfg.analyzer.frame_diff_min_region_area_px)
        self.obj_tracker = IouTracker(
            self.cfg.analyzer.tracker_iou_match_threshold,
            self.cfg.analyzer.tracker_max_missed_frames,
        )
        self.analyzer = SceneAnalyzer(self.cfg.analyzer)

    def reset(self):
        self._make_state()

    def process(self, bgr: np.ndarray, pts_ms: float) -> tuple[list, dict]:
        # seek / переоткрытие → сбрасываем модель фона и трекеры
        if self.last_pts is not None and abs(pts_ms - self.last_pts) > 2000.0:
            self.reset()
        self.last_pts = pts_ms
        self.frame_seq += 1
        h, w = bgr.shape[:2]

        t0 = time.perf_counter()
        # Один проход YOLO: люди + «оставляемые» предметы (второй источник кандидатов)
        # + «слабые» люди (маска подавления фантомов сидящих/ближнего плана).
        dets, yolo_objs, weak = self.model.detect_all(
            bgr, self.cfg.object_detect_classes, self.cfg.object_detect_conf,
            self.cfg.person_suppress_conf)
        t1 = time.perf_counter()

        # M2: ByteTrack людей
        stracks = self.tracker.update(dets)
        persons = []
        for st in stracks:
            b = st.tlbr
            persons.append(PersonDet(int(st.track_id), float(st.score),
                                     (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        # Расширенный список ТОЛЬКО для подавления регионов (в FSM/UI/owner НЕ идёт):
        #  + «потерянные» person-треки (Kalman-предсказание, мерцающий человек),
        #  + «слабые» сырые person-детекции (сидящие/ближний план, score<self.conf).
        suppress_persons = list(persons)
        lost = self.tracker.predicted_lost(self.cfg.analyzer.suppress_lost_person_frames)
        for t in lost:
            suppress_persons.append(PersonDet(int(t.track_id), float(t.score),
                                              tuple(float(x) for x in t.tlbr)))
        for b in weak:
            suppress_persons.append(PersonDet(-1, float(b[4]),
                                              (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        t2 = time.perf_counter()

        # M3: class-agnostic объекты сцены = MOG2 (движение) + YOLO (статичные предметы)
        objects = []
        if self.cfg.analyzer.use_frame_diff_detector:
            regions = self.candidates.process(bgr, suppress_persons)
            yolo_boxes = [(o[0], o[1], o[2], o[3]) for o in yolo_objs]
            regions = _merge_regions(yolo_boxes, regions, 0.40)
            # Жёсткое подавление кандидатов в зоне людей (главный давитель шума) —
            # включая предсказанные «потерянные» треки сидящих.
            regions = _suppress_near_persons(regions, suppress_persons)
            # Верхний размерный порог: силуэты крупных людей в ближнем плане — не вещи.
            regions = _drop_oversized(regions, self.cfg.analyzer.max_object_area_ratio, w, h)
            # Зона игнорирования объектов (config.analyzer.ignore_detection_norm_rect):
            # в py-infer-режиме native frame_filter не работает, фильтруем здесь.
            regions = _drop_in_ignore_rect(regions, self.cfg.analyzer.ignore_norm_rect, w, h)
            gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            objects = self.obj_tracker.update(regions, persons, gray_full)
        t3 = time.perf_counter()

        # M4: поведенческая FSM
        ts = time.time()
        events = self.analyzer.ingest(ts, pts_ms, self.cfg.camera_id, objects, persons, w, h)
        tracks = self.analyzer.tracks_snapshot(ts)
        t4 = time.perf_counter()

        frame_result = {
            "frame_id": self.frame_seq,
            "pts_ms": float(pts_ms),
            "stats": {
                "detections": len(objects),
                "persons": len(persons),
                "infer_ms": (t1 - t0) * 1000.0,
                "preprocess_ms": 0.0,
                "tracker_ms": (t2 - t1) * 1000.0,
                "analyzer_ms": (t4 - t3) * 1000.0,
            },
            "tracks": tracks,
            "persons": [
                {"track_id": p.track_id, "confidence": p.confidence,
                 "bbox": [p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]]}
                for p in persons
            ],
        }
        return events, frame_result


def _recv_exact(rd, n: int) -> bytes | None:
    buf = rd.read(n)
    if not buf or len(buf) != n:
        return None
    return buf


def handle_client(conn: socket.socket, addr, model: YoloOnnx, cfg: Config):
    log(f"client connected {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rd = conn.makefile("rb")
    sess = Session(model, cfg)
    try:
        while True:
            line = rd.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                log("bad meta json, skipping")
                continue
            w = int(meta.get("width", 0))
            h = int(meta.get("height", 0))
            pts_ms = float(meta.get("pts_ms", 0))
            if w <= 0 or h <= 0:
                continue

            len_buf = _recv_exact(rd, 4)
            if len_buf is None:
                break
            n = struct.unpack("<I", len_buf)[0]
            payload = _recv_exact(rd, n)
            if payload is None:
                break
            if n != w * h * 3:
                log(f"frame size mismatch got={n} expected={w*h*3}, skipping")
                continue

            bgr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)

            try:
                events, frame_result = sess.process(bgr, pts_ms)
            except Exception as e:  # инференс/трекинг не должны рвать соединение
                err = json.dumps({"type": "error", "msg": str(e)}) + "\n"
                conn.sendall(err.encode("utf-8"))
                continue

            out = []
            for ev in events:
                out.append(json.dumps({"type": "event", "payload": ev}))
            out.append(json.dumps({"type": "frame_result", "payload": frame_result}))
            conn.sendall(("\n".join(out) + "\n").encode("utf-8"))
    except (ConnectionError, OSError) as e:
        log(f"client {addr} session ended: {e}")
    finally:
        try:
            rd.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        log(f"client disconnected {addr}")


def main():
    ap = argparse.ArgumentParser(prog="pyinfer.worker")
    ap.add_argument("--listen", default="127.0.0.1:9910")
    args = ap.parse_args()

    root = Path(os.environ.get("INTEGRA_PROJECT_ROOT") or os.getcwd())
    cfg = load_config(root)
    yolo_conf = max(0.05, min(0.30, cfg.tracker.low_thresh))
    log(f"loading model: {cfg.model_path} (imgsz={cfg.imgsz}, yolo_conf={yolo_conf}, nms_iou={cfg.iou})")
    model = YoloOnnx(cfg.model_path, imgsz=cfg.imgsz, conf=yolo_conf,
                     iou=cfg.iou, person_class=cfg.person_class, threads=cfg.onnx_threads)
    log("model loaded")

    host, port = args.listen.rsplit(":", 1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(8)
    log(f"infer_worker listening {args.listen}")

    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, model, cfg), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
