"""YOLO + tracking слой с optional inference worker-процессом."""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import get_context
from multiprocessing.queues import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import ModelConfig


_TORCH = None
_YOLO_CLASS = None


def _get_torch():
    global _TORCH
    if _TORCH is None:
        import torch  # type: ignore

        _TORCH = torch
    return _TORCH


def _get_yolo_class():
    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        from ultralytics import YOLO  # type: ignore

        _YOLO_CLASS = YOLO
    return _YOLO_CLASS


@dataclass
class Detection:
    track_id: int
    cls_id: int
    cls_name: str
    confidence: float
    # Координаты bbox в формате xyxy в системе координат кадра.
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]


@dataclass
class DetectionResult:
    detections: list[Detection]
    persons: list[Detection]
    inference_ms: float
    # Время подэтапов (мс), например unknown_ref — для /api/metrics.
    stage_ms: dict[str, float] = field(default_factory=dict)


def detection_to_dict(det: Detection) -> dict:
    return {
        "track_id": int(det.track_id),
        "cls_id": int(det.cls_id),
        "cls_name": str(det.cls_name),
        "confidence": float(det.confidence),
        "bbox": [float(v) for v in det.bbox],
        "centroid": [float(v) for v in det.centroid],
    }


def detection_from_dict(data: dict) -> Detection:
    bbox = tuple(float(v) for v in data.get("bbox", [0, 0, 0, 0]))
    centroid = tuple(float(v) for v in data.get("centroid", [0, 0]))
    return Detection(
        track_id=int(data.get("track_id", 0)),
        cls_id=int(data.get("cls_id", 0)),
        cls_name=str(data.get("cls_name", "")),
        confidence=float(data.get("confidence", 0.0)),
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        centroid=(centroid[0], centroid[1]),
    )


def detection_result_to_dict(result: DetectionResult) -> dict:
    return {
        "detections": [detection_to_dict(d) for d in result.detections],
        "persons": [detection_to_dict(p) for p in result.persons],
        "inference_ms": float(result.inference_ms),
        "stage_ms": dict(result.stage_ms or {}),
    }


def detection_result_from_dict(data: dict) -> DetectionResult:
    dets = [detection_from_dict(x) for x in (data.get("detections") or [])]
    persons = [detection_from_dict(x) for x in (data.get("persons") or [])]
    sm = data.get("stage_ms")
    return DetectionResult(
        detections=dets,
        persons=persons,
        inference_ms=float(data.get("inference_ms", 0.0)),
        stage_ms=dict(sm) if isinstance(sm, dict) else {},
    )


class _LocalYoloTracker:
    """Потокобезопасная обертка для single-worker сценария.

    Трекер хранит внутреннее состояние между вызовами, поэтому доступ
    сериализуется через lock.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        torch = _get_torch()
        self._cfg = cfg
        self._lock = threading.Lock()
        self._model = None
        # Если GPU недоступен, автоматически откатываемся на CPU.
        self._device = cfg.device if torch.cuda.is_available() else "cpu"
        self._half = cfg.half and self._device.startswith("cuda")
        self._frame_counter = 0
        self._aux_next_id = 1_000_000
        self._aux_tracks: dict[int, tuple[int, tuple[float, float], tuple[float, float, float, float], float]] = {}
        self._motion_prev_gray: Optional[np.ndarray] = None
        self._static_gray: Optional[np.ndarray] = None
        self._long_static_gray: Optional[np.ndarray] = None
        self._pre_motion_gray: Optional[np.ndarray] = None
        self._person_motion_mask: Optional[np.ndarray] = None
        self._gray_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(4, int(cfg.unknown_gray_history_maxlen))
        )
        self._bg_seen_frames = 0
        self._no_motion_frames = 0
        if self._device.startswith("cuda"):
            # Небольшие оптимизации CUDA для стабильного fps.
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        self._load()

    def _load(self) -> None:
        YOLO = _get_yolo_class()
        # При имени официального чекпойнта Ultralytics сам скачает веса.
        weights = self._cfg.weights
        if not Path(weights).exists() and "/" not in weights and "\\" not in weights:
            # Здесь ничего не делаем: библиотека сама скачает официальный чекпойнт.
            pass
        try:
            self._model = YOLO(weights)
        except Exception as exc:
            msg = str(exc).lower()
            # Если файл весов скачался/скопировался частично, torch читает его как
            # битый zip-архив. Удаляем поврежденный файл и пробуем загрузить снова.
            if "failed finding central directory" in msg or "pytorchstreamreader failed" in msg:
                self._cleanup_corrupted_weights(weights)
                self._model = YOLO(weights)
            else:
                raise
        # Перенос на устройство делаем один раз.
        try:
            self._model.to(self._device)
        except Exception:
            pass

    @staticmethod
    def _cleanup_corrupted_weights(weights: str) -> None:
        candidates = []
        p = Path(weights)
        if p.exists():
            candidates.append(p)
        # Ultralytics при short-name часто кладет файл в cwd.
        cwd_copy = Path.cwd() / p.name
        if cwd_copy.exists():
            candidates.append(cwd_copy)
        # Удаляем дубликаты по абсолютному пути.
        seen: set[str] = set()
        for item in candidates:
            key = str(item.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                item.unlink(missing_ok=True)
            except Exception:
                pass

    def reload(self, cfg: ModelConfig) -> None:
        torch = _get_torch()
        with self._lock:
            # Полная перезагрузка нужна, если меняли веса/девайс/half/tracker.
            self._cfg = cfg
            self._device = cfg.device if torch.cuda.is_available() else "cpu"
            self._half = cfg.half and self._device.startswith("cuda")
            self._frame_counter = 0
            self._load()
            self._motion_prev_gray = None
            self._static_gray = None
            self._long_static_gray = None
            self._pre_motion_gray = None
            self._person_motion_mask = None
            self._gray_history = deque(maxlen=max(4, int(cfg.unknown_gray_history_maxlen)))
            self._bg_seen_frames = 0
            self._no_motion_frames = 0

    @property
    def names(self) -> dict[int, str]:
        if self._model is None:
            return {}
        # В разных версиях Ultralytics names может быть dict или list.
        names = self._model.names  # type: ignore[attr-defined]
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        return {i: n for i, n in enumerate(names)}

    def reset(self) -> None:
        """Сброс состояния трекера (после seek или смены файла)."""
        with self._lock:
            if self._model is None:
                return
            try:
                if hasattr(self._model, "predictor") and self._model.predictor is not None:
                    if hasattr(self._model.predictor, "trackers") and self._model.predictor.trackers:
                        for t in self._model.predictor.trackers:
                            if hasattr(t, "reset"):
                                t.reset()
            except Exception:
                pass
            self._aux_tracks.clear()
            self._aux_next_id = 1_000_000
            self._frame_counter = 0
            self._motion_prev_gray = None
            self._static_gray = None
            self._long_static_gray = None
            self._pre_motion_gray = None
            self._person_motion_mask = None
            self._gray_history = deque(maxlen=max(4, int(self._cfg.unknown_gray_history_maxlen)))
            self._bg_seen_frames = 0
            self._no_motion_frames = 0

    def infer(
        self,
        frame: np.ndarray,
        *,
        max_roi_count: Optional[int] = None,
        priority: Optional[str] = None,
    ) -> DetectionResult:
        torch = _get_torch()
        cfg = self._cfg
        self._frame_counter += 1
        # Грубый фильтр по именам классов для подавления ложных тревог
        # (мебель, животные, электроника и т.п.).
        excluded = {str(n).strip().lower() for n in cfg.excluded_class_names if str(n).strip()}
        class_min_conf = {
            str(k).strip().lower(): float(v) for k, v in (cfg.class_min_conf or {}).items() if str(k).strip()
        }
        min_box_by_class = {
            str(k).strip().lower(): int(v) for k, v in (cfg.min_box_size_by_class or {}).items() if str(k).strip()
        }
        with self._lock:
            assert self._model is not None
            # Пустой object_classes => разрешаем все классы (кроме person
            # в отдельной ветке), чтобы не терять "неочевидные" объекты.
            classes = None if not cfg.object_classes else list(set(cfg.object_classes + [cfg.person_class]))
            # persist=True сохраняет внутреннее состояние трекера между кадрами.
            # Для сохранения мелких объектов в верхней/дальней зоне запускаем
            # модель с более низким "сырым" conf, а окончательную фильтрацию
            # делаем ниже адаптивно по Y-координате.
            raw_conf = float(min(cfg.conf, cfg.min_conf_upper, cfg.min_conf_lower))
            with torch.inference_mode():
                results = self._model.track(
                    source=frame,
                    imgsz=cfg.imgsz,
                    conf=raw_conf,
                    iou=cfg.iou,
                    device=self._device,
                    half=self._half,
                    tracker=cfg.tracker,
                    persist=True,
                    classes=classes,
                    verbose=False,
                )
        if not results:
            return DetectionResult([], [], 0.0)
        r = results[0]
        speed = r.speed or {}
        inference_ms = float(speed.get("inference", 0.0)) + float(speed.get("preprocess", 0.0))
        names = self.names

        detections: list[Detection] = []
        persons: list[Detection] = []
        if r.boxes is None:
            return DetectionResult([], [], inference_ms)

        ids: list[int] = []
        has_ids = r.boxes.id is not None
        if has_ids:
            ids = r.boxes.id.int().cpu().tolist()
        cls = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.float().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()
        if not has_ids:
            # Fallback: если трекер не выдал id, создаем стабильные id локально.
            ids = self._assign_aux_ids(cls, xyxy, frame.shape[0], frame.shape[1])
        h, w = frame.shape[:2]
        upper_cut = float(h) * float(cfg.upper_region_y_ratio)
        bottom_cut = float(h) * float(cfg.bottom_region_y_ratio)
        border_relax = int(max(0, cfg.border_relax_px))
        min_box_size = float(max(1, cfg.min_box_size_px))
        for tid, c, p, box in zip(ids, cls, conf, xyxy):
            cls_name = names.get(int(c), str(c))
            cls_norm = cls_name.strip().lower()
            # Жестко убираем монитор/TV-класс: для этой задачи это шум.
            if cls_norm in {"tv", "monitor"}:
                continue
            if cls_norm in excluded:
                # Убираем заведомо "шумные" классы из пользовательского blacklist.
                continue
            x1, y1, x2, y2 = box
            min_box_size_cls = float(max(1, min_box_by_class.get(cls_norm, int(min_box_size))))
            if (x2 - x1) < min_box_size_cls or (y2 - y1) < min_box_size_cls:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # Перспективная фильтрация confidence:
            # в верхней части разрешаем более низкий порог, в нижней — строже.
            if cy <= upper_cut:
                min_conf = float(cfg.min_conf_upper)
            elif cy >= bottom_cut:
                min_conf = float(cfg.min_conf_bottom)
            else:
                min_conf = float(cfg.min_conf_lower)
            touches_border = (
                x1 <= border_relax or y1 <= border_relax or x2 >= (w - border_relax) or y2 >= (h - border_relax)
            )
            if touches_border:
                min_conf = min(min_conf, float(cfg.min_conf_border))
            if cls_norm in class_min_conf:
                min_conf = min(min_conf, class_min_conf[cls_norm])
            if float(p) < min_conf:
                continue
            det = Detection(
                track_id=int(tid),
                cls_id=int(c),
                cls_name=cls_name,
                confidence=float(p),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                centroid=(cx, cy),
            )
            if int(c) == cfg.person_class:
                persons.append(det)
            elif not cfg.object_classes or int(c) in cfg.object_classes:
                # В detections складываем только целевые объекты (без person).
                detections.append(det)
        # Дополнительная стабилизация ID людей:
        # даже при кратких сбоях трекера сохраняем непрерывность track_id.
        if persons:
            person_ids = self._assign_aux_ids(
                classes=[cfg.person_class] * len(persons),
                boxes_xyxy=[list(p.bbox) for p in persons],
                h=h,
                w=w,
            )
            for p, stable_id in zip(persons, person_ids):
                p.track_id = int(stable_id)

        # Второй проход по зоне пола: повышает recall для объектов на полу.
        # При scheduler hints в режиме перегруза ограничиваем/отключаем ROI-верификацию.
        floor_roi_enabled = bool(cfg.floor_roi_enabled and frame.shape[0] > 0)
        if priority == "low":
            floor_roi_enabled = False
        if max_roi_count is not None and int(max_roi_count) <= 1:
            floor_roi_enabled = False
        stage_ms: dict[str, float] = {}
        if floor_roi_enabled:
            t0 = time.perf_counter()
            floor_det = self._infer_floor_roi(frame, cfg, excluded, class_min_conf, names)
            stage_ms["floor_roi"] = (time.perf_counter() - t0) * 1000.0
            if floor_det:
                if max_roi_count is not None:
                    cap = max(1, int(max_roi_count))
                    floor_det = sorted(floor_det, key=lambda d: d.confidence, reverse=True)[:cap]
                detections = self._merge_detections(detections, floor_det, cfg.merge_iou_threshold)
        if (
            cfg.table_roi_enabled
            and frame.shape[0] > 0
            and (self._frame_counter % max(1, int(cfg.table_roi_every_n_frames)) == 0)
        ):
            t0 = time.perf_counter()
            roi_det = self._infer_table_roi_cup(frame, cfg, names)
            stage_ms["table_roi"] = (time.perf_counter() - t0) * 1000.0
            if roi_det:
                detections = self._merge_detections(detections, roi_det, cfg.merge_iou_threshold)
        # Отдельный слой "unknown" добавляет кандидаты новых объектов, которые
        # YOLO не смог уверенно отнести к известному классу.
        if cfg.unknown_layer_enabled:
            t0 = time.perf_counter()
            unknown_det = self._infer_unknown_objects(frame, detections, persons, cfg)
            stage_ms["unknown_ref"] = (time.perf_counter() - t0) * 1000.0
            if unknown_det:
                detections = self._merge_detections(detections, unknown_det, cfg.merge_iou_threshold)
        return DetectionResult(detections, persons, inference_ms, stage_ms)

    def _infer_floor_roi(
        self,
        frame: np.ndarray,
        cfg: ModelConfig,
        excluded: set[str],
        class_min_conf: dict[str, float],
        names: dict[int, str],
    ) -> list[Detection]:
        torch = _get_torch()
        h, w = frame.shape[:2]
        min_box_size = float(max(1, cfg.min_box_size_px))
        min_box_by_class = {
            str(k).strip().lower(): int(v) for k, v in (cfg.min_box_size_by_class or {}).items() if str(k).strip()
        }
        y0 = int(max(0, min(h - 1, h * float(cfg.floor_roi_y_ratio))))
        crop = frame[y0:, :]
        if crop.size == 0:
            return []
        with self._lock:
            assert self._model is not None
            classes = None if not cfg.object_classes else list(set(cfg.object_classes + [cfg.person_class]))
            with torch.inference_mode():
                rs = self._model.predict(
                    source=crop,
                    imgsz=int(cfg.floor_roi_imgsz),
                    conf=float(cfg.floor_roi_conf),
                    iou=cfg.iou,
                    device=self._device,
                    half=self._half,
                    classes=classes,
                    verbose=False,
                )
        if not rs:
            return []
        r = rs[0]
        if r.boxes is None:
            return []
        cls = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.float().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()
        aux_ids = self._assign_aux_ids(cls, [[x1, y1 + y0, x2, y2 + y0] for x1, y1, x2, y2 in xyxy], h, w)
        out: list[Detection] = []
        for tid, c, p, box in zip(aux_ids, cls, conf, xyxy):
            cls_name = names.get(int(c), str(c))
            cls_norm = cls_name.strip().lower()
            if cls_norm in {"tv", "monitor"} or cls_norm in excluded:
                continue
            x1, y1, x2, y2 = box
            min_box_size_cls = float(max(1, min_box_by_class.get(cls_norm, int(min_box_size))))
            if (x2 - x1) < min_box_size_cls or (y2 - y1) < min_box_size_cls:
                continue
            y1 += y0
            y2 += y0
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            min_conf = class_min_conf.get(cls_norm, float(cfg.min_conf_bottom))
            if float(p) < min_conf:
                continue
            if int(c) == cfg.person_class:
                continue
            out.append(
                Detection(
                    track_id=int(tid),
                    cls_id=int(c),
                    cls_name=cls_name,
                    confidence=float(p),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    centroid=(cx, cy),
                )
            )
        return out

    def _infer_table_roi_cup(
        self,
        frame: np.ndarray,
        cfg: ModelConfig,
        names: dict[int, str],
    ) -> list[Detection]:
        h, w = frame.shape[:2]
        x1 = int(max(0, min(w - 1, round(float(cfg.table_roi_x1) * w))))
        y1 = int(max(0, min(h - 1, round(float(cfg.table_roi_y1) * h))))
        x2 = int(max(x1 + 1, min(w, round(float(cfg.table_roi_x2) * w))))
        y2 = int(max(y1 + 1, min(h, round(float(cfg.table_roi_y2) * h))))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        with self._lock:
            assert self._model is not None
            with torch.inference_mode():
                rs = self._model.predict(
                    source=crop,
                    imgsz=int(cfg.table_roi_imgsz),
                    conf=float(cfg.table_roi_conf),
                    iou=float(cfg.table_roi_iou),
                    device=self._device,
                    half=self._half,
                    classes=[int(cfg.cup_class)],
                    verbose=False,
                )
        if not rs:
            return []
        r = rs[0]
        if r.boxes is None:
            return []

        cls = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.float().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()
        full_boxes = [[bx1 + x1, by1 + y1, bx2 + x1, by2 + y1] for bx1, by1, bx2, by2 in xyxy]
        aux_ids = self._assign_aux_ids(cls, full_boxes, h, w)
        min_box_size = float(max(1, cfg.min_box_size_px))
        min_box_by_class = {
            str(k).strip().lower(): int(v) for k, v in (cfg.min_box_size_by_class or {}).items() if str(k).strip()
        }

        out: list[Detection] = []
        for tid, c, p, box in zip(aux_ids, cls, conf, full_boxes):
            cls_name = names.get(int(c), str(c))
            cls_norm = cls_name.strip().lower()
            min_box_size_cls = float(max(1, min_box_by_class.get(cls_norm, int(min_box_size))))
            bx1, by1, bx2, by2 = box
            if (bx2 - bx1) < min_box_size_cls or (by2 - by1) < min_box_size_cls:
                continue
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            out.append(
                Detection(
                    track_id=int(tid),
                    cls_id=int(c),
                    cls_name=cls_name,
                    confidence=float(p),
                    bbox=(float(bx1), float(by1), float(bx2), float(by2)),
                    centroid=(cx, cy),
                )
            )
        return out

    def _assign_aux_ids(
        self,
        classes: list[int],
        boxes_xyxy: list[list[float]],
        h: int,
        w: int,
    ) -> list[int]:
        import time
        now_ts = time.time()
        # Clean stale auxiliary tracks.
        stale = [tid for tid, (_, _, _, ts) in self._aux_tracks.items() if (now_ts - ts) > 1.5]
        for tid in stale:
            self._aux_tracks.pop(tid, None)

        assigned: list[int] = []
        used: set[int] = set()
        for c, b in zip(classes, boxes_xyxy):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best_id = None
            best_score = 0.0
            for tid, (tc, (pcx, pcy), pb, _) in self._aux_tracks.items():
                if tid in used or tc != int(c):
                    continue
                iou = self._iou_tuple((x1, y1, x2, y2), pb)
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                score = iou - min(1.0, dist / 150.0) * 0.2
                if score > best_score and (iou >= 0.2 or dist <= 60.0):
                    best_score = score
                    best_id = tid
            if best_id is None:
                best_id = self._aux_next_id
                self._aux_next_id += 1
            self._aux_tracks[best_id] = (int(c), (cx, cy), (x1, y1, x2, y2), now_ts)
            used.add(best_id)
            assigned.append(best_id)
        return assigned

    @staticmethod
    def _iou_tuple(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _merge_detections(self, base: list[Detection], extra: list[Detection], iou_thr: float) -> list[Detection]:
        merged = list(base)
        for d in extra:
            duplicate = False
            for m in merged:
                if d.cls_id != m.cls_id:
                    continue
                if self._iou_tuple(d.bbox, m.bbox) >= float(iou_thr):
                    duplicate = True
                    # Оставляем более уверенную детекцию.
                    if d.confidence > m.confidence:
                        m.bbox = d.bbox
                        m.confidence = d.confidence
                        m.centroid = d.centroid
                        m.track_id = d.track_id
                    break
            if not duplicate:
                merged.append(d)
        return merged

    def _infer_unknown_objects(
        self,
        frame: np.ndarray,
        known_objects: list[Detection],
        persons: list[Detection],
        cfg: ModelConfig,
    ) -> list[Detection]:
        # Reference/motion на пониженном разрешении: меньше RAM и CPU; bbox -> полный кадр.
        h, w = frame.shape[:2]
        scale = float(max(0.15, min(1.0, float(cfg.unknown_reference_scale))))
        lw = max(32, int(round(w * scale)))
        lh = max(32, int(round(h * scale)))
        sx = w / float(lw)
        sy = h / float(lh)
        small = cv2.resize(frame, (lw, lh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        now_ts = time.time()
        self._gray_history.append((now_ts, gray.copy()))
        self._bg_seen_frames += 1
        if self._motion_prev_gray is None or self._motion_prev_gray.shape != gray.shape:
            self._motion_prev_gray = gray
            self._static_gray = gray.copy()
            self._long_static_gray = gray.copy()
            self._person_motion_mask = np.zeros(gray.shape, dtype=np.uint8)
            return []
        if self._bg_seen_frames <= int(max(1, cfg.unknown_warmup_frames)):
            self._motion_prev_gray = gray
            self._static_gray = gray.copy()
            self._long_static_gray = gray.copy()
            return []

        motion_thr = int(max(0, min(255, cfg.unknown_motion_threshold)))
        motion_diff = cv2.absdiff(gray, self._motion_prev_gray)
        _, motion_mask = cv2.threshold(motion_diff, motion_thr, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_min_area_low = float(max(1.0, cfg.unknown_motion_min_area_px)) * (scale * scale)
        in_motion = any(cv2.contourArea(c) >= motion_min_area_low for c in motion_contours)
        self._motion_prev_gray = gray

        if in_motion:
            if self._pre_motion_gray is None and self._static_gray is not None:
                self._pre_motion_gray = self._static_gray.copy()
            if self._person_motion_mask is None:
                self._person_motion_mask = np.zeros(gray.shape, dtype=np.uint8)
            person_pad = int(max(0, cfg.unknown_person_exclusion_px))
            for p in persons:
                px1, py1, px2, py2 = p.bbox
                x1 = max(0, int((px1 - person_pad) / sx))
                y1 = max(0, int((py1 - person_pad) / sy))
                x2 = min(lw, int((px2 + person_pad) / sx))
                y2 = min(lh, int((py2 + person_pad) / sy))
                if x2 > x1 and y2 > y1:
                    self._person_motion_mask[y1:y2, x1:x2] = 255
            self._no_motion_frames = 0
            return []
        self._no_motion_frames += 1

        settle_frames = int(max(1, cfg.unknown_settle_frames))
        if self._no_motion_frames < settle_frames:
            return []

        reference_gray = self._pre_motion_gray if self._pre_motion_gray is not None else self._static_gray
        if reference_gray is None:
            self._static_gray = gray.copy()
            return []
        if self._long_static_gray is None:
            self._long_static_gray = reference_gray.copy()
        interval_ref = self._get_interval_reference(now_ts, float(max(0.5, cfg.unknown_compare_interval_sec)))

        intensity_thr = int(max(0, min(255, cfg.unknown_diff_intensity_threshold)))
        grad_thr = int(max(0, min(255, cfg.unknown_grad_threshold)))
        diff_abs_pre = cv2.absdiff(gray, reference_gray)
        diff_abs_long = cv2.absdiff(gray, self._long_static_gray)
        diff_abs_interval = cv2.absdiff(gray, interval_ref)
        grad_cur = self._gradient_mag(gray)
        grad_ref = self._gradient_mag(reference_gray)
        grad_long = self._gradient_mag(self._long_static_gray)
        grad_interval = self._gradient_mag(interval_ref)
        grad_diff_pre = cv2.absdiff(grad_cur, grad_ref)
        grad_diff_long = cv2.absdiff(grad_cur, grad_long)
        grad_diff_interval = cv2.absdiff(grad_cur, grad_interval)
        mask_pre = (diff_abs_pre >= intensity_thr) & (grad_diff_pre >= grad_thr)
        mask_long = (diff_abs_long >= intensity_thr) & (grad_diff_long >= grad_thr)
        mask_interval = (diff_abs_interval >= intensity_thr) & (grad_diff_interval >= grad_thr)
        mask = np.where(mask_pre | mask_long | mask_interval, 255, 0).astype(np.uint8)
        if self._person_motion_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(self._person_motion_mask))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._static_gray = gray.copy()
            self._long_static_gray = cv2.addWeighted(self._long_static_gray, 0.95, gray, 0.05, 0)
            self._pre_motion_gray = None
            if self._person_motion_mask is not None:
                self._person_motion_mask.fill(0)
            return []

        frame_area = float(max(1, h * w))
        min_area = float(max(1, cfg.unknown_min_area_px))
        max_area = float(max(min_area, cfg.unknown_max_area_ratio * frame_area))
        min_fill = float(max(0.01, min(1.0, cfg.unknown_min_fill_ratio)))
        person_iou_thr = float(max(0.0, min(1.0, cfg.unknown_person_iou_threshold)))
        person_pad = int(max(0, cfg.unknown_person_exclusion_px))
        min_y = float(h) * float(max(0.0, min(1.0, cfg.unknown_min_y_ratio)))
        border_px = int(max(0, cfg.unknown_border_px))
        min_box_size = float(max(1, cfg.min_box_size_px))
        pseudo_conf = float(max(0.01, min(1.0, cfg.unknown_confidence)))
        expanded_person_boxes: list[tuple[float, float, float, float]] = []
        for p in persons:
            px1, py1, px2, py2 = p.bbox
            expanded_person_boxes.append(
                (
                    max(0.0, px1 - person_pad),
                    max(0.0, py1 - person_pad),
                    min(float(w), px2 + person_pad),
                    min(float(h), py2 + person_pad),
                )
            )

        candidate_boxes: list[tuple[float, float, float, float]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1f, y1f = x * sx, y * sy
            x2f, y2f = (x + bw) * sx, (y + bh) * sy
            area_full = max(0.0, (x2f - x1f) * (y2f - y1f))
            if area_full < min_area or area_full > max_area:
                continue
            bw_f, bh_f = x2f - x1f, y2f - y1f
            ar = bw_f / max(1.0, float(bh_f))
            if ar > 4.0 or ar < 0.2:
                continue
            if bw_f < min_box_size or bh_f < min_box_size:
                continue
            if x1f <= border_px or y1f <= border_px or x2f >= (w - border_px) or y2f >= (h - border_px):
                continue
            if y2f < min_y:
                continue
            roi = mask[y : y + bh, x : x + bw]
            if roi.size == 0:
                continue
            fill_ratio = float(cv2.countNonZero(roi)) / float(roi.size)
            if fill_ratio < min_fill:
                continue

            bbox = (x1f, y1f, x2f, y2f)
            if any(self._iou_tuple(bbox, d.bbox) >= person_iou_thr for d in known_objects):
                continue
            if any(self._iou_tuple(bbox, p.bbox) >= person_iou_thr for p in persons):
                continue
            if any(self._boxes_intersect(bbox, pb) for pb in expanded_person_boxes):
                continue

            candidate_boxes.append(bbox)
        if not candidate_boxes:
            return []

        aux_ids = self._assign_aux_ids(
            classes=[-1] * len(candidate_boxes),
            boxes_xyxy=[list(b) for b in candidate_boxes],
            h=h,
            w=w,
        )
        out: list[Detection] = []
        for tid, bbox in zip(aux_ids, candidate_boxes):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            out.append(
                Detection(
                    track_id=int(tid),
                    cls_id=-1,
                    cls_name="unknown",
                    confidence=pseudo_conf,
                    bbox=bbox,
                    centroid=(cx, cy),
                )
            )
        self._static_gray = gray.copy()
        self._long_static_gray = cv2.addWeighted(self._long_static_gray, 0.9, gray, 0.1, 0)
        self._pre_motion_gray = None
        if self._person_motion_mask is not None:
            self._person_motion_mask.fill(0)
        return out

    @staticmethod
    def _boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2

    @staticmethod
    def _gradient_mag(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        ax = cv2.convertScaleAbs(gx)
        ay = cv2.convertScaleAbs(gy)
        return cv2.addWeighted(ax, 0.5, ay, 0.5, 0)

    def _get_interval_reference(self, now_ts: float, interval_sec: float) -> np.ndarray:
        target_ts = now_ts - interval_sec
        selected = self._gray_history[0][1]
        for ts, g in self._gray_history:
            if ts <= target_ts:
                selected = g
            else:
                break
        min_keep_ts = now_ts - max(interval_sec * 2.0, 8.0)
        while len(self._gray_history) > 2 and self._gray_history[0][0] < min_keep_ts:
            self._gray_history.popleft()
        return selected


def _worker_main(cfg: ModelConfig, command_q: Queue, result_q: Queue) -> None:
    tracker = _LocalYoloTracker(cfg)
    shm_cache: dict[str, SharedMemory] = {}
    result_q.put({"type": "ready", "names": tracker.names})
    while True:
        msg = command_q.get()
        op = msg.get("op")
        req_id = int(msg.get("id", 0))
        try:
            if op == "infer":
                max_roi_count = msg.get("max_roi_count")
                if max_roi_count is not None:
                    max_roi_count = int(max_roi_count)
                priority = msg.get("priority")
                if msg.get("use_shm"):
                    shm_name = str(msg["shm_name"])
                    shape = tuple(msg["shape"])
                    dtype = np.dtype(str(msg["dtype"]))
                    shm = shm_cache.get(shm_name)
                    if shm is None:
                        shm = SharedMemory(name=shm_name)
                        shm_cache[shm_name] = shm
                    frame = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
                else:
                    frame = msg["frame"]
                result = tracker.infer(
                    frame,
                    max_roi_count=max_roi_count,
                    priority=priority,
                )
                result_q.put({"id": req_id, "ok": True, "result": result})
            elif op == "reload":
                tracker.reload(msg["cfg"])
                result_q.put({"id": req_id, "ok": True})
            elif op == "reset":
                tracker.reset()
                result_q.put({"id": req_id, "ok": True})
            elif op == "shutdown":
                result_q.put({"id": req_id, "ok": True})
                break
            elif op == "names":
                result_q.put({"id": req_id, "ok": True, "names": tracker.names})
            else:
                result_q.put({"id": req_id, "ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:
            result_q.put({"id": req_id, "ok": False, "error": str(exc)})
    for shm in shm_cache.values():
        try:
            shm.close()
        except Exception:
            pass


class _WorkerYoloTracker:
    """Тонкий RPC-клиент к отдельному Python inference worker."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._req_id = 0
        self._frame_shm: Optional[SharedMemory] = None
        self._frame_shm_size: int = 0
        ctx = get_context(cfg.worker_mp_start_method or "spawn")
        qsize = max(1, int(cfg.worker_queue_size))
        self._command_q: Queue = ctx.Queue(maxsize=qsize)
        self._result_q: Queue = ctx.Queue(maxsize=qsize)
        self._proc = ctx.Process(
            target=_worker_main,
            args=(cfg, self._command_q, self._result_q),
            name="yolo-inference-worker",
            daemon=True,
        )
        self._proc.start()
        self._names: dict[int, str] = {}
        self._wait_ready(float(cfg.worker_startup_timeout_sec))

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _wait_ready(self, timeout_sec: float) -> None:
        try:
            msg = self._result_q.get(timeout=timeout_sec)
        except queue.Empty as exc:
            raise RuntimeError("inference worker startup timeout") from exc
        if msg.get("type") != "ready":
            raise RuntimeError(f"inference worker failed to start: {msg}")
        names = msg.get("names") or {}
        self._names = {int(k): str(v) for k, v in names.items()}

    def _call(self, op: str, **payload):
        with self._lock:
            req_id = self._next_id()
            self._command_q.put({"id": req_id, "op": op, **payload}, timeout=self._cfg.worker_timeout_sec)
            while True:
                try:
                    msg = self._result_q.get(timeout=self._cfg.worker_timeout_sec)
                except queue.Empty as exc:
                    raise RuntimeError(f"inference worker timeout on op={op}") from exc
                if int(msg.get("id", -1)) != req_id:
                    continue
                if not msg.get("ok", False):
                    raise RuntimeError(msg.get("error", f"inference worker op={op} failed"))
                return msg

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    def infer(
        self,
        frame: np.ndarray,
        *,
        max_roi_count: Optional[int] = None,
        priority: Optional[str] = None,
    ) -> DetectionResult:
        use_shm = bool(self._cfg.worker_use_shared_memory)
        if use_shm:
            self._ensure_frame_shm(frame.nbytes)
            assert self._frame_shm is not None
            shm_view = np.ndarray(shape=frame.shape, dtype=frame.dtype, buffer=self._frame_shm.buf)
            shm_view[...] = frame
            msg = self._call(
                "infer",
                use_shm=True,
                shm_name=self._frame_shm.name,
                shape=list(frame.shape),
                dtype=str(frame.dtype),
                max_roi_count=max_roi_count,
                priority=priority,
            )
        else:
            msg = self._call(
                "infer",
                use_shm=False,
                frame=frame,
                max_roi_count=max_roi_count,
                priority=priority,
            )
        result = msg.get("result")
        if not isinstance(result, DetectionResult):
            raise RuntimeError("inference worker returned invalid payload")
        return result

    def _ensure_frame_shm(self, nbytes: int) -> None:
        if self._frame_shm is not None and self._frame_shm_size >= nbytes:
            return
        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
            finally:
                try:
                    self._frame_shm.unlink()
                except Exception:
                    pass
            self._frame_shm = None
            self._frame_shm_size = 0
        self._frame_shm = SharedMemory(create=True, size=nbytes)
        self._frame_shm_size = nbytes

    def reload(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._call("reload", cfg=cfg)
        names_msg = self._call("names")
        names = names_msg.get("names") or {}
        self._names = {int(k): str(v) for k, v in names.items()}

    def reset(self) -> None:
        self._call("reset")

    def shutdown(self) -> None:
        try:
            self._call("shutdown")
        except Exception:
            pass
        try:
            if self._proc.is_alive():
                self._proc.join(timeout=1.0)
        finally:
            if self._proc.is_alive():
                self._proc.kill()
        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
            finally:
                try:
                    self._frame_shm.unlink()
                except Exception:
                    pass
            self._frame_shm = None
            self._frame_shm_size = 0


class _RpcYoloTracker:
    """Клиент внешнего inference service (JSON-RPC over TCP)."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._req_id = 0
        self._frame_shm: Optional[SharedMemory] = None
        self._frame_shm_size: int = 0
        self._names: dict[int, str] = {}
        self._health()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _parse_addr(self) -> tuple[str, int]:
        raw = (self._cfg.inference_rpc_addr or "").strip()
        if ":" not in raw:
            raise RuntimeError("inference_rpc_addr must be host:port")
        host, port = raw.rsplit(":", 1)
        return host.strip(), int(port)

    def _rpc_call(self, op: str, **payload) -> dict:
        host, port = self._parse_addr()
        req = {"id": self._next_id(), "op": op, **payload}
        timeout = float(self._cfg.inference_rpc_timeout_sec)
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            line = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
            sock.sendall(line)
            chunks = bytearray()
            while True:
                b = sock.recv(1)
                if not b:
                    break
                if b == b"\n":
                    break
                chunks.extend(b)
        if not chunks:
            raise RuntimeError(f"inference rpc empty response op={op}")
        msg = json.loads(chunks.decode("utf-8"))
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("error", f"inference rpc op={op} failed"))
        return msg

    def _health(self) -> None:
        with self._lock:
            msg = self._rpc_call("health")
            names = msg.get("names") or {}
            self._names = {int(k): str(v) for k, v in names.items()}

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    def _ensure_frame_shm(self, nbytes: int) -> None:
        if self._frame_shm is not None and self._frame_shm_size >= nbytes:
            return
        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
            finally:
                try:
                    self._frame_shm.unlink()
                except Exception:
                    pass
            self._frame_shm = None
            self._frame_shm_size = 0
        self._frame_shm = SharedMemory(create=True, size=nbytes)
        self._frame_shm_size = nbytes

    def infer(
        self,
        frame: np.ndarray,
        *,
        max_roi_count: Optional[int] = None,
        priority: Optional[str] = None,
    ) -> DetectionResult:
        with self._lock:
            use_shm = bool(self._cfg.worker_use_shared_memory)
            if use_shm:
                self._ensure_frame_shm(frame.nbytes)
                assert self._frame_shm is not None
                shm_view = np.ndarray(shape=frame.shape, dtype=frame.dtype, buffer=self._frame_shm.buf)
                shm_view[...] = frame
                msg = self._rpc_call(
                    "infer_shm",
                    shm_name=self._frame_shm.name,
                    shape=list(frame.shape),
                    dtype=str(frame.dtype),
                    max_roi_count=max_roi_count,
                    priority=priority,
                )
            else:
                # Внешний RPC путь ориентирован на SHM, raw frame здесь не поддерживаем.
                raise RuntimeError("RPC inference currently requires worker_use_shared_memory=true")
            payload = msg.get("result") or {}
            return detection_result_from_dict(payload)

    def reload(self, cfg: ModelConfig) -> None:
        with self._lock:
            self._cfg = cfg
            msg = self._rpc_call("reload", cfg=cfg.__dict__)
            names = msg.get("names") or {}
            self._names = {int(k): str(v) for k, v in names.items()}

    def reset(self) -> None:
        with self._lock:
            self._rpc_call("reset")

    def shutdown(self) -> None:
        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
            finally:
                try:
                    self._frame_shm.unlink()
                except Exception:
                    pass
            self._frame_shm = None
            self._frame_shm_size = 0


class YoloTracker:
    """Совместимый фасад: local или separate worker inference backend."""

    def __init__(self, cfg: ModelConfig) -> None:
        if (cfg.inference_rpc_addr or "").strip():
            self._impl = _RpcYoloTracker(cfg)
        elif cfg.use_inference_worker:
            self._impl = _WorkerYoloTracker(cfg)
        else:
            self._impl = _LocalYoloTracker(cfg)

    @property
    def names(self) -> dict[int, str]:
        return self._impl.names

    def reload(self, cfg: ModelConfig) -> None:
        use_rpc = bool((cfg.inference_rpc_addr or "").strip())
        if use_rpc and not isinstance(self._impl, _RpcYoloTracker):
            self.shutdown()
            self._impl = _RpcYoloTracker(cfg)
            return
        if (not use_rpc) and isinstance(self._impl, _RpcYoloTracker):
            self.shutdown()
            if cfg.use_inference_worker:
                self._impl = _WorkerYoloTracker(cfg)
            else:
                self._impl = _LocalYoloTracker(cfg)
            return
        if (not use_rpc) and cfg.use_inference_worker and not isinstance(self._impl, _WorkerYoloTracker):
            self.shutdown()
            self._impl = _WorkerYoloTracker(cfg)
            return
        if (not use_rpc) and (not cfg.use_inference_worker) and not isinstance(self._impl, _LocalYoloTracker):
            self.shutdown()
            self._impl = _LocalYoloTracker(cfg)
            return
        self._impl.reload(cfg)

    def reset(self) -> None:
        self._impl.reset()

    def infer(
        self,
        frame: np.ndarray,
        *,
        max_roi_count: Optional[int] = None,
        priority: Optional[str] = None,
    ) -> DetectionResult:
        return self._impl.infer(frame, max_roi_count=max_roi_count, priority=priority)

    def shutdown(self) -> None:
        if hasattr(self._impl, "shutdown"):
            self._impl.shutdown()
