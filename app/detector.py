"""Обертка над Ultralytics YOLOv11 + встроенный трекинг BoT-SORT.

Инференс вызывается по одному кадру через `model.track(..., persist=True)`.
Так пайплайн полностью контролирует источник кадров (seek/pause/play).

Модель загружается один раз и закрепляется на выбранном device.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from .config import ModelConfig


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


class YoloTracker:
    """Потокобезопасная обертка для single-worker сценария.

    Трекер хранит внутреннее состояние между вызовами, поэтому доступ
    сериализуется через lock.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._model: Optional[YOLO] = None
        # Если GPU недоступен, автоматически откатываемся на CPU.
        self._device = cfg.device if torch.cuda.is_available() else "cpu"
        self._half = cfg.half and self._device.startswith("cuda")
        self._aux_next_id = 1_000_000
        self._aux_tracks: dict[int, tuple[int, tuple[float, float], tuple[float, float, float, float], float]] = {}
        if self._device.startswith("cuda"):
            # Небольшие оптимизации CUDA для стабильного fps.
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        self._load()

    def _load(self) -> None:
        # При имени официального чекпойнта Ultralytics сам скачает веса.
        weights = self._cfg.weights
        if not Path(weights).exists() and "/" not in weights and "\\" not in weights:
            # Здесь ничего не делаем: библиотека сама скачает официальный чекпойнт.
            pass
        self._model = YOLO(weights)
        # Перенос на устройство делаем один раз.
        try:
            self._model.to(self._device)
        except Exception:
            pass

    def reload(self, cfg: ModelConfig) -> None:
        with self._lock:
            # Полная перезагрузка нужна, если меняли веса/девайс/half/tracker.
            self._cfg = cfg
            self._device = cfg.device if torch.cuda.is_available() else "cpu"
            self._half = cfg.half and self._device.startswith("cuda")
            self._load()

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

    def infer(self, frame: np.ndarray) -> DetectionResult:
        cfg = self._cfg
        # Грубый фильтр по именам классов для подавления ложных тревог
        # (мебель, животные, электроника и т.п.).
        excluded = {str(n).strip().lower() for n in cfg.excluded_class_names if str(n).strip()}
        class_min_conf = {
            str(k).strip().lower(): float(v) for k, v in (cfg.class_min_conf or {}).items() if str(k).strip()
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

        # Второй проход по зоне пола: повышает recall для объектов на полу.
        if cfg.floor_roi_enabled and frame.shape[0] > 0:
            floor_det = self._infer_floor_roi(frame, cfg, excluded, class_min_conf, names)
            if floor_det:
                detections = self._merge_detections(detections, floor_det, cfg.merge_iou_threshold)
        return DetectionResult(detections, persons, inference_ms)

    def _infer_floor_roi(
        self,
        frame: np.ndarray,
        cfg: ModelConfig,
        excluded: set[str],
        class_min_conf: dict[str, float],
        names: dict[int, str],
    ) -> list[Detection]:
        h, w = frame.shape[:2]
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
