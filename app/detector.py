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

    def infer(self, frame: np.ndarray) -> DetectionResult:
        cfg = self._cfg
        # Грубый фильтр по именам классов для подавления ложных тревог
        # (мебель, животные, электроника и т.п.).
        excluded = {str(n).strip().lower() for n in cfg.excluded_class_names if str(n).strip()}
        with self._lock:
            assert self._model is not None
            # Пустой object_classes => разрешаем все классы (кроме person
            # в отдельной ветке), чтобы не терять "неочевидные" объекты.
            classes = None if not cfg.object_classes else list(set(cfg.object_classes + [cfg.person_class]))
            # persist=True сохраняет внутреннее состояние трекера между кадрами.
            results = self._model.track(
                source=frame,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
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
        if r.boxes is None or r.boxes.id is None:
            return DetectionResult([], [], inference_ms)

        ids = r.boxes.id.int().cpu().tolist()
        cls = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.float().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().tolist()
        for tid, c, p, box in zip(ids, cls, conf, xyxy):
            cls_name = names.get(int(c), str(c))
            if cls_name.strip().lower() in excluded:
                # Убираем заведомо "шумные" классы из пользовательского blacklist.
                continue
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
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
        return DetectionResult(detections, persons, inference_ms)
