"""YOLO26 инференс через ultralytics (TensorRT .engine / .pt / .onnx).

YOLO26 — end-to-end (NMS-free): голова уже выдаёт финальные боксы (выход ONNX
[1,300,6] = x1,y1,x2,y2,conf,cls). Поэтому ручной NMS из `yolo_onnx.YoloOnnx`
здесь не нужен и сломал бы парсинг. Делегируем препроцесс/постпроцесс ultralytics,
а сами лишь раскладываем детекции на people / objects / weak_people — ровно тот же
контракт, что и у старого `YoloOnnx`, чтобы `worker.py` не менялся.

Один forward за кадр на НИЗКОМ пороге conf; деление high/low отдаём ByteTrack и
фильтрам воркера (как и раньше).
"""
from __future__ import annotations

import os
import sys

import numpy as np


def _pick_device() -> str:
    """0 (GPU) если CUDA доступна и не запрещена INTEGRA_FORCE_CPU; иначе 'cpu'."""
    if os.environ.get("INTEGRA_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class YoloUltra:
    def __init__(self, model_path: str, imgsz: int = 800, conf: float = 0.1,
                 iou: float = 0.45, person_class: int = 0, threads: int = 0):
        from ultralytics import YOLO

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.person_class = int(person_class)
        self.device = _pick_device()
        # .engine собран под FP16; для .pt/.onnx включаем half только на GPU.
        self.half = self.device != "cpu"

        self.model = YOLO(model_path, task="detect")
        # Прогрев + проверка устройства (engine привязан к GPU, на котором собран).
        warm = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(warm, imgsz=self.imgsz, conf=0.25, iou=self.iou,
                               half=self.half, device=self.device, verbose=False)
        except Exception as e:  # engine на чужом GPU / нет half — fallback на .pt CPU выше по стеку
            print(f"[pyinfer] yolo_ultra warmup failed: {e}", file=sys.stderr, flush=True)
            raise
        print(f"[pyinfer] yolo_ultra: model={os.path.basename(model_path)} "
              f"imgsz={self.imgsz} device={self.device} half={self.half}",
              file=sys.stderr, flush=True)

    def _predict_raw(self, bgr: np.ndarray, floor_conf: float):
        """Один проход. Возвращает (xyxy[N,4], cls[N] int, conf[N] float) в коорд.
        ИСХОДНОГО кадра (ultralytics сам ремапит из letterbox)."""
        res = self.model.predict(
            bgr, imgsz=self.imgsz, conf=float(max(0.01, floor_conf)),
            iou=self.iou, half=self.half, device=self.device, verbose=False)
        if not res:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        b = res[0].boxes
        if b is None or b.shape[0] == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        xyxy = b.xyxy.detach().cpu().numpy().astype(np.float32)
        cls = b.cls.detach().cpu().numpy().astype(np.int32)
        conf = b.conf.detach().cpu().numpy().astype(np.float32)
        return xyxy, cls, conf

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Nx5 [x1,y1,x2,y2,score] для класса person (orig coords)."""
        xyxy, cls, conf = self._predict_raw(bgr, self.conf)
        keep = (cls == self.person_class) & (conf >= self.conf)
        if not np.any(keep):
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate([xyxy[keep], conf[keep, None]], axis=1).astype(np.float32)

    def detect_all(self, bgr: np.ndarray, object_classes, object_conf: float,
                   person_floor_conf: float = 0.0):
        """Один проход → (persons Nx5, objects list[(x1,y1,x2,y2,score,cls_id)],
        weak_persons Nx5). Контракт идентичен yolo_onnx.YoloOnnx.detect_all."""
        floors = [self.conf, float(object_conf) if object_classes else self.conf]
        if person_floor_conf and 0.0 < person_floor_conf < self.conf:
            floors.append(person_floor_conf)
        xyxy, cls, conf = self._predict_raw(bgr, min(floors))

        if xyxy.shape[0] == 0:
            return np.zeros((0, 5), np.float32), [], np.zeros((0, 5), np.float32)

        is_person = cls == self.person_class

        # люди (для трекинга — порог self.conf)
        pmask = is_person & (conf >= self.conf)
        persons = (np.concatenate([xyxy[pmask], conf[pmask, None]], axis=1).astype(np.float32)
                   if np.any(pmask) else np.zeros((0, 5), np.float32))

        # слабые люди (для подавления фантомов — пониженный порог, < self.conf)
        weak_persons = np.zeros((0, 5), np.float32)
        if person_floor_conf and 0.0 < person_floor_conf < self.conf:
            wmask = is_person & (conf >= person_floor_conf)
            if np.any(wmask):
                weak_persons = np.concatenate(
                    [xyxy[wmask], conf[wmask, None]], axis=1).astype(np.float32)

        # объекты — «оставляемые» COCO-классы
        objects = []
        if object_classes:
            want = np.isin(cls, np.array(list(object_classes), dtype=cls.dtype))
            omask = want & (conf >= float(object_conf))
            for i in np.nonzero(omask)[0]:
                bx = xyxy[i]
                objects.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]),
                                float(conf[i]), int(cls[i])))
        return persons, objects, weak_persons
