"""YOLOv11 инференс через onnxruntime (CPU). Только класс person.

Низкий порог conf на детекции (отдаём кандидатов вплоть до low_thresh) — деление
high/low делает ByteTrack. NMS — numpy/cv2.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import onnxruntime as ort


def _enable_torch_cuda_dlls() -> None:
    """onnxruntime-gpu ищет cudnn/cublas DLL по PATH. В виндах CUDA-libs идут
    в комплекте с torch (`torch/lib/`) — добавляем их к загрузчику DLL до
    инициализации сессии."""
    if sys.platform != "win32":
        return
    try:
        import torch  # noqa: F401
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.submodule_search_locations:
            return
        torch_lib = os.path.join(spec.submodule_search_locations[0], "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except (FileNotFoundError, OSError):
                pass
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


def _pick_providers() -> list:
    """CUDA EP если доступен и не запрещён INTEGRA_FORCE_CPU; иначе CPU."""
    force_cpu = os.environ.get("INTEGRA_FORCE_CPU", "").lower() in ("1", "true", "yes")
    avail = ort.get_available_providers()
    out = []
    if not force_cpu and "CUDAExecutionProvider" in avail:
        _enable_torch_cuda_dlls()
        out.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }))
    out.append("CPUExecutionProvider")
    return out


def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    """Resize с сохранением пропорций + паддинг. Возвращает (img, r, dw, dh)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - nw) / 2, (new_shape - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top


class YoloOnnx:
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.1,
                 iou: float = 0.45, person_class: int = 0, threads: int = 0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
        providers = _pick_providers()
        # CPU EP — сжимаем arena (для GPU режима arena полезна, оставляем дефолт)
        if all((p == "CPUExecutionProvider" if isinstance(p, str) else p[0] == "CPUExecutionProvider")
               for p in providers):
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=providers
        )
        active = self.session.get_providers()
        print(f"[pyinfer] yolo providers active: {active}", file=sys.stderr, flush=True)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.person_class = int(person_class)

    def _infer(self, bgr: np.ndarray):
        """Один forward-pass. Возвращает (xyxy[N,4] в коорд. исходного кадра,
        cls_ids[N], cls_conf[N]) для всех кандидатов с conf >= низкого порога."""
        img, r, dw, dh = letterbox(bgr, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None]

        out = self.session.run(None, {self.input_name: blob})[0]  # [1,84,8400]
        out = np.squeeze(out, axis=0)
        if out.shape[0] < out.shape[1]:
            out = out.T  # -> [8400,84]

        boxes_xywh = out[:, :4]
        scores_all = out[:, 4:]
        cls_ids = np.argmax(scores_all, axis=1)
        cls_conf = scores_all[np.arange(scores_all.shape[0]), cls_ids]

        floor = min(self.conf, 0.05)
        keep = cls_conf >= floor
        boxes_xywh = boxes_xywh[keep]
        cls_ids = cls_ids[keep]
        cls_conf = cls_conf[keep].astype(np.float32)
        if boxes_xywh.shape[0] == 0:
            return np.zeros((0, 4), np.float32), cls_ids, cls_conf

        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - w / 2.0 - dw) / r
        y1 = (cy - h / 2.0 - dh) / r
        x2 = (cx + w / 2.0 - dw) / r
        y2 = (cy + h / 2.0 - dh) / r

        H, W = bgr.shape[:2]
        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W - 1)
        y2 = np.clip(y2, 0, H - 1)
        xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        return xyxy, cls_ids, cls_conf

    @staticmethod
    def _nms_subset(xyxy, confs, mask, conf_thr, iou_thr):
        """NMS по подмножеству детекций (mask). Возвращает индексы (в исходном массиве)."""
        idx_pool = np.nonzero(mask & (confs >= conf_thr))[0]
        if idx_pool.size == 0:
            return []
        b = xyxy[idx_pool]
        wh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(wh, confs[idx_pool].tolist(), conf_thr, iou_thr)
        if len(keep) == 0:
            return []
        return idx_pool[np.array(keep).reshape(-1)].tolist()

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Nx5 [x1,y1,x2,y2,score] для класса person (orig coords)."""
        xyxy, cls_ids, cls_conf = self._infer(bgr)
        if xyxy.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)
        idxs = self._nms_subset(xyxy, cls_conf, cls_ids == self.person_class, self.conf, self.iou)
        if not idxs:
            return np.zeros((0, 5), dtype=np.float32)
        res = np.concatenate([xyxy[idxs], cls_conf[idxs, None]], axis=1)
        return res.astype(np.float32)

    def detect_all(self, bgr: np.ndarray, object_classes, object_conf: float,
                   person_floor_conf: float = 0.0):
        """Один проход → (persons Nx5, objects list[(x1,y1,x2,y2,score,cls_id)],
        weak_persons Nx5).

        object_classes — множество COCO-id «оставляемых» предметов.
        weak_persons — person-детекции на ПОНИЖЕННОМ пороге `person_floor_conf`
        (включают сильных). Нужны не для трекинга, а как маска ПОДАВЛЕНИЯ объект-
        кандидатов: сидящий/ближний человек даёт слабый person-отклик (0.05–0.3),
        которого не хватает на трек ByteTrack, но достаточно чтобы погасить его
        силуэт-фантом. Реальный предмет person-отклика не даёт → не подавляется.
        """
        xyxy, cls_ids, cls_conf = self._infer(bgr)
        if xyxy.shape[0] == 0:
            return np.zeros((0, 5), np.float32), [], np.zeros((0, 5), np.float32)

        # люди (для трекинга — порог self.conf)
        pidx = self._nms_subset(xyxy, cls_conf, cls_ids == self.person_class, self.conf, self.iou)
        persons = (np.concatenate([xyxy[pidx], cls_conf[pidx, None]], axis=1).astype(np.float32)
                   if pidx else np.zeros((0, 5), np.float32))

        # слабые люди (для подавления — пониженный порог)
        weak_persons = np.zeros((0, 5), np.float32)
        if person_floor_conf > 0.0 and person_floor_conf < self.conf:
            widx = self._nms_subset(xyxy, cls_conf, cls_ids == self.person_class,
                                    person_floor_conf, self.iou)
            if widx:
                weak_persons = np.concatenate(
                    [xyxy[widx], cls_conf[widx, None]], axis=1).astype(np.float32)

        # объекты — NMS отдельно по каждому классу
        objects = []
        if object_classes:
            want = np.isin(cls_ids, np.array(list(object_classes), dtype=cls_ids.dtype))
            for c in object_classes:
                oidx = self._nms_subset(xyxy, cls_conf, want & (cls_ids == c), object_conf, self.iou)
                for i in oidx:
                    b = xyxy[i]
                    objects.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                    float(cls_conf[i]), int(c)))
        return persons, objects, weak_persons
