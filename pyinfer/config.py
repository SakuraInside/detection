"""Чтение config.json и маппинг в параметры воркера."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _f(d: dict, key: str, default: float) -> float:
    v = d.get(key)
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _i(d: dict, key: str, default: int) -> int:
    v = d.get(key)
    try:
        return int(v) if v is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


@dataclass
class TrackerCfg:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: float = 30.0


@dataclass
class AnalyzerCfg:
    static_displacement_px: float = 7.0
    static_window_sec: float = 3.0
    abandon_time_sec: float = 15.0
    owner_proximity_px: float = 180.0
    owner_left_sec: float = 5.0
    disappear_grace_sec: float = 4.0
    min_object_area_px: float = 600.0
    centroid_history_maxlen: int = 72
    max_active_tracks: int = 256
    person_class_id: int = 0
    # class-agnostic объекты
    use_frame_diff_detector: bool = True
    frame_diff_min_region_area_px: int = 100
    tracker_iou_match_threshold: float = 0.15
    tracker_max_missed_frames: int = 15
    # Верхняя граница площади объект-региона (доля кадра). Регион крупнее — это не
    # «оставленная вещь», а ближний план / силуэт крупного сидящего человека → дроп.
    # По умолчанию ВЫКЛ (0): приоритет recall — близко к камере предмет крупный.
    max_object_area_ratio: float = 0.0
    # Подавление объект-регионов по «потерянным» person-трекам ByteTrack: сидящий
    # человек мерцает в YOLO; столько кадров держим его предсказанный бокс как зону
    # подавления, чтобы силуэт не стал ложным объектом. 0 — выключить.
    suppress_lost_person_frames: int = 20
    # зона игнорирования детекций объектов (норм. координаты), [x1,y1,x2,y2] или None
    ignore_norm_rect: tuple | None = None


# COCO-классы «оставляемых» предметов (yolo11 обучен на COCO).
# 24 backpack, 25 umbrella, 26 handbag, 28 suitcase, 39 bottle, 40 wine glass,
# 41 cup, 45 bowl, 63 laptop, 64 mouse, 66 keyboard, 67 cell phone, 73 book,
# 76 scissors, 77 teddy bear, 79 toothbrush.
DEFAULT_OBJECT_CLASSES = [24, 25, 26, 28, 39, 40, 41, 45, 63, 64, 66, 67, 73, 76, 77]


@dataclass
class Config:
    model_path: str = "models/yolo11s.onnx"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.45
    person_class: int = 0
    camera_id: str = "main"
    onnx_threads: int = 0  # 0 => onnxruntime default
    # YOLO как второй источник предметов (ловит статичные объекты, которые MOG2
    # пропускает: положены до warmup / постановка под окклюзией).
    object_detect_classes: tuple = tuple(DEFAULT_OBJECT_CLASSES)
    object_detect_conf: float = 0.35
    # Порог «слабых» person-детекций для ПОДАВЛЕНИЯ объект-фантомов (сидящие/ближний
    # план, которых YOLO видит слабо). 0 — выключить. Ниже self.conf(=low_thresh).
    # По умолчанию ВЫКЛ: приоритет recall (может душить реально оставленный предмет).
    person_suppress_conf: float = 0.0
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    analyzer: AnalyzerCfg = field(default_factory=AnalyzerCfg)


def load_config(root: Path) -> Config:
    cfg_path = root / "config.json"
    raw: dict = {}
    if cfg_path.is_file():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    model = raw.get("model", {}) or {}
    analyzer = raw.get("analyzer", {}) or {}
    native = raw.get("native_analytics", {}) or {}
    trk = model.get("tracker", {}) or {}

    # модель: предпочитаем .onnx из native_analytics.model_path, иначе models/yolo11s.onnx
    model_path = native.get("model_path") or "models/yolo11s.onnx"
    if not os.path.isabs(model_path):
        model_path = str((root / model_path).resolve())

    c = Config()
    c.model_path = model_path
    c.imgsz = _i(model, "imgsz", _i(native, "input_size", 640))
    c.conf = _f(model, "conf", 0.35)
    c.iou = _f(model, "iou", 0.45)
    c.person_class = _i(model, "person_class", 0)
    c.camera_id = (native.get("camera_id") or "main")
    c.onnx_threads = _i(raw.get("pipeline", {}) or {}, "py_onnx_threads", 0)

    odc = model.get("detect_object_classes")
    if isinstance(odc, list) and all(isinstance(x, (int, float)) for x in odc):
        c.object_detect_classes = tuple(int(x) for x in odc)
    elif odc == [] or model.get("disable_object_detect"):
        c.object_detect_classes = tuple()  # явное отключение YOLO-предметов
    c.object_detect_conf = _f(model, "detect_object_conf", 0.35)
    c.person_suppress_conf = _f(model, "person_suppress_conf", 0.08)

    c.tracker = TrackerCfg(
        high_thresh=_f(trk, "high_thresh", 0.5),
        low_thresh=_f(trk, "low_thresh", 0.1),
        new_thresh=_f(trk, "new_thresh", 0.6),
        match_thresh=_f(trk, "match_thresh", 0.8),
        track_buffer=_i(trk, "track_buffer", 30),
        frame_rate=_f(model, "bytetrack_frame_rate", 30.0),
    )

    rect = analyzer.get("ignore_detection_norm_rect")
    ignore_rect = None
    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in rect)
            if x2 > x1 and y2 > y1:
                ignore_rect = (x1, y1, x2, y2)
        except (TypeError, ValueError):
            ignore_rect = None

    c.analyzer = AnalyzerCfg(
        static_displacement_px=_f(analyzer, "static_displacement_px", 7.0),
        static_window_sec=_f(analyzer, "static_window_sec", 3.0),
        abandon_time_sec=_f(analyzer, "abandon_time_sec", 15.0),
        owner_proximity_px=_f(analyzer, "owner_proximity_px", 180.0),
        owner_left_sec=_f(analyzer, "owner_left_sec", 5.0),
        disappear_grace_sec=_f(analyzer, "disappear_grace_sec", 4.0),
        min_object_area_px=_f(analyzer, "min_object_area_px", 600.0),
        centroid_history_maxlen=_i(analyzer, "centroid_history_maxlen", 72),
        max_active_tracks=_i(analyzer, "max_active_tracks", 256),
        person_class_id=c.person_class,
        use_frame_diff_detector=bool(analyzer.get("use_frame_diff_detector", True)),
        frame_diff_min_region_area_px=_i(analyzer, "frame_diff_min_region_area_px", 100),
        tracker_iou_match_threshold=_f(analyzer, "tracker_iou_match_threshold", 0.15),
        tracker_max_missed_frames=_i(analyzer, "tracker_max_missed_frames", 15),
        max_object_area_ratio=_f(analyzer, "max_object_area_ratio", 0.10),
        suppress_lost_person_frames=_i(analyzer, "suppress_lost_person_frames", 20),
        ignore_norm_rect=ignore_rect,
    )
    return c
