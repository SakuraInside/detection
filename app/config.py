"""Конфигурация приложения с поддержкой горячего обновления из JSON.

Веб-интерфейс опирается на структуру, описанную в этом файле.
Чтобы добавить новый параметр, обычно достаточно:
1) расширить dataclass ниже;
2) добавить поле в `config.json`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    # Путь к весам модели детектора.
    weights: str = "yolo11x.pt"
    # Размер входного изображения для инференса.
    imgsz: int = 640
    # Устройство вычислений: "cpu" или "cuda:0".
    device: str = "cuda:0"
    # FP16 ускоряет инференс на GPU (если поддерживается).
    half: bool = True
    # Порог уверенности и IoU для фильтрации детекций.
    conf: float = 0.35
    iou: float = 0.5
    # Конфиг трекера Ultralytics.
    tracker: str = "botsort.yaml"
    # ID классов COCO, которые считаем "потенциально оставляемыми предметами" (+ человек).
    object_classes: list[int] = field(
        default_factory=lambda: [24, 26, 28, 39, 41, 63, 64, 65, 66, 67, 73, 76]
    )
    # Дополнительный blacklist по имени класса для подавления известных ложных срабатываний.
    excluded_class_names: list[str] = field(
        default_factory=lambda: [
            "chair",
            "couch",
            "bed",
            "dining table",
            "potted plant",
            "toilet",
            "cat",
            "dog",
            "bird",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ]
    )
    # ID класса "person" в COCO.
    person_class: int = 0
    # Адаптивный confidence-фильтр по вертикали кадра:
    # верхняя часть (дальняя зона) обычно содержит меньшие объекты.
    # Для нее разрешаем более низкий confidence.
    upper_region_y_ratio: float = 0.62
    min_conf_upper: float = 0.08
    min_conf_lower: float = 0.20
    # Дополнительное послабление у нижней кромки кадра (частично обрезанные объекты).
    bottom_region_y_ratio: float = 0.88
    min_conf_bottom: float = 0.12
    # Послабление для bbox, касающихся границ кадра.
    border_relax_px: int = 24
    min_conf_border: float = 0.10
    # Классовые минимальные пороги confidence (по имени класса YOLO).
    # Нужны, когда отдельный класс (например bottle) стабильно теряется.
    class_min_conf: dict[str, float] = field(
        default_factory=lambda: {
            "bottle": 0.03,
        }
    )
    # Дополнительный проход детекции по зоне пола (нижняя часть кадра).
    floor_roi_enabled: bool = True
    floor_roi_y_ratio: float = 0.55
    floor_roi_imgsz: int = 960
    floor_roi_conf: float = 0.04
    # Порог IoU для дедупликации "full frame" и "floor ROI" детекций.
    merge_iou_threshold: float = 0.45


@dataclass
class PipelineConfig:
    # Размер очередей между потоками декодирования и обработки.
    decode_queue: int = 4
    result_queue: int = 4
    # Запускать детекцию на каждом N-м кадре.
    detect_every_n_frames: int = 1
    # Ограничение частоты декодирования.
    target_fps: int = 30
    # Рисовать оверлей (рамки, подписи) поверх кадра.
    render_overlay: bool = True
    # Пытаемся включить аппаратное ускорение декодирования (если доступно).
    prefer_hw_decode: bool = True
    # Качество JPEG для web-стрима. Ниже -> меньше CPU и сеть.
    jpeg_quality: int = 80
    # Кодировать JPEG не для каждого кадра (ускоряет рендер на слабом CPU).
    render_every_n_frames: int = 1


@dataclass
class AnalyzerConfig:
    # Порог смещения и окно времени для признания объекта статичным.
    static_displacement_px: float = 8.0
    static_window_sec: float = 3.0
    # Через сколько секунд без владельца поднимать тревогу "abandoned".
    abandon_time_sec: float = 15.0
    # Радиус поиска владельца и задержка "владелец ушел".
    owner_proximity_px: float = 180.0
    owner_left_sec: float = 5.0
    # Задержка перед событием "disappeared", чтобы отсечь кратковременные пропуски трека.
    disappear_grace_sec: float = 4.0
    # Минимальная площадь объекта в пикселях (фильтр шума).
    min_object_area_px: float = 600.0
    # Доля присутствия в окне наблюдения для устойчивых решений FSM.
    presence_ratio_threshold: float = 0.7


@dataclass
class UIConfig:
    # Включить звуковой сигнал тревоги в интерфейсе.
    alarm_sound: bool = True
    # Показывать детекции класса person.
    show_persons: bool = True
    # Показывать track_id возле боксов.
    show_track_ids: bool = True
    # Рисовать текстовые подписи (выключение снижает CPU на рендере).
    show_labels: bool = True


@dataclass
class AppConfig:
    # Общая структура конфигурации, которую читает/пишет ConfigStore.
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _merge(target: Any, source: dict[str, Any]) -> None:
    """Сливает только известные поля, игнорируя лишние (прямая совместимость)."""
    for key, value in source.items():
        # Неизвестные поля пропускаем, чтобы старый код не падал на новом config.json.
        if not hasattr(target, key):
            continue
        cur = getattr(target, key)
        # Рекурсивно сливаем вложенные dataclass-секции.
        if hasattr(cur, "__dataclass_fields__") and isinstance(value, dict):
            _merge(cur, value)
        else:
            # Примитивные поля (или списки) заменяем напрямую.
            setattr(target, key, value)


class ConfigStore:
    """Потокобезопасное хранилище конфигурации с сохранением на диск."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # RLock позволяет безопасно вызывать вложенные операции под тем же замком.
        self._lock = threading.RLock()
        # Инициализируем значения по умолчанию, затем пытаемся загрузить с диска.
        self._cfg = AppConfig()
        if path.exists():
            try:
                self._cfg = self._load()
            except Exception as exc:
                # При ошибке чтения остаемся на дефолтной конфигурации.
                print(f"[config] failed to load {path}: {exc}; using defaults")

    def _load(self) -> AppConfig:
        # Читаем JSON и накладываем только поддерживаемые поля на дефолтный шаблон.
        with self._path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = AppConfig()
        _merge(cfg, data)
        return cfg

    @property
    def cfg(self) -> AppConfig:
        return self._cfg

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            # Возвращаем "снимок" для безопасной передачи наружу.
            return asdict(self._cfg)

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            # Применяем частичное обновление, затем сразу сохраняем.
            _merge(self._cfg, patch)
            self._save()
            return asdict(self._cfg)

    def _save(self) -> None:
        # Запись через временный файл делает сохранение атомарным и снижает риск порчи.
        tmp = self._path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self._cfg), fh, indent=2, ensure_ascii=False)
        # Атомарная подмена целевого файла после успешной записи.
        tmp.replace(self._path)
