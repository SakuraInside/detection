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
    # ID классов COCO, которые считаем "целевыми предметами" (+ человек).
    # Пустой список => учитываем все классы (кроме исключенных в excluded_class_names).
    object_classes: list[int] = field(
        default_factory=list
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
    # ID класса "cup" в COCO.
    cup_class: int = 41
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
            "cup": 0.0015,
            "wine glass": 0.0015,
            "bowl": 0.003,
        }
    )
    # Дополнительный проход детекции по зоне пола (нижняя часть кадра).
    floor_roi_enabled: bool = True
    floor_roi_y_ratio: float = 0.55
    floor_roi_imgsz: int = 960
    floor_roi_conf: float = 0.04
    # Порог IoU для дедупликации "full frame" и "floor ROI" детекций.
    merge_iou_threshold: float = 0.45
    # Запускать YOLO в отдельном Python worker-процессе (thin inference layer).
    # Это снижает влияние модели на основной runtime и упрощает дальнейший вынос оркестрации.
    use_inference_worker: bool = False
    # Ограничение IPC-очередей запросов/ответов к worker.
    worker_queue_size: int = 2
    # Таймаут RPC-запросов к worker (сек).
    worker_timeout_sec: float = 20.0
    # Таймаут ожидания запуска worker и загрузки модели (сек).
    worker_startup_timeout_sec: float = 90.0
    # Старт-метод multiprocessing.
    # На Windows/кроссплатформенно безопаснее "spawn".
    worker_mp_start_method: str = "spawn"
    # Передавать кадры в worker через shared memory, а не через pickle очереди.
    worker_use_shared_memory: bool = True
    # Адрес внешнего inference service (JSON-RPC), например "127.0.0.1:7788".
    # Если задан, модель обслуживается внешним процессом, а не локальным worker.
    inference_rpc_addr: str = ""
    # Таймаут вызова внешнего inference service (сек).
    inference_rpc_timeout_sec: float = 20.0
    # Дополнительный проход cup-only в ROI стола.
    table_roi_enabled: bool = True
    # ROI стола в относительных координатах [0..1].
    table_roi_x1: float = 0.12
    table_roi_y1: float = 0.14
    table_roi_x2: float = 0.50
    table_roi_y2: float = 0.66
    # Параметры инференса cup-only на ROI.
    table_roi_imgsz: int = 1280
    table_roi_conf: float = 0.01
    table_roi_iou: float = 0.50
    # Детектить ROI не на каждом кадре для экономии FPS.
    table_roi_every_n_frames: int = 2
    # Минимальная ширина/высота bbox для учета детекции.
    min_box_size_px: int = 20
    # Пониженный минимум bbox для отдельных маленьких классов.
    min_box_size_by_class: dict[str, int] = field(
        default_factory=lambda: {
            "cup": 12,
            "wine glass": 12,
            "bowl": 12,
        }
    )
    # Слой "неизвестный предмет" поверх YOLO (по появлению/движению в сцене).
    unknown_layer_enabled: bool = False
    # Сколько первых кадров использовать только для адаптации фона.
    unknown_warmup_frames: int = 45
    # Порог пиксельного движения для маски "область в движении".
    unknown_motion_threshold: int = 18
    # Минимальная площадь движущейся области.
    unknown_motion_min_area_px: int = 400
    # Сколько кадров ждать после остановки движения перед сравнением before/after.
    unknown_settle_frames: int = 8
    # Интервал в секундах для сравнения текущего кадра с прошлым.
    unknown_compare_interval_sec: float = 5.0
    # Порог изменения яркости (before/after).
    unknown_diff_intensity_threshold: int = 20
    # Порог изменения градиента (before/after).
    unknown_grad_threshold: int = 25
    # Порог бинаризации foreground mask от MOG2.
    unknown_fg_threshold: int = 220
    # Минимальная площадь кандидата "unknown" в пикселях.
    unknown_min_area_px: int = 250
    # Максимальная площадь кандидата "unknown" (защита от "вспышек" на весь кадр).
    unknown_max_area_ratio: float = 0.12
    # Минимальная "плотность" маски внутри bbox (доля foreground-пикселей).
    unknown_min_fill_ratio: float = 0.18
    # Если overlap с person выше порога — считаем, что это человек, не объект.
    unknown_person_iou_threshold: float = 0.15
    # Дополнительный отступ вокруг bbox человека для подавления ложных unknown.
    unknown_person_exclusion_px: int = 40
    # Обрабатываем unknown только ниже этой доли высоты кадра.
    unknown_min_y_ratio: float = 0.30
    # Игнорируем unknown вплотную к границам кадра (частый шум).
    unknown_border_px: int = 8
    # Псевдо-confidence для unknown, чтобы Analyzer/UI отображали объект.
    unknown_confidence: float = 0.3
    # Доля полного разрешения для reference/motion (0.15–1.0). Меньше -> меньше RAM и CPU.
    unknown_reference_scale: float = 0.25
    # Сколько lowres gray хранить для интервального сравнения (не full-frame).
    unknown_gray_history_maxlen: int = 24


@dataclass
class PipelineConfig:
    # Размер очередей между потоками декодирования и обработки.
    # Меньше значения -> меньше пиковая RAM (меньше полных кадров в ожидании).
    decode_queue: int = 1
    result_queue: int = 1
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
    # --- Preview path (снижение RAM/CPU): отдельно от аналитики ---
    # 0 = MJPEG в полном разрешении; иначе ограничить длинную сторону кадра (px) перед оверлеем и JPEG.
    preview_max_long_edge: int = 960
    # Качество JPEG для preview (ниже -> меньше размер буфера в сети и чуть меньше работы кодека).
    preview_jpeg_quality: int = 55
    # Минимальный интервал между перекодированием preview (мс); 0 = без ограничения.
    # Снижает частоту аллокаций сжатого буфера и нагрузку на CPU.
    preview_min_interval_ms: float = 100.0
    # Не ставить кадр в очередь рендера, если MJPEG всё равно переиспользует старый JPEG
    # (тот же интервал/шаг render_every). Освобождает буфер пула без ожидания render-потока.
    preview_skip_redundant_enqueue: bool = True
    # Число переиспользуемых полноразмерных RGB-буферов (0 = без пула, буфер из cap.read).
    # Пул ограничивает одновременно «живые» полные кадры и даёт обратное давление декодеру.
    frame_pool_size: int = 3
    # Длинная сторона JPEG снимка тревоги (0 = без даунскейла перед записью на диск).
    snapshot_max_long_edge: int = 1280
    # Качество JPEG для снимков abandoned/disappeared.
    snapshot_jpeg_quality: int = 82
    # Декод видео во внешнем процессе `video-bridge` (Rust). Пусто = OpenCV в Python.
    # Запуск: cargo run --manifest-path video-bridge/Cargo.toml -- --listen 127.0.0.1:9876
    rust_video_bridge_addr: str = ""
    # Адрес Rust runtime-core ingest bridge (JSON line protocol), например "127.0.0.1:7878".
    runtime_core_addr: str = ""
    # Таймаут отправки метаданных в runtime-core (сек).
    runtime_core_timeout_sec: float = 0.05
    # Адрес control-plane scheduler в runtime-core, например "127.0.0.1:7879".
    runtime_control_addr: str = ""
    # Таймаут запроса решения should_infer (сек).
    runtime_control_timeout_sec: float = 0.01
    # Кольцо последних JPEG preview для диагностики (0 = выкл). Удерживает только сжатые байты.
    forensic_ring_max: int = 0
    # Пороги графика RSS в UI (байты).
    memory_chart_warning_bytes: int = 805306368
    memory_chart_critical_bytes: int = 1006632960
    # Ограничение частоты кодирования preview (0 = без лимита).
    preview_encode_max_fps: float = 15.0
    # Дополнительные PID для суммирования RSS аналитики (video-bridge, worker).
    analytics_extra_pids: list[int] = field(default_factory=list)


@dataclass
class AnalyzerConfig:
    # Длина истории центроидов на трек (меньше -> меньше RAM на активные треки).
    centroid_history_maxlen: int = 64
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
