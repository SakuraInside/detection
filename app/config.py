"""Runtime configuration with hot-reload from JSON.

The whole web UI binds to the shape declared here, so adding a new tunable
parameter only requires extending the dataclasses below and `config.json`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    weights: str = "yolo11x.pt"
    imgsz: int = 640
    device: str = "cuda:0"
    half: bool = True
    conf: float = 0.35
    iou: float = 0.5
    tracker: str = "botsort.yaml"
    # COCO ids of "things that can be left behind" + person.
    object_classes: list[int] = field(
        default_factory=lambda: [24, 26, 28, 39, 41, 63, 64, 65, 66, 67, 73, 76]
    )
    person_class: int = 0


@dataclass
class PipelineConfig:
    decode_queue: int = 4
    result_queue: int = 4
    detect_every_n_frames: int = 1
    target_fps: int = 30
    render_overlay: bool = True


@dataclass
class AnalyzerConfig:
    static_displacement_px: float = 8.0
    static_window_sec: float = 3.0
    abandon_time_sec: float = 15.0
    owner_proximity_px: float = 180.0
    owner_left_sec: float = 5.0
    disappear_grace_sec: float = 4.0
    min_object_area_px: float = 600.0
    presence_ratio_threshold: float = 0.7


@dataclass
class UIConfig:
    alarm_sound: bool = True
    show_persons: bool = True
    show_track_ids: bool = True


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _merge(target: Any, source: dict[str, Any]) -> None:
    """Apply only known fields, ignore extras (forward compat)."""
    for key, value in source.items():
        if not hasattr(target, key):
            continue
        cur = getattr(target, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(value, dict):
            _merge(cur, value)
        else:
            setattr(target, key, value)


class ConfigStore:
    """Thread-safe wrapper that persists changes to disk."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._cfg = AppConfig()
        if path.exists():
            try:
                self._cfg = self._load()
            except Exception as exc:
                print(f"[config] failed to load {path}: {exc}; using defaults")

    def _load(self) -> AppConfig:
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
            return asdict(self._cfg)

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            _merge(self._cfg, patch)
            self._save()
            return asdict(self._cfg)

    def _save(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self._cfg), fh, indent=2, ensure_ascii=False)
        tmp.replace(self._path)
