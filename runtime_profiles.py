"""Runtime profile presets for migration stages."""

from __future__ import annotations

import json
from pathlib import Path

from runtime_env import (
    INFERENCE_RPC_ADDR,
    INFERENCE_RPC_TIMEOUT_SEC,
    RUNTIME_CONTROL_ADDR,
    RUNTIME_CONTROL_TIMEOUT_SEC,
    RUNTIME_CORE_TIMEOUT_SEC,
    RUNTIME_INGEST_ADDR,
    VIDEO_BRIDGE_ADDR,
)


def apply_profile(config_path: Path, profile: str, low_memory: bool = False) -> None:
    profile = profile.strip().lower()
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    model = data.setdefault("model", {})
    pipeline = data.setdefault("pipeline", {})

    # Shared defaults for migration runtime.
    model["worker_use_shared_memory"] = True
    model["worker_queue_size"] = int(model.get("worker_queue_size", 2) or 2)
    model["worker_timeout_sec"] = float(model.get("worker_timeout_sec", 20.0) or 20.0)
    model["worker_startup_timeout_sec"] = float(model.get("worker_startup_timeout_sec", 90.0) or 90.0)
    model["worker_mp_start_method"] = "spawn"
    model["inference_rpc_timeout_sec"] = float(
        model.get("inference_rpc_timeout_sec", INFERENCE_RPC_TIMEOUT_SEC) or INFERENCE_RPC_TIMEOUT_SEC
    )
    pipeline["runtime_core_timeout_sec"] = float(
        pipeline.get("runtime_core_timeout_sec", RUNTIME_CORE_TIMEOUT_SEC) or RUNTIME_CORE_TIMEOUT_SEC
    )
    pipeline["runtime_control_timeout_sec"] = float(
        pipeline.get("runtime_control_timeout_sec", RUNTIME_CONTROL_TIMEOUT_SEC) or RUNTIME_CONTROL_TIMEOUT_SEC
    )

    if profile == "legacy":
        model["use_inference_worker"] = False
        model["inference_rpc_addr"] = ""
        pipeline["runtime_core_addr"] = ""
        pipeline["runtime_control_addr"] = ""
        pipeline["rust_video_bridge_addr"] = ""
    elif profile == "rust-video":
        # Декод в процессе video-bridge (Rust); YOLO + Analyzer в Python.
        model["use_inference_worker"] = False
        model["inference_rpc_addr"] = ""
        pipeline["runtime_core_addr"] = ""
        pipeline["runtime_control_addr"] = ""
        pipeline["rust_video_bridge_addr"] = VIDEO_BRIDGE_ADDR
    elif profile == "hybrid":
        model["use_inference_worker"] = True
        model["inference_rpc_addr"] = ""
        pipeline["runtime_core_addr"] = RUNTIME_INGEST_ADDR
        pipeline["runtime_control_addr"] = RUNTIME_CONTROL_ADDR
        pipeline["rust_video_bridge_addr"] = ""
    elif profile == "external":
        model["use_inference_worker"] = True
        model["inference_rpc_addr"] = INFERENCE_RPC_ADDR
        pipeline["runtime_core_addr"] = RUNTIME_INGEST_ADDR
        pipeline["runtime_control_addr"] = RUNTIME_CONTROL_ADDR
        pipeline["rust_video_bridge_addr"] = ""
    else:
        raise ValueError(f"unsupported profile: {profile}")

    if low_memory:
        # Сильнее ужимаем RAM/CPU: меньшая модель, ниже imgsz, реже ROI и меньше буферов.
        model["weights"] = "models/yolo11s.pt"
        model["imgsz"] = min(int(model.get("imgsz", 576) or 576), 576)
        model["floor_roi_enabled"] = True
        model["floor_roi_imgsz"] = min(int(model.get("floor_roi_imgsz", 480) or 480), 480)
        model["table_roi_imgsz"] = min(int(model.get("table_roi_imgsz", 512) or 512), 512)
        model["table_roi_every_n_frames"] = max(2, int(model.get("table_roi_every_n_frames", 2) or 2))
        pipeline["detect_every_n_frames"] = max(2, int(pipeline.get("detect_every_n_frames", 2) or 2))
        pipeline["render_every_n_frames"] = max(2, int(pipeline.get("render_every_n_frames", 2) or 2))
        pipeline["decode_queue"] = min(2, int(pipeline.get("decode_queue", 2) or 2))
        pipeline["result_queue"] = min(2, int(pipeline.get("result_queue", 2) or 2))
        pipeline["frame_pool_size"] = min(2, int(pipeline.get("frame_pool_size", 2) or 2))
        pipeline["forensic_ring_max"] = 0

    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
