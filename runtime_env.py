"""Centralized runtime environment defaults."""

from __future__ import annotations

import os


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


RUNTIME_INGEST_ADDR = env_str("RUNTIME_INGEST_ADDR", "127.0.0.1:7878")
RUNTIME_CONTROL_ADDR = env_str("RUNTIME_CONTROL_ADDR", "127.0.0.1:7879")
INFERENCE_RPC_ADDR = env_str("INFERENCE_RPC_ADDR", "127.0.0.1:7788")

RUNTIME_CORE_TIMEOUT_SEC = env_float("RUNTIME_CORE_TIMEOUT_SEC", 0.05)
RUNTIME_CONTROL_TIMEOUT_SEC = env_float("RUNTIME_CONTROL_TIMEOUT_SEC", 0.01)
INFERENCE_RPC_TIMEOUT_SEC = env_float("INFERENCE_RPC_TIMEOUT_SEC", 20.0)

APP_HOST_DEFAULT = env_str("APP_HOST", "127.0.0.1")
APP_PORT_DEFAULT = env_int("APP_PORT", 8000)

# TCP `video-bridge` (Rust): python pipeline.rust_video_bridge_addr
VIDEO_BRIDGE_ADDR = env_str("VIDEO_BRIDGE_ADDR", "127.0.0.1:9876")
