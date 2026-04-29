"""Best-effort bridge: Python pipeline -> Rust runtime-core."""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuntimeBridgeConfig:
    addr: str = ""
    timeout_sec: float = 0.05
    control_addr: str = ""
    control_timeout_sec: float = 0.01


@dataclass
class RuntimeControlDecision:
    infer: bool
    target_interval: int = 1
    priority: str = "normal"
    max_roi_count: int = 1
    overload_level: int = 0
    latency_ema_ms: float = 0.0


class RuntimeCoreBridge:
    def __init__(self, cfg: RuntimeBridgeConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None

    def update(self, cfg: RuntimeBridgeConfig) -> None:
        with self._lock:
            changed = (
                (cfg.addr != self._cfg.addr)
                or (cfg.timeout_sec != self._cfg.timeout_sec)
                or (cfg.control_addr != self._cfg.control_addr)
                or (cfg.control_timeout_sec != self._cfg.control_timeout_sec)
            )
            self._cfg = cfg
            if changed:
                self._close_locked()

    def send(self, payload: dict) -> None:
        if not self._cfg.addr:
            return
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            try:
                sock = self._ensure_connected_locked()
                sock.sendall(line)
            except Exception:
                self._close_locked()

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def should_infer(
        self, camera_id: str, frame_id: int, default_interval: int
    ) -> Optional[RuntimeControlDecision]:
        addr = (self._cfg.control_addr or "").strip()
        if not addr:
            return None
        timeout = max(0.001, float(self._cfg.control_timeout_sec))
        try:
            host, port = self._parse_addr(addr)
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                payload = {
                    "type": "should_infer",
                    "camera_id": camera_id,
                    "frame_id": int(frame_id),
                    "default_interval": max(1, int(default_interval)),
                }
                sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                data = bytearray()
                while True:
                    b = sock.recv(1)
                    if not b or b == b"\n":
                        break
                    data.extend(b)
            if not data:
                return None
            msg = json.loads(data.decode("utf-8"))
            if msg.get("type") != "should_infer_result":
                return None
            return RuntimeControlDecision(
                infer=bool(msg.get("infer", False)),
                target_interval=max(1, int(msg.get("target_interval", default_interval))),
                priority=str(msg.get("priority", "normal")),
                max_roi_count=max(1, int(msg.get("max_roi_count", 1))),
                overload_level=max(0, int(msg.get("overload_level", 0))),
                latency_ema_ms=float(msg.get("latency_ema_ms", 0.0)),
            )
        except Exception:
            return None

    def _ensure_connected_locked(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        host, port = self._parse_addr(self._cfg.addr)
        sock = socket.create_connection((host, port), timeout=max(0.01, float(self._cfg.timeout_sec)))
        sock.settimeout(max(0.01, float(self._cfg.timeout_sec)))
        self._sock = sock
        return sock

    @staticmethod
    def _parse_addr(addr: str) -> tuple[str, int]:
        if ":" not in addr:
            raise RuntimeError("runtime_core_addr must be host:port")
        host, port = addr.rsplit(":", 1)
        return host.strip(), int(port)

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
