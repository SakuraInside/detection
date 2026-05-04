"""TCP-клиент к Rust `video-bridge`: декод в Rust, numpy BGR в Python для YOLO + Analyzer."""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np


class RustVideoBridge:
    """Один TCP-сессия: после open() читать кадры через read_frame()."""

    def __init__(self, addr: str) -> None:
        host, _, port_s = addr.rpartition(":")
        if not host or not port_s:
            raise ValueError(f"invalid rust_video_bridge_addr: {addr!r}, expected host:port")
        self._addr = (host.strip(), int(port_s))
        self._sock: Optional[socket.socket] = None
        self.handshake: dict[str, Any] = {}
        self._payload_buf: Optional[bytearray] = None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def open_video(
        self,
        path: str,
        seek_frame: Optional[int] = None,
        *,
        prefer_hw_decode: bool = False,
    ) -> dict[str, Any]:
        """Открыть файл на стороне bridge (новое TCP-подключение)."""
        self.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect(self._addr)
        cmd: dict[str, Any] = {
            "cmd": "open",
            "path": str(Path(path).resolve()),
            "prefer_hw_decode": bool(prefer_hw_decode),
        }
        if seek_frame is not None:
            cmd["seek_frame"] = int(seek_frame)
        s.sendall((json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
        line = self._readline(s)
        h = json.loads(line)
        if h.get("type") != "handshake" or not h.get("ok", False):
            s.close()
            raise RuntimeError(h.get("message") or "handshake failed")
        self._sock = s
        self.handshake = h
        iw = int(h.get("width") or 0)
        ih = int(h.get("height") or 0)
        self._payload_buf = bytearray(max(0, iw * ih * 3)) if iw > 0 and ih > 0 else None
        return h

    def read_frame(self) -> Optional[tuple[np.ndarray, float, int]]:
        """Один кадр BGR uint8 (view в переиспользуемый буфер) или None при EOF/обрыве.

        Следующий read_frame перезапишет буфер — до следующего вызова нужно скопировать кадр
        (например, в пул пайплайна).
        """
        if self._sock is None:
            return None
        try:
            line = self._readline(self._sock)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return None
        if not line.strip():
            return None
        meta = json.loads(line)
        ln_b = self._recv_exact(self._sock, 4)
        ln = int(struct.unpack("<I", ln_b)[0])
        h = int(meta["height"])
        w = int(meta["width"])
        if self._payload_buf is None or len(self._payload_buf) < ln:
            self._payload_buf = bytearray(ln)
        self._recv_exact_into(self._sock, self._payload_buf, ln)
        arr = np.frombuffer(self._payload_buf, dtype=np.uint8, count=ln).reshape((h, w, 3))
        return arr, float(meta["pos_ms"]), int(meta["frame_id"])

    @staticmethod
    def _readline(sock: socket.socket) -> str:
        buf = bytearray()
        while True:
            ch = sock.recv(1)
            if not ch:
                break
            if ch == b"\n":
                break
            buf.extend(ch)
        return buf.decode("utf-8", errors="replace")

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        parts: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise OSError("short read")
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    @staticmethod
    def _recv_exact_into(sock: socket.socket, dst: bytearray, n: int) -> None:
        mv = memoryview(dst)[:n]
        pos = 0
        while pos < n:
            got = sock.recv_into(mv[pos:], n - pos)
            if got == 0:
                raise OSError("short read")
            pos += got
