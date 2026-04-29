"""Standalone JSON-RPC inference service for Rust control-plane integration.

Protocol: newline-delimited JSON over TCP.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from .config import ModelConfig
from .detector import (
    _LocalYoloTracker,
    detection_result_to_dict,
)


def _model_cfg_from_payload(payload: dict[str, Any]) -> ModelConfig:
    cfg = ModelConfig()
    for k, v in payload.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    tracker: _LocalYoloTracker,
) -> None:
    peer = writer.get_extra_info("peername")
    print(f"[inference_service] client connected: {peer}")
    try:
        while True:
            raw = await reader.readline()
            if not raw:
                break
            try:
                req = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                await _send(writer, {"id": None, "ok": False, "error": f"invalid json: {exc}"})
                continue
            req_id = req.get("id")
            op = req.get("op")
            try:
                if op == "health":
                    await _send(writer, {"id": req_id, "ok": True, "status": "ready", "names": tracker.names})
                elif op == "reset":
                    tracker.reset()
                    await _send(writer, {"id": req_id, "ok": True})
                elif op == "reload":
                    cfg = _model_cfg_from_payload(req.get("cfg") or {})
                    tracker.reload(cfg)
                    await _send(writer, {"id": req_id, "ok": True, "names": tracker.names})
                elif op == "infer_shm":
                    shm_name = str(req["shm_name"])
                    shape = tuple(int(x) for x in req["shape"])
                    dtype = np.dtype(str(req["dtype"]))
                    max_roi_count = req.get("max_roi_count")
                    if max_roi_count is not None:
                        max_roi_count = int(max_roi_count)
                    priority = req.get("priority")
                    shm = SharedMemory(name=shm_name)
                    try:
                        frame = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
                        result = tracker.infer(
                            frame,
                            max_roi_count=max_roi_count,
                            priority=priority,
                        )
                    finally:
                        shm.close()
                    await _send(writer, {"id": req_id, "ok": True, "result": detection_result_to_dict(result)})
                else:
                    await _send(writer, {"id": req_id, "ok": False, "error": f"unknown op: {op}"})
            except Exception as exc:
                await _send(writer, {"id": req_id, "ok": False, "error": str(exc)})
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[inference_service] client disconnected: {peer}")


async def _send(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    await writer.drain()


async def run_server(host: str, port: int, cfg: ModelConfig) -> None:
    tracker = _LocalYoloTracker(cfg)
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, tracker),
        host=host,
        port=port,
    )
    sockets = server.sockets or []
    addrs = ", ".join(str(s.getsockname()) for s in sockets)
    print(f"[inference_service] listening on {addrs}")
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone YOLO inference JSON-RPC service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7788, type=int)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = ModelConfig()
    if args.weights:
        cfg.weights = args.weights
    if args.device:
        cfg.device = args.device
    cfg.use_inference_worker = False
    cfg.worker_use_shared_memory = True
    asyncio.run(run_server(args.host, args.port, cfg))


if __name__ == "__main__":
    main()
