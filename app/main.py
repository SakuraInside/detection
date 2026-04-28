"""FastAPI server: REST control, MJPEG video, WebSocket events."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ConfigStore
from .logger_db import EventLogger
from .pipeline import VideoPipeline


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
LOGS_DIR = ROOT / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
DB_PATH = LOGS_DIR / "events.db"
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"


class OpenRequest(BaseModel):
    path: str


class SeekRequest(BaseModel):
    frame: int


class SettingsPatch(BaseModel):
    model: dict[str, Any] | None = None
    pipeline: dict[str, Any] | None = None
    analyzer: dict[str, Any] | None = None
    ui: dict[str, Any] | None = None


class WSHub:
    """In-process pub/sub for browser clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            dead: list[WebSocket] = []
            for ws in list(self._clients):
                try:
                    await ws.send_text(json.dumps(message, ensure_ascii=False))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def broadcast_threadsafe(self, message: dict) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


def create_app() -> FastAPI:
    app = FastAPI(title="Integra-LOST", version="0.1.0")
    cfg_store = ConfigStore(CONFIG_PATH)
    logger = EventLogger(DB_PATH)
    hub = WSHub()

    pipeline = VideoPipeline(
        config_store=cfg_store,
        logger=logger,
        snapshots_dir=SNAPSHOTS_DIR,
        on_event=lambda payload: hub.broadcast_threadsafe({"type": "event", **payload}),
    )

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        hub.attach(loop)
        loop.create_task(_status_pump(hub, pipeline))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        pipeline.shutdown()
        logger.close()

    # ------------------------------------------------------------ static UI

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def _index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/logs/snapshots/{name}")
    async def _snapshot(name: str) -> FileResponse:
        path = SNAPSHOTS_DIR / name
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------ REST API

    @app.get("/api/info")
    async def _info() -> dict:
        return pipeline.info()

    @app.get("/api/files")
    async def _files() -> dict:
        videos: list[dict] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        exts = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}
        for p in sorted(DATA_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                videos.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    }
                )
        return {"data_dir": str(DATA_DIR), "files": videos}

    @app.post("/api/open")
    async def _open(req: OpenRequest) -> dict:
        # Allow either an absolute path or a name in data/.
        candidate = Path(req.path)
        if not candidate.is_absolute():
            candidate = DATA_DIR / req.path
        try:
            return pipeline.open(str(candidate))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/play")
    async def _play() -> dict:
        pipeline.play()
        return {"playing": True}

    @app.post("/api/pause")
    async def _pause() -> dict:
        pipeline.pause()
        return {"playing": False}

    @app.post("/api/seek")
    async def _seek(req: SeekRequest) -> dict:
        pipeline.seek(req.frame)
        return {"ok": True, "frame": req.frame}

    @app.get("/api/settings")
    async def _get_settings() -> dict:
        return cfg_store.snapshot()

    @app.put("/api/settings")
    async def _put_settings(patch: SettingsPatch) -> dict:
        body = patch.model_dump(exclude_none=True)
        snapshot = cfg_store.update(body)
        # If model section changed, reload heavy weights; otherwise just push
        # the new analyzer thresholds to the running FSM.
        if "model" in body:
            pipeline.reload_model()
        else:
            pipeline.update_settings()
        return snapshot

    @app.get("/api/events")
    async def _events(limit: int = 200) -> dict:
        return {"events": logger.recent(limit=max(1, min(limit, 1000)))}

    @app.delete("/api/events")
    async def _clear_events() -> dict:
        logger.clear()
        return {"ok": True}

    # ------------------------------------------------------------- streams

    @app.get("/video_feed")
    def _video_feed():
        boundary = "frame"

        def gen():
            last_sent_id = -1
            while True:
                jpeg = pipeline.latest_jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                # We always send the freshest frame; deduping via frame_id is
                # implicit in `latest_rendered`.
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                time.sleep(1.0 / 60.0)

        headers = {"Cache-Control": "no-store", "Pragma": "no-cache"}
        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers=headers,
        )

    @app.websocket("/ws")
    async def _ws(ws: WebSocket) -> None:
        await ws.accept()
        await hub.add(ws)
        try:
            await ws.send_text(json.dumps({"type": "hello", "info": pipeline.info()}))
            while True:
                # We don't expect inbound messages; just keep the socket alive.
                msg = await ws.receive_text()
                if msg == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.remove(ws)

    return app


async def _status_pump(hub: WSHub, pipeline: VideoPipeline) -> None:
    """Periodic broadcast of pipeline status / track snapshot."""
    while True:
        try:
            await hub.broadcast({"type": "status", "info": pipeline.info()})
        except Exception:
            pass
        await asyncio.sleep(0.5)


app = create_app()
