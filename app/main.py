"""FastAPI-сервер приложения.

Этот модуль отвечает только за web-слой:
- REST API для управления плеером и настройками;
- MJPEG-стрим текущего кадра для тега <img>;
- WebSocket для push-обновлений статуса и событий.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
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

# Кеш тяжёлых системных метрик (nvidia-smi / psutil), чтобы UI не дёргал GPU на каждый опрос.
_SYSTEM_METRICS_LOCK = threading.Lock()
_SYSTEM_METRICS_CACHE: dict[str, Any] = {"t_mono": 0.0, "data": None}
_SYSTEM_METRICS_TTL_SEC = 1.0

# EMA / пик RSS для графика видеоаналитики (основной процесс + analytics_extra_pids).
_ANALYTICS_MEM_LOCK = threading.Lock()
_ANALYTICS_MEM_STATE: dict[str, Any] = {
    "rss_ema_bytes": None,
    "rss_peak_window_bytes": None,
    "rss_samples": [],  # (t_mono, rss_sum) для UI-истории
}


def _get_system_metrics_cached() -> dict:
    now = time.perf_counter()
    with _SYSTEM_METRICS_LOCK:
        cached = _SYSTEM_METRICS_CACHE["data"]
        t0 = float(_SYSTEM_METRICS_CACHE["t_mono"])
        if cached is not None and (now - t0) < _SYSTEM_METRICS_TTL_SEC:
            return cached
    data = _system_metrics()
    with _SYSTEM_METRICS_LOCK:
        _SYSTEM_METRICS_CACHE["data"] = data
        _SYSTEM_METRICS_CACHE["t_mono"] = time.perf_counter()
    return data


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
    """Простой pub/sub-хаб в памяти процесса для браузерных клиентов."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        # Запоминаем event loop, чтобы можно было рассылать из других потоков.
        self._loop = loop

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        # Отправляем сообщение всем активным клиентам.
        # "Умершие" сокеты (ошибка отправки) удаляем из пула.
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
        # Безопасная отправка из не-async контекста (например, из worker-потока).
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
        # Поднимаем периодическую рассылку статуса в WebSocket.
        loop = asyncio.get_running_loop()
        hub.attach(loop)
        loop.create_task(_status_pump(hub, pipeline))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Корректно завершаем фоновые компоненты.
        pipeline.shutdown()
        logger.close()

    # ------------------------------------------------------------ static UI
    # Отдача SPA-страницы и статических ресурсов.

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
    # Управление воспроизведением, настройками и журналом.

    @app.get("/api/info")
    async def _info() -> dict:
        return pipeline.info()

    @app.get("/api/metrics")
    async def _metrics() -> dict:
        return {
            "process": _process_analytics_metrics(cfg_store, pipeline),
            "system": _get_system_metrics_cached(),
            "pipeline": pipeline.metrics(),
        }

    @app.get("/api/files")
    async def _files() -> dict:
        videos: list[dict] = []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Поддерживаем только видео-расширения, которые реально ожидаем в пайплайне.
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
        # Разрешаем и абсолютный путь, и имя файла внутри data/.
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
        # Возвращаем полный снимок конфигурации для формы UI.
        return cfg_store.snapshot()

    @app.put("/api/settings")
    async def _put_settings(patch: SettingsPatch) -> dict:
        body = patch.model_dump(exclude_none=True)
        snapshot = cfg_store.update(body)
        # Если менялся блок model — перегружаем веса.
        # Иначе достаточно обновить пороги анализатора "на лету".
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
    # MJPEG для живого видео и WS для событий/метрик.

    @app.get("/video_feed")
    def _video_feed():
        boundary = "frame"

        def gen():
            last_sent_id = -1
            while True:
                rendered = pipeline.latest_rendered()
                if rendered is None:
                    # Пока кадра нет, просто подождем.
                    time.sleep(0.02)
                    continue
                if rendered.frame_id == last_sent_id:
                    # Если новый кадр не появился, не отправляем дубль.
                    time.sleep(0.005)
                    continue
                last_sent_id = rendered.frame_id
                jpeg = rendered.jpeg
                # Отдаем только новый кадр, чтобы не засорять канал дубликатами.
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
                time.sleep(0.001)

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
                # Входящие сообщения не используем (кроме ping), сокет нужен
                # как push-канал сервер -> клиент.
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
    """Периодически рассылает статус пайплайна и снимок треков."""
    while True:
        try:
            await hub.broadcast({"type": "status", "info": pipeline.info()})
        except Exception:
            pass
        await asyncio.sleep(0.5)


app = create_app()


def _process_metrics() -> dict:
    rss_bytes = None
    vms_bytes = None
    cpu_percent = None
    private_bytes = None
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        mem = p.memory_info()
        rss_bytes = int(mem.rss)
        vms_bytes = int(mem.vms)
        cpu_percent = float(p.cpu_percent(interval=0.0))
        if hasattr(mem, "private"):
            private_bytes = int(mem.private)  # type: ignore[attr-defined]
    except Exception:
        pass
    return {
        "pid": os.getpid(),
        "rss_bytes": rss_bytes,
        "vms_bytes": vms_bytes,
        "private_bytes": private_bytes,
        "cpu_percent": cpu_percent,
    }


def _process_analytics_metrics(cfg_store: ConfigStore, pipeline: VideoPipeline) -> dict:
    base = _process_metrics()
    rss_sum = int(base.get("rss_bytes") or 0)
    extra = list(cfg_store.cfg.pipeline.analytics_extra_pids or [])
    try:
        import psutil  # type: ignore

        for pid in extra:
            try:
                rss_sum += int(psutil.Process(int(pid)).memory_info().rss)
            except Exception:
                pass
    except Exception:
        pass
    cuda_alloc = None
    cuda_reserved = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            cuda_alloc = int(torch.cuda.memory_allocated())
            cuda_reserved = int(torch.cuda.memory_reserved())
    except Exception:
        pass
    pm = pipeline.metrics()
    prev_bytes = int(pm.get("buffers", {}).get("forensic_ring_bytes") or 0)
    preview_bytes_est = prev_bytes
    now = time.perf_counter()
    alpha = 0.12
    ema_f = float(rss_sum)
    peak_5s = float(rss_sum)
    hist_slice: list[tuple[float, float]] = []
    with _ANALYTICS_MEM_LOCK:
        ema = _ANALYTICS_MEM_STATE["rss_ema_bytes"]
        if ema is None:
            ema_f = float(rss_sum)
        else:
            ema_f = float(ema) * (1.0 - alpha) + float(rss_sum) * alpha
        _ANALYTICS_MEM_STATE["rss_ema_bytes"] = ema_f
        win: list[float] = list(_ANALYTICS_MEM_STATE.get("rss_peak_window_bytes") or [])
        win.append(float(rss_sum))
        if len(win) > 120:
            win = win[-120:]
        _ANALYTICS_MEM_STATE["rss_peak_window_bytes"] = win
        peak_5s = max(win[-20:]) if win else float(rss_sum)
        samples: list[tuple[float, float]] = list(_ANALYTICS_MEM_STATE.get("rss_samples") or [])
        samples.append((now, ema_f))
        if len(samples) > 180:
            samples = samples[-180:]
        _ANALYTICS_MEM_STATE["rss_samples"] = samples
        hist_slice = samples[-90:]
    t0 = hist_slice[0][0] if hist_slice else now
    rss_history = [{"t": round(t - t0, 3), "rss_ema_bytes": v} for t, v in hist_slice]
    return {
        **base,
        "rss_analytics_sum_bytes": rss_sum,
        "rss_ema_bytes": round(ema_f, 0),
        "rss_peak_recent_bytes": int(peak_5s),
        "cuda_memory_allocated_bytes": cuda_alloc,
        "cuda_memory_reserved_bytes": cuda_reserved,
        "preview_memory_bytes_est": preview_bytes_est,
        "rss_history": rss_history,
    }


def _system_metrics() -> dict:
    cpu_percent = None
    ram_used = None
    ram_total = None
    ram_percent = None
    try:
        import psutil  # type: ignore

        cpu_percent = float(psutil.cpu_percent(interval=0.0))
        vm = psutil.virtual_memory()
        ram_used = int(vm.used)
        ram_total = int(vm.total)
        ram_percent = float(vm.percent)
    except Exception:
        pass
    return {
        "cpu_percent": cpu_percent,
        "ram_used_bytes": ram_used,
        "ram_total_bytes": ram_total,
        "ram_percent": ram_percent,
        "gpu": _gpu_metrics(),
    }


def _gpu_metrics() -> dict:
    # Для production-like GPU-нод предпочитаем nvidia-smi как системный источник.
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5, check=True)
    except Exception:
        return {
            "available": False,
            "name": None,
            "util_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "memory_percent": None,
        }
    lines = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    if not lines:
        return {
            "available": False,
            "name": None,
            "util_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "memory_percent": None,
        }
    # Берем первую GPU для базового NOC-дашборда.
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 4:
        return {
            "available": False,
            "name": None,
            "util_percent": None,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "memory_percent": None,
        }
    name = parts[0]
    util = _to_float(parts[1])
    mem_used_mb = _to_float(parts[2])
    mem_total_mb = _to_float(parts[3])
    used_bytes = int(mem_used_mb * 1024 * 1024) if mem_used_mb is not None else None
    total_bytes = int(mem_total_mb * 1024 * 1024) if mem_total_mb is not None else None
    mem_percent = None
    if mem_used_mb is not None and mem_total_mb and mem_total_mb > 0:
        mem_percent = float((mem_used_mb / mem_total_mb) * 100.0)
    return {
        "available": True,
        "name": name,
        "util_percent": util,
        "memory_used_bytes": used_bytes,
        "memory_total_bytes": total_bytes,
        "memory_percent": mem_percent,
    }


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except Exception:
        return None
