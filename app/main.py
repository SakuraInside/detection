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

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
    # (t_mono, total_rss, pipeline_rss, inference_rss); inference=0 если YOLO в основном процессе.
    "rss_samples_v2": [],
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
    stream_id: str | None = None


class SeekRequest(BaseModel):
    frame: int


class StreamRequest(BaseModel):
    stream_id: str


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
    hub = WSHub()
    streams_lock = threading.RLock()
    streams: dict[str, dict[str, Any]] = {}

    def _norm_stream_id(raw: str | None) -> str:
        sid = (raw or "main").strip() or "main"
        if not sid.replace("-", "_").isalnum():
            raise HTTPException(400, "stream_id должен содержать только буквы/цифры/_/-")
        return sid[:64]

    def _create_stream(stream_id: str) -> dict[str, Any]:
        sid = _norm_stream_id(stream_id)
        snap_dir = SNAPSHOTS_DIR / sid
        db_path = LOGS_DIR / f"events_{sid}.db"
        logger = EventLogger(db_path)
        pipeline = VideoPipeline(
            config_store=cfg_store,
            logger=logger,
            snapshots_dir=snap_dir,
            on_event=lambda payload, s=sid: hub.broadcast_threadsafe(
                {"type": "event", "stream_id": s, **payload}
            ),
        )
        slot = {"id": sid, "pipeline": pipeline, "logger": logger}
        streams[sid] = slot
        return slot

    def _get_slot(stream_id: str | None, create: bool = True) -> dict[str, Any]:
        sid = _norm_stream_id(stream_id)
        with streams_lock:
            slot = streams.get(sid)
            if slot is None and create:
                slot = _create_stream(sid)
            if slot is None:
                raise HTTPException(404, f"stream {sid} not found")
            return slot

    _get_slot("main", create=True)

    @app.on_event("startup")
    async def _startup() -> None:
        # Поднимаем периодическую рассылку статуса в WebSocket.
        loop = asyncio.get_running_loop()
        hub.attach(loop)
        loop.create_task(_status_pump(hub, streams, streams_lock))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Корректно завершаем фоновые компоненты.
        with streams_lock:
            slots = list(streams.values())
        for slot in slots:
            try:
                slot["pipeline"].shutdown()
            except Exception:
                pass
            try:
                slot["logger"].close()
            except Exception:
                pass

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
    async def _info(stream_id: str = "main") -> dict:
        slot = _get_slot(stream_id)
        return slot["pipeline"].info()

    @app.get("/api/metrics")
    async def _metrics(stream_id: str = "main") -> dict:
        slot = _get_slot(stream_id)
        pipeline = slot["pipeline"]
        return {
            "process": _process_analytics_metrics(cfg_store, pipeline),
            "system": _get_system_metrics_cached(),
            "pipeline": pipeline.metrics(),
        }

    @app.get("/api/streams")
    async def _streams() -> dict:
        with streams_lock:
            items = list(streams.items())
        base_proc = _process_metrics()
        main_rss = int(base_proc.get("rss_bytes") or 0)
        share = int(main_rss / max(1, len(items)))
        out: list[dict[str, Any]] = []
        for sid, slot in items:
            pipe = slot["pipeline"]
            info = pipe.info()
            pm = pipe.metrics()
            worker_pid = pm.get("inference_worker_pid")
            worker_rss = 0
            if worker_pid:
                try:
                    import psutil  # type: ignore

                    worker_rss = int(psutil.Process(int(worker_pid)).memory_info().rss)
                except Exception:
                    worker_rss = 0
            buffers = pm.get("buffers") or {}
            frame_pool = int(buffers.get("frame_pool_bytes") or 0)
            forensic = int(buffers.get("forensic_ring_bytes") or 0)
            out.append(
                {
                    "stream_id": sid,
                    "loaded": bool(info.get("loaded")),
                    "playing": bool(info.get("playing")),
                    "video_path": info.get("video_path"),
                    "memory": {
                        "estimated_total_bytes": int(share + worker_rss + frame_pool + forensic),
                        "main_process_share_bytes": share,
                        "worker_rss_bytes": worker_rss or None,
                        "frame_pool_bytes": frame_pool,
                        "forensic_ring_bytes": forensic,
                    },
                }
            )
        return {"streams": out}

    @app.post("/api/streams")
    async def _create_stream_api(req: StreamRequest) -> dict:
        sid = _norm_stream_id(req.stream_id)
        with streams_lock:
            if sid in streams:
                return {"ok": True, "stream_id": sid, "created": False}
            _create_stream(sid)
        return {"ok": True, "stream_id": sid, "created": True}

    @app.delete("/api/streams/{stream_id}")
    async def _delete_stream_api(stream_id: str) -> dict:
        sid = _norm_stream_id(stream_id)
        if sid == "main":
            raise HTTPException(400, "stream main нельзя удалить")
        with streams_lock:
            slot = streams.pop(sid, None)
        if slot is None:
            raise HTTPException(404, f"stream {sid} not found")
        slot["pipeline"].shutdown()
        slot["logger"].close()
        return {"ok": True, "stream_id": sid}

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

    @app.post("/api/upload_video")
    async def _upload_video(file: UploadFile = File(...)) -> dict:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        src_name = Path(file.filename or "video.bin").name
        ext = Path(src_name).suffix.lower()
        allowed = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}
        if ext not in allowed:
            raise HTTPException(400, f"unsupported extension: {ext or 'none'}")
        stem = Path(src_name).stem or "video"
        target = DATA_DIR / src_name
        i = 1
        while target.exists():
            target = DATA_DIR / f"{stem}_{i}{ext}"
            i += 1
        size = 0
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                size += len(chunk)
        await file.close()
        return {"ok": True, "path": str(target), "name": target.name, "size_bytes": size}

    @app.post("/api/open")
    async def _open(req: OpenRequest) -> dict:
        sid = _norm_stream_id(req.stream_id)
        slot = _get_slot(sid, create=True)
        pipeline = slot["pipeline"]
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
    async def _play(stream_id: str = "main") -> dict:
        pipeline = _get_slot(stream_id)["pipeline"]
        pipeline.play()
        return {"playing": True}

    @app.post("/api/pause")
    async def _pause(stream_id: str = "main") -> dict:
        pipeline = _get_slot(stream_id)["pipeline"]
        pipeline.pause()
        return {"playing": False}

    @app.post("/api/seek")
    async def _seek(req: SeekRequest, stream_id: str = "main") -> dict:
        pipeline = _get_slot(stream_id)["pipeline"]
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
        with streams_lock:
            slots = list(streams.values())
        # Если менялся блок model — перегружаем веса во всех потоках.
        # Иначе обновляем только легкие настройки анализатора.
        for slot in slots:
            pipeline = slot["pipeline"]
            if "model" in body:
                pipeline.reload_model()
            else:
                pipeline.update_settings()
        return snapshot

    @app.get("/api/events")
    async def _events(limit: int = 200, stream_id: str = "main") -> dict:
        logger = _get_slot(stream_id)["logger"]
        return {"events": logger.recent(limit=max(1, min(limit, 1000)))}

    @app.delete("/api/events")
    async def _clear_events(stream_id: str = "main") -> dict:
        logger = _get_slot(stream_id)["logger"]
        logger.clear()
        return {"ok": True}

    # ------------------------------------------------------------- streams
    # MJPEG для живого видео и WS для событий/метрик.

    @app.get("/video_feed")
    def _video_feed(stream_id: str = "main"):
        pipeline = _get_slot(stream_id)["pipeline"]
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
            main = _get_slot("main")["pipeline"]
            await ws.send_text(json.dumps({"type": "hello", "stream_id": "main", "info": main.info()}))
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


async def _status_pump(hub: WSHub, streams: dict[str, dict[str, Any]], lock: threading.RLock) -> None:
    """Периодически рассылает статус пайплайна и снимок треков."""
    while True:
        try:
            with lock:
                items = list(streams.items())
            for sid, slot in items:
                await hub.broadcast({"type": "status", "stream_id": sid, "info": slot["pipeline"].info()})
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
    main_rss = int(base.get("rss_bytes") or 0)
    extra = list(cfg_store.cfg.pipeline.analytics_extra_pids or [])
    extras_rss = 0
    try:
        import psutil  # type: ignore

        for pid in extra:
            try:
                extras_rss += int(psutil.Process(int(pid)).memory_info().rss)
            except Exception:
                pass
    except Exception:
        pass

    pm = pipeline.metrics()
    worker_pid = pm.get("inference_worker_pid")
    infer_rss = 0
    if worker_pid:
        try:
            import psutil  # type: ignore

            infer_rss = int(psutil.Process(int(worker_pid)).memory_info().rss)
        except Exception:
            infer_rss = 0

    # Пайплайн (декод, FSM, веб, MJPEG, …) без отдельного inference-процесса; инференс — отдельная линия на графике.
    pipeline_rss = int(main_rss + extras_rss)
    total_footprint = int(pipeline_rss + infer_rss)
    rss_sum = total_footprint

    cuda_alloc = None
    cuda_reserved = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            cuda_alloc = int(torch.cuda.memory_allocated())
            cuda_reserved = int(torch.cuda.memory_reserved())
    except Exception:
        pass

    prev_bytes = int(pm.get("buffers", {}).get("forensic_ring_bytes") or 0)
    preview_bytes_est = prev_bytes
    now = time.perf_counter()
    alpha = 0.12
    ema_f = float(rss_sum)
    peak_5s = float(rss_sum)
    hist_slice: list[tuple[float, float, float, float]] = []
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
        samples: list[tuple[float, float, float, float]] = list(_ANALYTICS_MEM_STATE.get("rss_samples_v2") or [])
        samples.append((now, float(rss_sum), float(pipeline_rss), float(infer_rss)))
        if len(samples) > 180:
            samples = samples[-180:]
        _ANALYTICS_MEM_STATE["rss_samples_v2"] = samples
        hist_slice = samples[-90:]
    t0 = hist_slice[0][0] if hist_slice else now
    rss_history = [
        {
            "t": round(t - t0, 3),
            "rss_total_bytes": int(tr),
            "rss_pipeline_bytes": int(pr),
            "rss_inference_bytes": int(ir) if ir > 0 else None,
            # обратная совместимость для старых клиентов графика
            "rss_ema_bytes": int(tr),
        }
        for t, tr, pr, ir in hist_slice
    ]

    memory_breakdown: list[dict[str, Any]] = []
    wpath = Path(cfg_store.cfg.model.weights)
    if not wpath.is_absolute():
        wpath = ROOT / wpath
    if wpath.exists():
        memory_breakdown.append(
            {
                "kind": "disk",
                "label": "Файл весов (только диск)",
                "pid": None,
                "bytes": int(wpath.stat().st_size),
                "hint": str(wpath.name),
            }
        )

    if infer_rss > 0 and worker_pid:
        memory_breakdown.append(
            {
                "kind": "process_rss",
                "label": "Инференс: YOLO + трекер",
                "pid": int(worker_pid),
                "bytes": int(infer_rss),
                "hint": "отдельный процесс (model.use_inference_worker)",
            }
        )
    else:
        memory_breakdown.append(
            {
                "kind": "note",
                "label": "Инференс: YOLO + трекер",
                "pid": None,
                "bytes": None,
                "hint": "сейчас внутри основного процесса (включено в «Пайплайн»). Для отдельной линии на графике включите model.use_inference_worker",
            }
        )

    memory_breakdown.append(
        {
            "kind": "process_rss",
            "label": "Пайплайн: декод, очереди, FSM, веб-API, MJPEG",
            "pid": base.get("pid"),
            "bytes": int(main_rss),
            "hint": "без отдельного inference-worker; браузер сюда не входит",
        }
    )
    try:
        import psutil  # type: ignore

        for pid in extra:
            try:
                pr = psutil.Process(int(pid))
                br = int(pr.memory_info().rss)
                memory_breakdown.append(
                    {
                        "kind": "process_rss",
                        "label": f"Доп. процесс (PID {int(pid)})",
                        "pid": int(pid),
                        "bytes": br,
                        "hint": pr.name(),
                    }
                )
            except Exception:
                memory_breakdown.append(
                    {
                        "kind": "process_rss",
                        "label": f"Доп. PID {int(pid)}",
                        "pid": int(pid),
                        "bytes": None,
                        "hint": "нет доступа / процесс завершён",
                    }
                )
    except Exception:
        pass
    if cuda_reserved is not None:
        memory_breakdown.append(
            {
                "kind": "vram",
                "label": "CUDA VRAM (зарезервировано)",
                "pid": None,
                "bytes": int(cuda_reserved),
                "hint": "учёт на GPU, не RSS ОЗУ",
            }
        )
    if cuda_alloc is not None:
        memory_breakdown.append(
            {
                "kind": "vram",
                "label": "CUDA VRAM (аллоцировано)",
                "pid": None,
                "bytes": int(cuda_alloc),
                "hint": "учёт на GPU, не RSS ОЗУ",
            }
        )
    buf = pm.get("buffers") or {}
    fpb = int(buf.get("frame_pool_bytes") or 0)
    if fpb > 0:
        slots = buf.get("frame_pool_slots")
        memory_breakdown.append(
            {
                "kind": "buffer",
                "label": "Пул кадров BGR (оценка внутри процесса)",
                "pid": None,
                "bytes": fpb,
                "hint": f"до {slots} полных кадров" if slots is not None else None,
            }
        )
    frb = int(buf.get("forensic_ring_bytes") or 0)
    if frb > 0:
        memory_breakdown.append(
            {
                "kind": "buffer",
                "label": "Forensic ring (JPEG)",
                "pid": None,
                "bytes": frb,
                "hint": "диагностические снимки в RAM",
            }
        )

    return {
        **base,
        "rss_analytics_sum_bytes": rss_sum,
        "rss_pipeline_bytes": pipeline_rss,
        "rss_inference_worker_bytes": int(infer_rss) if infer_rss else None,
        "rss_ema_bytes": round(ema_f, 0),
        "rss_peak_recent_bytes": int(peak_5s),
        "cuda_memory_allocated_bytes": cuda_alloc,
        "cuda_memory_reserved_bytes": cuda_reserved,
        "preview_memory_bytes_est": preview_bytes_est,
        "rss_history": rss_history,
        "memory_breakdown": memory_breakdown,
    }


def _system_metrics() -> dict:
    cpu_percent = None
    ram_used = None
    ram_total = None
    ram_percent = None
    processes_top_rss: list[dict[str, Any]] = []
    try:
        import psutil  # type: ignore

        cpu_percent = float(psutil.cpu_percent(interval=0.0))
        vm = psutil.virtual_memory()
        ram_used = int(vm.used)
        ram_total = int(vm.total)
        ram_percent = float(vm.percent)
        scored: list[tuple[int, int, str]] = []
        for p in psutil.process_iter(attrs=["pid", "name", "memory_info"]):
            try:
                mi = p.info.get("memory_info")
                if mi is None:
                    continue
                r = int(mi.rss)
                if r < 80_000_000:
                    continue
                scored.append((r, int(p.info["pid"]), str(p.info["name"] or "?")))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        scored.sort(key=lambda x: -x[0])
        for r, pid, name in scored[:10]:
            processes_top_rss.append({"rss_bytes": r, "pid": pid, "name": name[:48]})
    except Exception:
        pass
    return {
        "cpu_percent": cpu_percent,
        "ram_used_bytes": ram_used,
        "ram_total_bytes": ram_total,
        "ram_percent": ram_percent,
        "gpu": _gpu_metrics(),
        "processes_top_rss": processes_top_rss,
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
