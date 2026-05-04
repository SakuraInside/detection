"""Пайплайн декодирования, инференса и визуализации.

Потоки
------
* `_decoder`  читает кадры из cv2.VideoCapture (через FFmpeg/OpenCV).
* `_worker`   запускает YOLO + трекер + FSM-анализатор + рендер оверлея.
* MJPEG endpoint читает самый свежий отрендеренный кадр из памяти.

Политика back-pressure
----------------------
Очередь декодирования ограничена. Перед постановкой нового кадра декодер
опустошает очередь (latest-only): воркер обрабатывает самый свежий кадр.
Опционально кадр копируется в пул буферов фиксированного размера (bounded RAM).
Preview (MJPEG) может даунскейлиться и троттлиться отдельно от аналитики;
если JPEG переиспользуется, кадр не ставится в очередь рендера — буфер
сразу возвращается в пул. События и логи не зависят от очереди рендера.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .analyzer import Analyzer, AnalyzerEvent
from .config import AppConfig, ConfigStore
from .detector import DetectionResult, YoloTracker
from .logger_db import EventLogger, EventRow, now
from .runtime_bridge import RuntimeBridgeConfig, RuntimeControlDecision, RuntimeCoreBridge

cv2.setUseOptimized(True)
# Не распараллеливаем OpenCV по CPU чрезмерно: это снижает конкуренцию
# потоков декодера/рендера с инференсом и стабилизирует latency.
cv2.setNumThreads(1)

_OVERLAY_COLORS = {
    "candidate": (0, 200, 255),
    "static": (0, 220, 220),
    "unattended": (0, 140, 255),
    "alarm_abandoned": (0, 0, 255),
    "alarm_disappeared": (255, 0, 200),
    "person": (255, 0, 0),
    "default": (180, 180, 180),
}


class FramePool:
    """Ограниченный пул буферов BGR uint8; обратное давление через блокировку acquire."""

    def __init__(self, height: int, width: int, channels: int, size: int) -> None:
        self._buffers = [
            np.empty((height, width, channels), dtype=np.uint8) for _ in range(max(1, size))
        ]
        self._free: "queue.Queue[int]" = queue.Queue(maxsize=len(self._buffers))
        for i in range(len(self._buffers)):
            self._free.put_nowait(i)

    def acquire_copy_from(
        self, src: np.ndarray, stop_event: threading.Event
    ) -> Optional[tuple[np.ndarray, int]]:
        while not stop_event.is_set():
            try:
                idx = self._free.get(timeout=0.25)
            except queue.Empty:
                continue
            np.copyto(self._buffers[idx], src)
            return self._buffers[idx], idx
        return None

    def release(self, idx: int) -> None:
        self._free.put_nowait(idx)


@dataclass
class FrameJob:
    frame_id: int
    pos_ms: float
    image: np.ndarray
    pts: float
    pool_idx: Optional[int] = None


@dataclass
class RenderedFrame:
    frame_id: int
    pos_ms: float
    jpeg: bytes
    detections: int
    persons: int
    inference_ms: float


@dataclass
class RenderJob:
    frame_id: int
    pos_ms: float
    image: np.ndarray
    result: DetectionResult
    tracks: list[dict]
    pool_idx: Optional[int] = None


class VideoPipeline:
    """Один активный источник видео на экземпляр приложения."""

    def __init__(
        self,
        config_store: ConfigStore,
        logger: EventLogger,
        snapshots_dir: Path,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._cfg_store = config_store
        self._logger = logger
        self._snap_dir = snapshots_dir
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        self._on_event = on_event

        self._tracker = YoloTracker(config_store.cfg.model)
        self._analyzer = Analyzer(config_store.cfg.analyzer)
        self._runtime_bridge = RuntimeCoreBridge(
            RuntimeBridgeConfig(
                addr=str(config_store.cfg.pipeline.runtime_core_addr or "").strip(),
                timeout_sec=float(config_store.cfg.pipeline.runtime_core_timeout_sec),
                control_addr=str(config_store.cfg.pipeline.runtime_control_addr or "").strip(),
                control_timeout_sec=float(config_store.cfg.pipeline.runtime_control_timeout_sec),
            )
        )

        # Состояние открытого видео и его метаданные.
        self._cap_lock = threading.RLock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: Optional[str] = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._frame_w: int = 0
        self._frame_h: int = 0

        self._play_event = threading.Event()
        self._stop_event = threading.Event()
        # Отложенный seek применяется декодером в "безопасной точке".
        self._seek_to: Optional[int] = None
        self._seek_lock = threading.Lock()
        self._frame_id = 0

        self._frame_pool: Optional[FramePool] = None
        self._snapshot_resize_buf: Optional[np.ndarray] = None
        # Декод через внешний `video-bridge` (Rust), иначе OpenCV в этом процессе.
        self._rust_bridge: Any = None

        self._decode_q: "queue.Queue[FrameJob]" = queue.Queue(
            maxsize=max(1, int(config_store.cfg.pipeline.decode_queue))
        )
        # Храним только последний отрендеренный кадр для MJPEG.
        self._latest_lock = threading.Lock()
        self._latest_rendered: Optional[RenderedFrame] = None
        self._last_result: Optional[DetectionResult] = None
        self._last_tracks: list[dict] = []
        self._stats_lock = threading.Lock()
        self._stats = {
            "decoded": 0,
            "dropped_decode": 0,
            "inferred": 0,
            "events": 0,
            "decode_fps": 0.0,
            "render_fps": 0.0,
            "inference_ms_avg": 0.0,
            "scheduler_mode": "local",
            "runtime_control_enabled": False,
            "runtime_control_decisions": 0,
            "runtime_control_fallbacks": 0,
            "scheduler_target_interval": 1,
            "scheduler_priority": "normal",
            "scheduler_overload_level": 0,
            "scheduler_latency_ema_ms": 0.0,
            "scheduler_max_roi_count": 1,
            "preview_skip_enqueue": 0,
        }
        self._decode_fps_window: list[float] = []
        self._render_fps_window: list[float] = []
        self._inference_ms_window: list[float] = []
        # Переиспользуемый буфер для preview (даунскейл перед оверлеем/JPEG).
        self._preview_buf: Optional[np.ndarray] = None
        self._last_preview_encode_mono: float = 0.0

        self._decoder_thread = threading.Thread(target=self._decoder_loop, name="decoder", daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop, name="inference", daemon=True)
        self._render_q: "queue.Queue[RenderJob]" = queue.Queue(
            maxsize=max(1, int(config_store.cfg.pipeline.result_queue))
        )
        self._render_thread = threading.Thread(target=self._render_loop, name="render", daemon=True)
        self._decoder_thread.start()
        self._worker_thread.start()
        self._render_thread.start()

    # ---------------------------------------------------------------- public

    @property
    def names(self) -> dict[int, str]:
        return self._tracker.names

    def info(self) -> dict:
        # Текущее состояние для UI: геометрия, позиции, метрики, треки.
        with self._cap_lock:
            cur_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) if self._cap is not None else 0
        with self._stats_lock:
            stats = dict(self._stats)
        return {
            "video_path": self._video_path,
            "fps": self._fps,
            "frame_count": self._total_frames,
            "current_frame": cur_frame,
            "duration_sec": self._total_frames / self._fps if self._fps else 0.0,
            "current_sec": cur_frame / self._fps if self._fps else 0.0,
            "width": self._frame_w,
            "height": self._frame_h,
            "playing": self._play_event.is_set(),
            "loaded": self._cap is not None,
            "stats": stats,
            "tracks": self._analyzer.tracks_snapshot(),
        }

    def metrics(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._latest_lock:
            latest = self._latest_rendered
        return {
            "stats": stats,
            "queues": {
                "decode_size": self._decode_q.qsize(),
                "decode_max": self._decode_q.maxsize,
                "render_size": self._render_q.qsize(),
                "render_max": self._render_q.maxsize,
            },
            "video": {
                "path": self._video_path,
                "fps": self._fps,
                "width": self._frame_w,
                "height": self._frame_h,
                "loaded": self._cap is not None,
                "playing": self._play_event.is_set(),
            },
            "latest_frame": None
            if latest is None
            else {
                "frame_id": latest.frame_id,
                "detections": latest.detections,
                "persons": latest.persons,
                "inference_ms": latest.inference_ms,
            },
        }

    def open(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        rust_addr = str(self._cfg_store.cfg.pipeline.rust_video_bridge_addr or "").strip()
        if rust_addr:
            from .rust_video_bridge import RustVideoBridge

            if self._rust_bridge is not None:
                try:
                    self._rust_bridge.close()
                except Exception:
                    pass
            br = RustVideoBridge(rust_addr)
            try:
                h = br.open_video(str(p))
            except Exception:
                br.close()
                raise
            self._rust_bridge = br
            with self._cap_lock:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                self._video_path = str(p)
                self._fps = float(h.get("fps") or 30.0)
                self._total_frames = int(h.get("frames") or 0)
                self._frame_w = int(h.get("width") or 0)
                self._frame_h = int(h.get("height") or 0)
        else:
            if self._rust_bridge is not None:
                try:
                    self._rust_bridge.close()
                except Exception:
                    pass
                self._rust_bridge = None
            backend = cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else cv2.CAP_ANY
            cap = cv2.VideoCapture(str(p), backend)
            if not cap.isOpened():
                raise RuntimeError(f"failed to open video: {p}")
            if self._cfg_store.cfg.pipeline.prefer_hw_decode:
                try:
                    if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
                        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except Exception:
                    pass
            with self._cap_lock:
                if self._cap is not None:
                    self._cap.release()
                self._cap = cap
                self._video_path = str(p)
                self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._tracker.reset()
        self._analyzer.reset()
        self._last_result = None
        self._preview_buf = None
        self._snapshot_resize_buf = None
        self._setup_frame_pool()
        self._play_event.set()
        return self.info()

    def play(self) -> None:
        self._play_event.set()

    def pause(self) -> None:
        self._play_event.clear()

    def seek(self, frame_index: int) -> None:
        # Не двигаем курсор напрямую из API-потока: делегируем декодеру.
        with self._seek_lock:
            self._seek_to = max(0, int(frame_index))
        # После seek старые треки/состояния больше невалидны.
        self._tracker.reset()
        self._analyzer.reset()
        self._last_result = None
        self._preview_buf = None
        self._snapshot_resize_buf = None

    def reload_model(self) -> None:
        # Применяем изменения model + analyzer (если были).
        self._tracker.reload(self._cfg_store.cfg.model)
        self._analyzer.update_config(self._cfg_store.cfg.analyzer)
        self._runtime_bridge.update(
            RuntimeBridgeConfig(
                addr=str(self._cfg_store.cfg.pipeline.runtime_core_addr or "").strip(),
                timeout_sec=float(self._cfg_store.cfg.pipeline.runtime_core_timeout_sec),
                control_addr=str(self._cfg_store.cfg.pipeline.runtime_control_addr or "").strip(),
                control_timeout_sec=float(self._cfg_store.cfg.pipeline.runtime_control_timeout_sec),
            )
        )
        self._last_result = None

    def update_settings(self) -> None:
        # Легкое обновление без перезагрузки нейросети.
        self._analyzer.update_config(self._cfg_store.cfg.analyzer)
        self._runtime_bridge.update(
            RuntimeBridgeConfig(
                addr=str(self._cfg_store.cfg.pipeline.runtime_core_addr or "").strip(),
                timeout_sec=float(self._cfg_store.cfg.pipeline.runtime_core_timeout_sec),
                control_addr=str(self._cfg_store.cfg.pipeline.runtime_control_addr or "").strip(),
                control_timeout_sec=float(self._cfg_store.cfg.pipeline.runtime_control_timeout_sec),
            )
        )

    def latest_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return None if self._latest_rendered is None else self._latest_rendered.jpeg

    def latest_rendered(self) -> Optional[RenderedFrame]:
        with self._latest_lock:
            return self._latest_rendered

    def shutdown(self) -> None:
        # Даем потокам сигнал завершения и освобождаем ресурсы видео.
        self._stop_event.set()
        self._play_event.set()
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
        if self._rust_bridge is not None:
            try:
                self._rust_bridge.close()
            except Exception:
                pass
            self._rust_bridge = None
        self._decoder_thread.join(timeout=2.0)
        self._worker_thread.join(timeout=2.0)
        self._render_thread.join(timeout=2.0)
        self._drain_decode_queue_release()
        try:
            while True:
                j = self._render_q.get_nowait()
                self._release_pool_idx(j.pool_idx)
        except queue.Empty:
            pass
        self._runtime_bridge.close()
        self._tracker.shutdown()

    def _setup_frame_pool(self) -> None:
        self._frame_pool = None
        n = int(self._cfg_store.cfg.pipeline.frame_pool_size or 0)
        if n > 0 and self._frame_h > 0 and self._frame_w > 0:
            self._frame_pool = FramePool(self._frame_h, self._frame_w, 3, n)

    def _release_pool_idx(self, pool_idx: Optional[int]) -> None:
        if pool_idx is None or self._frame_pool is None:
            return
        self._frame_pool.release(pool_idx)

    def _drain_decode_queue_release(self) -> None:
        try:
            while True:
                j = self._decode_q.get_nowait()
                self._release_pool_idx(j.pool_idx)
        except queue.Empty:
            pass

    def _should_skip_render_enqueue(self, job: FrameJob) -> bool:
        if not bool(self._cfg_store.cfg.pipeline.preview_skip_redundant_enqueue):
            return False
        with self._latest_lock:
            lr = self._latest_rendered
            last_mono = self._last_preview_encode_mono
        now_m = time.perf_counter()
        preview_iv = float(self._cfg_store.cfg.pipeline.preview_min_interval_ms or 0)
        if preview_iv > 0 and lr is not None and (now_m - last_mono) * 1000.0 < preview_iv:
            return True
        render_every = max(1, int(self._cfg_store.cfg.pipeline.render_every_n_frames))
        if lr is not None and (job.frame_id % render_every) != 0:
            return True
        return False

    def _bump_latest_render_meta(self, job: FrameJob, result: DetectionResult) -> None:
        with self._latest_lock:
            lr = self._latest_rendered
            if lr is None:
                return
            self._latest_rendered = RenderedFrame(
                frame_id=job.frame_id,
                pos_ms=job.pos_ms,
                jpeg=lr.jpeg,
                detections=len(result.detections),
                persons=len(result.persons),
                inference_ms=float(result.inference_ms),
            )

    def _dispatch_render(self, job: FrameJob, result: DetectionResult, tracks: list[dict]) -> None:
        if self._should_skip_render_enqueue(job):
            self._bump_latest_render_meta(job, result)
            with self._stats_lock:
                self._stats["preview_skip_enqueue"] += 1
            self._release_pool_idx(job.pool_idx)
            return
        self._enqueue_render(job, result, tracks)

    # --------------------------------------------------------------- threads

    def _decoder_loop(self) -> None:
        # Поток декодирования: OpenCV локально или TCP-поток из Rust video-bridge.
        last_ts = time.perf_counter()
        while not self._stop_event.is_set():
            self._play_event.wait(timeout=0.2)
            if self._stop_event.is_set():
                break

            rb = self._rust_bridge
            if rb is not None:
                with self._seek_lock:
                    target = self._seek_to
                    self._seek_to = None
                if target is not None and self._video_path:
                    self._drain_decode_queue_release()
                    try:
                        rb.open_video(self._video_path, seek_frame=target)
                    except Exception as exc:
                        print(f"[pipeline] rust bridge seek failed: {exc}")
                        time.sleep(0.05)
                        continue
                try:
                    pkt = rb.read_frame()
                except Exception as exc:
                    print(f"[pipeline] rust bridge read failed: {exc}")
                    self._play_event.clear()
                    time.sleep(0.05)
                    continue
                if pkt is None:
                    self._play_event.clear()
                    time.sleep(0.05)
                    continue
                frame, pos_ms, remote_fid = pkt
                self._frame_id = remote_fid
            else:
                with self._cap_lock:
                    cap = self._cap
                    fps = self._fps
                if cap is None:
                    time.sleep(0.05)
                    continue

                with self._seek_lock:
                    target = self._seek_to
                    self._seek_to = None
                if target is not None:
                    with self._cap_lock:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    self._drain_decode_queue_release()

                with self._cap_lock:
                    ok, frame = cap.read()
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not ok:
                    self._play_event.clear()
                    time.sleep(0.05)
                    continue

                self._frame_id += 1

            pool_idx: Optional[int] = None
            if self._frame_pool is not None:
                got = self._frame_pool.acquire_copy_from(frame, self._stop_event)
                if got is None:
                    break
                image, pool_idx = got
            else:
                image = frame
            job = FrameJob(
                frame_id=self._frame_id,
                pos_ms=pos_ms,
                image=image,
                pts=time.perf_counter(),
                pool_idx=pool_idx,
            )
            # Latest-only: не держим устаревшие кадры — только самый свежий в очереди.
            drained = 0
            try:
                while True:
                    dropped = self._decode_q.get_nowait()
                    drained += 1
                    self._release_pool_idx(dropped.pool_idx)
            except queue.Empty:
                pass
            if drained:
                with self._stats_lock:
                    self._stats["dropped_decode"] += drained
            try:
                self._decode_q.put_nowait(job)
            except queue.Full:
                try:
                    evicted = self._decode_q.get_nowait()
                    self._release_pool_idx(evicted.pool_idx)
                    with self._stats_lock:
                        self._stats["dropped_decode"] += 1
                    self._decode_q.put_nowait(job)
                except (queue.Empty, queue.Full):
                    self._release_pool_idx(job.pool_idx)

            with self._stats_lock:
                self._stats["decoded"] += 1
            self._tick_window(self._decode_fps_window, "decode_fps")

            # Ограничиваем частоту декодирования настройкой target_fps,
            # чтобы не грузить CPU лишними кадрами.
            target_fps = float(self._cfg_store.cfg.pipeline.target_fps or 0)
            effective_fps = float(self._fps)
            if target_fps > 0:
                effective_fps = min(effective_fps, target_fps)
            period = 1.0 / max(1.0, effective_fps)
            now_ts = time.perf_counter()
            sleep_for = period - (now_ts - last_ts)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_ts = time.perf_counter()

    def _worker_loop(self) -> None:
        # Поток инференса: берет кадр из очереди и обрабатывает его.
        while not self._stop_event.is_set():
            try:
                job = self._decode_q.get(timeout=0.2)
            except queue.Empty:
                continue
            ts_now = now()
            # Можно запускать тяжелый инференс не на каждом кадре.
            detect_every = max(1, int(self._cfg_store.cfg.pipeline.detect_every_n_frames))
            should_infer = self._last_result is None or (job.frame_id % detect_every == 0)
            scheduler_max_roi_count = 3
            scheduler_priority = "normal"
            # Если подключен runtime-core control-plane, решение по infer
            # централизуется в Rust scheduler.
            if self._last_result is not None:
                runtime_control_enabled = bool(
                    str(self._cfg_store.cfg.pipeline.runtime_control_addr or "").strip()
                )
                with self._stats_lock:
                    self._stats["runtime_control_enabled"] = runtime_control_enabled
                    if not runtime_control_enabled:
                        self._stats["scheduler_mode"] = "local"
                decision: Optional[RuntimeControlDecision] = self._runtime_bridge.should_infer(
                    camera_id=self._video_path or "single-camera",
                    frame_id=job.frame_id,
                    default_interval=detect_every,
                )
                if decision is not None:
                    should_infer = decision.infer
                    scheduler_max_roi_count = int(decision.max_roi_count)
                    scheduler_priority = str(decision.priority)
                    with self._stats_lock:
                        self._stats["scheduler_mode"] = "runtime-core"
                        self._stats["runtime_control_decisions"] += 1
                        self._stats["scheduler_target_interval"] = int(decision.target_interval)
                        self._stats["scheduler_priority"] = str(decision.priority)
                        self._stats["scheduler_overload_level"] = int(decision.overload_level)
                        self._stats["scheduler_latency_ema_ms"] = float(decision.latency_ema_ms)
                        self._stats["scheduler_max_roi_count"] = int(decision.max_roi_count)
                elif runtime_control_enabled:
                    with self._stats_lock:
                        self._stats["scheduler_mode"] = "local-fallback"
                        self._stats["runtime_control_fallbacks"] += 1

            if should_infer:
                try:
                    result = self._tracker.infer(
                        job.image,
                        max_roi_count=scheduler_max_roi_count,
                        priority=scheduler_priority,
                    )
                except Exception as exc:
                    print(f"[pipeline] inference error: {exc}")
                    self._release_pool_idx(job.pool_idx)
                    continue
                self._last_result = result

                events = self._analyzer.ingest(ts_now, result)
                for ev in events:
                    self._handle_event(ev, job, ts_now)
                self._last_tracks = self._analyzer.tracks_snapshot()
                self._runtime_bridge.send(
                    {
                        "type": "inference_result",
                        "camera_id": self._video_path or "single-camera",
                        "frame_id": job.frame_id,
                        "pos_ms": float(job.pos_ms),
                        "inference_ms": float(result.inference_ms),
                        "detections": len(result.detections),
                        "persons": len(result.persons),
                        "tracks": len(self._last_tracks),
                        "events": len(events),
                        "scheduler_priority": scheduler_priority,
                        "scheduler_max_roi_count": scheduler_max_roi_count,
                    }
                )

                with self._stats_lock:
                    self._stats["inferred"] += 1
                    self._inference_ms_window.append(result.inference_ms)
                    if len(self._inference_ms_window) > 60:
                        self._inference_ms_window = self._inference_ms_window[-60:]
                    if self._inference_ms_window:
                        self._stats["inference_ms_avg"] = sum(self._inference_ms_window) / len(
                            self._inference_ms_window
                        )
            else:
                # Повторно используем прошлый результат между кадрами,
                # чтобы разгрузить модель и повысить плавность.
                result = self._last_result
            tracks = list(self._last_tracks)
            self._runtime_bridge.send(
                {
                    "type": "frame_observed",
                    "camera_id": self._video_path or "single-camera",
                    "frame_id": job.frame_id,
                    "pos_ms": float(job.pos_ms),
                    "width": int(job.image.shape[1]) if job.image is not None else 0,
                    "height": int(job.image.shape[0]) if job.image is not None else 0,
                    "inferred": bool(should_infer),
                }
            )
            if result is None:
                self._release_pool_idx(job.pool_idx)
                continue
            self._dispatch_render(job, result, tracks)

    def _enqueue_render(self, job: FrameJob, result: DetectionResult, tracks: list[dict]) -> None:
        render_job = RenderJob(
            frame_id=job.frame_id,
            pos_ms=job.pos_ms,
            image=job.image,
            result=result,
            tracks=tracks,
            pool_idx=job.pool_idx,
        )
        try:
            self._render_q.put_nowait(render_job)
        except queue.Full:
            # Preview-контур может отставать, но analytics path блокировать нельзя.
            try:
                old = self._render_q.get_nowait()
                self._release_pool_idx(old.pool_idx)
            except queue.Empty:
                pass
            try:
                self._render_q.put_nowait(render_job)
            except queue.Full:
                self._release_pool_idx(render_job.pool_idx)

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._render_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                preview_iv = float(self._cfg_store.cfg.pipeline.preview_min_interval_ms or 0)
                render_every = max(1, int(self._cfg_store.cfg.pipeline.render_every_n_frames))
                with self._latest_lock:
                    lr = self._latest_rendered
                    last_mono = self._last_preview_encode_mono
                now_m = time.perf_counter()
                new_encode_ts: Optional[float] = None
                if preview_iv > 0 and lr is not None and (now_m - last_mono) * 1000.0 < preview_iv:
                    jpeg = lr.jpeg
                elif (job.frame_id % render_every) != 0 and lr is not None:
                    jpeg = lr.jpeg
                else:
                    jpeg = self._render(job, job.result, job.tracks)
                    new_encode_ts = time.perf_counter()
                with self._latest_lock:
                    if new_encode_ts is not None:
                        self._last_preview_encode_mono = new_encode_ts
                    self._latest_rendered = RenderedFrame(
                        frame_id=job.frame_id,
                        pos_ms=job.pos_ms,
                        jpeg=jpeg,
                        detections=len(job.result.detections),
                        persons=len(job.result.persons),
                        inference_ms=job.result.inference_ms,
                    )
                self._tick_window(self._render_fps_window, "render_fps")
            finally:
                self._release_pool_idx(job.pool_idx)

    # --------------------------------------------------------------- вспомогательные методы

    def _tick_window(self, window: list[float], stat_key: str) -> None:
        # Скользящее окно 1 секунда: количество отметок ~= мгновенный FPS.
        t = time.perf_counter()
        window.append(t)
        cutoff = t - 1.0
        while window and window[0] < cutoff:
            window.pop(0)
        with self._stats_lock:
            self._stats[stat_key] = float(len(window))

    def _ensure_preview_buf(self, height: int, width: int) -> np.ndarray:
        need = (height, width, 3)
        if self._preview_buf is None or self._preview_buf.shape != need:
            self._preview_buf = np.empty(need, dtype=np.uint8)
        return self._preview_buf

    def _render(self, job: RenderJob, result: DetectionResult, tracks_data: list[dict]) -> bytes:
        # Рисуем оверлей и кодируем в JPEG для web-стрима (опционально preview даунскейл).
        cfg: AppConfig = self._cfg_store.cfg
        src = job.image
        fh, fw = int(src.shape[0]), int(src.shape[1])
        long_edge = max(fh, fw)
        max_edge = int(cfg.pipeline.preview_max_long_edge or 0)
        sx = sy = 1.0
        img: np.ndarray
        if max_edge > 0:
            if long_edge > max_edge:
                scale = max_edge / float(long_edge)
                nw, nh = int(round(fw * scale)), int(round(fh * scale))
                buf = self._ensure_preview_buf(nh, nw)
                cv2.resize(src, (nw, nh), dst=buf, interpolation=cv2.INTER_AREA)
                img = buf
                sx = nw / float(fw)
                sy = nh / float(fh)
            else:
                img = src
        else:
            img = src

        if cfg.pipeline.render_overlay:
            if img is src:
                img = src.copy()
            tracks = {t["id"]: t for t in tracks_data}

            def map_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
                if sx == 1.0 and sy == 1.0:
                    return bbox
                x1, y1, x2, y2 = bbox
                return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

            if cfg.ui.show_persons:
                for p in result.persons:
                    person_label = f"person #{p.track_id} {p.confidence:.2f}"
                    self._draw_box(img, map_bbox(p.bbox), _OVERLAY_COLORS["person"], person_label)
            for det in result.detections:
                state = tracks.get(det.track_id, {}).get("state", "candidate")
                color = _OVERLAY_COLORS.get(state, _OVERLAY_COLORS["default"])
                label = ""
                if cfg.ui.show_labels:
                    label_parts = [det.cls_name, f"{det.confidence:.2f}"]
                    if cfg.ui.show_track_ids:
                        label_parts.insert(0, f"#{det.track_id}")
                    label_parts.append(state)
                    label = " ".join(label_parts)
                self._draw_box(img, map_bbox(det.bbox), color, label)

            self._draw_hud(img, result)

        if max_edge > 0 and long_edge > max_edge:
            q = int(cfg.pipeline.preview_jpeg_quality)
        else:
            q = int(cfg.pipeline.jpeg_quality)
        quality = int(max(40, min(95, q)))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else b""

    def _draw_box(self, img: np.ndarray, bbox, color, label: str) -> None:
        # Рисуем рамку и подпись одним стилем для всех типов объектов.
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

    def _draw_hud(self, img: np.ndarray, result: DetectionResult) -> None:
        # Верхняя строка телеметрии для быстрой диагностики производительности.
        h, w = img.shape[:2]
        with self._stats_lock:
            stats = dict(self._stats)
        text = (
            f"decode {stats['decode_fps']:.0f} fps | render {stats['render_fps']:.0f} fps | "
            f"inf {stats['inference_ms_avg']:.1f} ms | "
            f"obj {len(result.detections)} | persons {len(result.persons)}"
        )
        cv2.rectangle(img, (0, 0), (w, 26), (20, 20, 20), -1)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _handle_event(self, ev: AnalyzerEvent, job: FrameJob, ts: float) -> None:
        # Единая точка маршрутизации события:
        # SQLite -> WS -> снапшот на диск.
        snapshot_path: Optional[str] = None
        if ev.type in ("abandoned", "disappeared"):
            snapshot_path = self._save_snapshot(job, ev)

        row = EventRow(
            ts=ts,
            video_pos_ms=job.pos_ms,
            type=ev.type,
            track_id=ev.track_id,
            cls_id=ev.cls_id,
            cls_name=ev.cls_name,
            confidence=ev.confidence,
            bbox=ev.bbox,
            snapshot_path=snapshot_path,
            note=ev.note,
        )
        self._logger.log(row)

        with self._stats_lock:
            self._stats["events"] += 1

        if self._on_event is not None:
            try:
                self._on_event(
                    {
                        "ts": ts,
                        "video_pos_ms": job.pos_ms,
                        "type": ev.type,
                        "track_id": ev.track_id,
                        "cls_name": ev.cls_name,
                        "confidence": ev.confidence,
                        "bbox": list(ev.bbox),
                        "snapshot_path": snapshot_path,
                        "note": ev.note,
                    }
                )
            except Exception as exc:
                print(f"[pipeline] on_event broadcast failed: {exc}")

    def _save_snapshot(self, job: FrameJob, ev: AnalyzerEvent) -> Optional[str]:
        try:
            pc = self._cfg_store.cfg.pipeline
            x1, y1, x2, y2 = (int(v) for v in ev.bbox)
            h, w = job.image.shape[:2]
            # Делаем небольшой отступ вокруг bbox, чтобы контекст был читаемее.
            x1 = max(0, x1 - 20); y1 = max(0, y1 - 20)
            x2 = min(w, x2 + 20); y2 = min(h, y2 + 20)
            crop = job.image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else job.image
            out = crop
            max_edge = int(pc.snapshot_max_long_edge or 0)
            if max_edge > 0:
                ch, cw = crop.shape[:2]
                le = max(ch, cw)
                if le > max_edge:
                    scale = max_edge / float(le)
                    nw, nh = int(round(cw * scale)), int(round(ch * scale))
                    buf = self._ensure_snapshot_buf(nh, nw)
                    cv2.resize(crop, (nw, nh), dst=buf, interpolation=cv2.INTER_AREA)
                    out = buf
            q = int(max(40, min(95, int(pc.snapshot_jpeg_quality or 85))))
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{ev.type}_{ev.track_id}_{ts_str}_{int(job.pos_ms)}.jpg"
            path = self._snap_dir / fname
            cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            return f"logs/snapshots/{fname}"
        except Exception as exc:
            print(f"[pipeline] snapshot save failed: {exc}")
            return None

    def _ensure_snapshot_buf(self, height: int, width: int) -> np.ndarray:
        need = (height, width, 3)
        if self._snapshot_resize_buf is None or self._snapshot_resize_buf.shape != need:
            self._snapshot_resize_buf = np.empty(need, dtype=np.uint8)
        return self._snapshot_resize_buf
