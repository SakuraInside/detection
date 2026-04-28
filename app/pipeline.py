"""Video decoding + inference pipeline.

Threads
-------
* `_decoder`  reads frames from cv2.VideoCapture (MKV/MP4 via FFmpeg).
* `_worker`   runs YOLO + tracker + FSM analyzer + overlay rendering.
* The MJPEG endpoint reads the latest rendered frame via `latest_jpeg()`.

Back-pressure policy
--------------------
The decode queue keeps at most N frames.  When the worker can't keep up
the decoder drops the *oldest* frame and pushes the newest, so the live
display stays low-latency.  Logging and WebSocket events are decoupled
from this queue and never share back-pressure with the video stream,
which means we never lose a detection event because the renderer is
slow.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .analyzer import Analyzer, AnalyzerEvent
from .config import AppConfig, ConfigStore
from .detector import DetectionResult, YoloTracker
from .logger_db import EventLogger, EventRow, now


@dataclass
class FrameJob:
    frame_id: int
    pos_ms: float
    image: np.ndarray
    pts: float


@dataclass
class RenderedFrame:
    frame_id: int
    pos_ms: float
    jpeg: bytes
    detections: int
    persons: int
    inference_ms: float


_OVERLAY_COLORS = {
    "candidate": (0, 200, 255),
    "static": (0, 220, 220),
    "unattended": (0, 140, 255),
    "alarm_abandoned": (0, 0, 255),
    "alarm_disappeared": (255, 0, 200),
    "person": (255, 200, 0),
    "default": (180, 180, 180),
}


class VideoPipeline:
    """Singleton-ish: one active video source per app instance."""

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

        self._cap_lock = threading.RLock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: Optional[str] = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._frame_w: int = 0
        self._frame_h: int = 0

        self._play_event = threading.Event()
        self._stop_event = threading.Event()
        self._seek_to: Optional[int] = None
        self._seek_lock = threading.Lock()
        self._frame_id = 0

        self._decode_q: "queue.Queue[FrameJob]" = queue.Queue(
            maxsize=max(2, config_store.cfg.pipeline.decode_queue)
        )
        self._latest_lock = threading.Lock()
        self._latest_rendered: Optional[RenderedFrame] = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "decoded": 0,
            "dropped_decode": 0,
            "inferred": 0,
            "events": 0,
            "decode_fps": 0.0,
            "render_fps": 0.0,
            "inference_ms_avg": 0.0,
        }
        self._decode_fps_window: list[float] = []
        self._render_fps_window: list[float] = []
        self._inference_ms_window: list[float] = []

        self._decoder_thread = threading.Thread(target=self._decoder_loop, name="decoder", daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop, name="inference", daemon=True)
        self._decoder_thread.start()
        self._worker_thread.start()

    # ---------------------------------------------------------------- public

    @property
    def names(self) -> dict[int, str]:
        return self._tracker.names

    def info(self) -> dict:
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

    def open(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {p}")
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
        self._play_event.set()
        return self.info()

    def play(self) -> None:
        self._play_event.set()

    def pause(self) -> None:
        self._play_event.clear()

    def seek(self, frame_index: int) -> None:
        with self._seek_lock:
            self._seek_to = max(0, int(frame_index))
        self._tracker.reset()
        self._analyzer.reset()

    def reload_model(self) -> None:
        self._tracker.reload(self._cfg_store.cfg.model)
        self._analyzer.update_config(self._cfg_store.cfg.analyzer)

    def update_settings(self) -> None:
        self._analyzer.update_config(self._cfg_store.cfg.analyzer)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._latest_lock:
            return None if self._latest_rendered is None else self._latest_rendered.jpeg

    def shutdown(self) -> None:
        self._stop_event.set()
        self._play_event.set()
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
        self._decoder_thread.join(timeout=2.0)
        self._worker_thread.join(timeout=2.0)

    # --------------------------------------------------------------- threads

    def _decoder_loop(self) -> None:
        last_ts = time.perf_counter()
        period = 1.0 / 60.0
        while not self._stop_event.is_set():
            self._play_event.wait(timeout=0.2)
            if self._stop_event.is_set():
                break
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
                self._drain_decode_queue()

            with self._cap_lock:
                ok, frame = cap.read()
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok:
                # End of stream — stop playback but keep file mounted.
                self._play_event.clear()
                time.sleep(0.05)
                continue

            self._frame_id += 1
            job = FrameJob(frame_id=self._frame_id, pos_ms=pos_ms, image=frame, pts=time.perf_counter())
            try:
                self._decode_q.put_nowait(job)
            except queue.Full:
                # Drop oldest to keep latency low.
                try:
                    self._decode_q.get_nowait()
                    with self._stats_lock:
                        self._stats["dropped_decode"] += 1
                except queue.Empty:
                    pass
                try:
                    self._decode_q.put_nowait(job)
                except queue.Full:
                    pass

            with self._stats_lock:
                self._stats["decoded"] += 1
            self._tick_window(self._decode_fps_window, "decode_fps")

            # Honour source FPS.
            period = 1.0 / max(1.0, fps)
            now_ts = time.perf_counter()
            sleep_for = period - (now_ts - last_ts)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_ts = time.perf_counter()

    def _drain_decode_queue(self) -> None:
        try:
            while True:
                self._decode_q.get_nowait()
        except queue.Empty:
            pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._decode_q.get(timeout=0.2)
            except queue.Empty:
                continue
            ts_now = now()
            try:
                result = self._tracker.infer(job.image)
            except Exception as exc:
                print(f"[pipeline] inference error: {exc}")
                continue

            events = self._analyzer.ingest(ts_now, result)
            for ev in events:
                self._handle_event(ev, job, ts_now)

            with self._stats_lock:
                self._stats["inferred"] += 1
                self._inference_ms_window.append(result.inference_ms)
                if len(self._inference_ms_window) > 60:
                    self._inference_ms_window = self._inference_ms_window[-60:]
                if self._inference_ms_window:
                    self._stats["inference_ms_avg"] = sum(self._inference_ms_window) / len(
                        self._inference_ms_window
                    )

            jpeg = self._render(job, result)
            with self._latest_lock:
                self._latest_rendered = RenderedFrame(
                    frame_id=job.frame_id,
                    pos_ms=job.pos_ms,
                    jpeg=jpeg,
                    detections=len(result.detections),
                    persons=len(result.persons),
                    inference_ms=result.inference_ms,
                )
            self._tick_window(self._render_fps_window, "render_fps")

    # --------------------------------------------------------------- helpers

    def _tick_window(self, window: list[float], stat_key: str) -> None:
        t = time.perf_counter()
        window.append(t)
        cutoff = t - 1.0
        while window and window[0] < cutoff:
            window.pop(0)
        with self._stats_lock:
            self._stats[stat_key] = float(len(window))

    def _render(self, job: FrameJob, result: DetectionResult) -> bytes:
        cfg: AppConfig = self._cfg_store.cfg
        img = job.image
        if cfg.pipeline.render_overlay:
            img = img.copy()
            tracks = {t["id"]: t for t in self._analyzer.tracks_snapshot()}
            if cfg.ui.show_persons:
                for p in result.persons:
                    self._draw_box(img, p.bbox, _OVERLAY_COLORS["person"], f"person {p.confidence:.2f}")
            for det in result.detections:
                state = tracks.get(det.track_id, {}).get("state", "candidate")
                color = _OVERLAY_COLORS.get(state, _OVERLAY_COLORS["default"])
                label_parts = [det.cls_name, f"{det.confidence:.2f}"]
                if cfg.ui.show_track_ids:
                    label_parts.insert(0, f"#{det.track_id}")
                label_parts.append(state)
                self._draw_box(img, det.bbox, color, " ".join(label_parts))

            self._draw_hud(img, result)

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b""

    def _draw_box(self, img: np.ndarray, bbox, color, label: str) -> None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def _draw_hud(self, img: np.ndarray, result: DetectionResult) -> None:
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
        snapshot_path: Optional[str] = None
        if ev.type in ("abandoned", "disappeared", "appeared"):
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
            x1, y1, x2, y2 = (int(v) for v in ev.bbox)
            h, w = job.image.shape[:2]
            x1 = max(0, x1 - 20); y1 = max(0, y1 - 20)
            x2 = min(w, x2 + 20); y2 = min(h, y2 + 20)
            crop = job.image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else job.image
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{ev.type}_{ev.track_id}_{ts_str}_{int(job.pos_ms)}.jpg"
            path = self._snap_dir / fname
            cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return f"logs/snapshots/{fname}"
        except Exception as exc:
            print(f"[pipeline] snapshot save failed: {exc}")
            return None
