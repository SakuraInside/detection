"""Abandoned/disappeared object analyzer.

This is the FSM block from the architecture in the project doc.

Per-track state machine
-----------------------
NONE → CANDIDATE → STATIC → UNATTENDED → ALARM_ABANDONED
                                          ↓ (track lost ≥ disappear_grace_sec)
                                    ALARM_DISAPPEARED

Transitions:
    CANDIDATE  -> STATIC          when centroid drift over `static_window_sec`
                                  stays below `static_displacement_px`.
    STATIC     -> UNATTENDED      when no person within `owner_proximity_px`
                                  for `owner_left_sec` seconds.
    UNATTENDED -> ALARM_ABANDONED when staying unattended `abandon_time_sec`.
    ALARM_*    -> NO_OBJECT       when the operator acknowledges or the
                                  object actually moves again.

The `ingest()` method consumes one DetectionResult per video frame and
returns the events that have just been raised on this frame.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .config import AnalyzerConfig
from .detector import Detection, DetectionResult


class TrackState(str, Enum):
    NONE = "none"
    CANDIDATE = "candidate"
    STATIC = "static"
    UNATTENDED = "unattended"
    ALARM_ABANDONED = "alarm_abandoned"
    ALARM_DISAPPEARED = "alarm_disappeared"


@dataclass
class TrackHistory:
    track_id: int
    cls_id: int
    cls_name: str
    state: TrackState = TrackState.NONE
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_centroid: tuple[float, float] = (0.0, 0.0)
    last_conf: float = 0.0
    static_since_ts: float = 0.0
    unattended_since_ts: float = 0.0
    abandoned_at_ts: float = 0.0
    last_owner_near_ts: float = 0.0
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=180))
    presence_count: int = 0
    frames_seen: int = 0
    raised_abandoned: bool = False
    raised_disappeared: bool = False

    def area(self) -> float:
        x1, y1, x2, y2 = self.last_bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    def displacement_window(self, since_ts: float) -> float:
        if not self.centroid_history:
            return 0.0
        xs = [p for ts, p in self.centroid_history if ts >= since_ts]
        if len(xs) < 2:
            return 0.0
        cx0, cy0 = xs[0]
        cx1, cy1 = xs[-1]
        return math.hypot(cx1 - cx0, cy1 - cy0)


@dataclass
class AnalyzerEvent:
    type: str
    track_id: int
    cls_id: int
    cls_name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    note: Optional[str] = None


class Analyzer:
    """Stateful, single-threaded — owned by the inference worker."""

    def __init__(self, cfg: AnalyzerConfig) -> None:
        self._cfg = cfg
        self._tracks: dict[int, TrackHistory] = {}

    def update_config(self, cfg: AnalyzerConfig) -> None:
        self._cfg = cfg

    def reset(self) -> None:
        self._tracks.clear()

    def tracks_snapshot(self) -> list[dict]:
        out: list[dict] = []
        for t in self._tracks.values():
            out.append(
                {
                    "id": t.track_id,
                    "cls": t.cls_name,
                    "state": t.state.value,
                    "bbox": list(t.last_bbox),
                    "conf": round(t.last_conf, 3),
                    "static_for_sec": max(0.0, time.time() - t.static_since_ts) if t.static_since_ts else 0.0,
                    "unattended_for_sec": max(0.0, time.time() - t.unattended_since_ts) if t.unattended_since_ts else 0.0,
                    "alarm": t.state in (TrackState.ALARM_ABANDONED, TrackState.ALARM_DISAPPEARED),
                }
            )
        return out

    def ingest(self, ts: float, result: DetectionResult) -> list[AnalyzerEvent]:
        events: list[AnalyzerEvent] = []
        seen_ids: set[int] = set()
        cfg = self._cfg

        for det in result.detections:
            seen_ids.add(det.track_id)
            track = self._tracks.get(det.track_id)
            if track is None:
                track = TrackHistory(
                    track_id=det.track_id,
                    cls_id=det.cls_id,
                    cls_name=det.cls_name,
                    first_seen_ts=ts,
                )
                self._tracks[det.track_id] = track
                track.state = TrackState.CANDIDATE
                events.append(
                    AnalyzerEvent(
                        "appeared",
                        det.track_id,
                        det.cls_id,
                        det.cls_name,
                        det.bbox,
                        det.confidence,
                    )
                )

            track.last_seen_ts = ts
            track.last_bbox = det.bbox
            track.last_centroid = det.centroid
            track.last_conf = det.confidence
            track.presence_count += 1
            track.frames_seen += 1
            track.centroid_history.append((ts, det.centroid))

            if track.area() < cfg.min_object_area_px:
                continue

            person_near = self._is_person_near(det, result.persons)
            if person_near:
                track.last_owner_near_ts = ts

            displacement = track.displacement_window(ts - cfg.static_window_sec)
            is_static = displacement <= cfg.static_displacement_px and (
                ts - track.first_seen_ts
            ) >= cfg.static_window_sec

            if track.state == TrackState.CANDIDATE and is_static:
                track.state = TrackState.STATIC
                track.static_since_ts = ts
            elif track.state == TrackState.STATIC and not is_static:
                # Object started moving again — drop static lock.
                track.state = TrackState.CANDIDATE
                track.static_since_ts = 0.0
                track.unattended_since_ts = 0.0
                continue

            if track.state in (TrackState.STATIC, TrackState.UNATTENDED, TrackState.ALARM_ABANDONED):
                without_owner_for = ts - track.last_owner_near_ts if track.last_owner_near_ts else ts - track.first_seen_ts
                if without_owner_for >= cfg.owner_left_sec:
                    if track.state == TrackState.STATIC:
                        track.state = TrackState.UNATTENDED
                        track.unattended_since_ts = ts

            if track.state == TrackState.UNATTENDED:
                unattended_for = ts - track.unattended_since_ts
                if unattended_for >= cfg.abandon_time_sec and not track.raised_abandoned:
                    track.state = TrackState.ALARM_ABANDONED
                    track.abandoned_at_ts = ts
                    track.raised_abandoned = True
                    events.append(
                        AnalyzerEvent(
                            "abandoned",
                            track.track_id,
                            track.cls_id,
                            track.cls_name,
                            track.last_bbox,
                            track.last_conf,
                            note=f"static {ts - track.static_since_ts:.1f}s, owner-left {unattended_for:.1f}s",
                        )
                    )

        # Detect disappearance for alarm-grade objects that vanished from view.
        to_drop: list[int] = []
        for tid, track in self._tracks.items():
            if tid in seen_ids:
                continue
            silent_for = ts - track.last_seen_ts
            if track.raised_abandoned and not track.raised_disappeared:
                if silent_for >= cfg.disappear_grace_sec:
                    track.state = TrackState.ALARM_DISAPPEARED
                    track.raised_disappeared = True
                    events.append(
                        AnalyzerEvent(
                            "disappeared",
                            track.track_id,
                            track.cls_id,
                            track.cls_name,
                            track.last_bbox,
                            track.last_conf,
                            note=f"missing {silent_for:.1f}s after abandonment",
                        )
                    )
            # Garbage collect untracked, non-alarm hypotheses.
            if not track.raised_abandoned and silent_for > max(10.0, cfg.disappear_grace_sec * 2):
                to_drop.append(tid)
        for tid in to_drop:
            self._tracks.pop(tid, None)

        return events

    def _is_person_near(self, obj: Detection, persons: list[Detection]) -> bool:
        if not persons:
            return False
        prox = self._cfg.owner_proximity_px
        ox, oy = obj.centroid
        ox1, oy1, ox2, oy2 = obj.bbox
        for person in persons:
            px, py = person.centroid
            if math.hypot(px - ox, py - oy) <= prox:
                return True
            # Also count "overlap": person bbox intersects object bbox.
            px1, py1, px2, py2 = person.bbox
            if px1 < ox2 and ox1 < px2 and py1 < oy2 and oy1 < py2:
                return True
        return False
