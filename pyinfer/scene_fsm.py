"""Поведенческая FSM сцены (контур M4 по ТЗ) — порт native/src/scene_analyzer.cpp.

События определяются ПОВЕДЕНИЕМ во времени, не классом YOLO:
person_interaction / object_left / object_unattended / object_removed / object_missing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from .config import AnalyzerCfg
from .geom import bbox_area, centroid, iou_xyxy

CANDIDATE_DROP_SILENT_SEC = 2.0
INTERACTION_THROTTLE_SEC = 1.5


class St(IntEnum):
    NONE = 0
    CANDIDATE = 1
    STATIC = 2
    UNATTENDED = 3
    ALARM_UNATTENDED = 4
    ALARM_REMOVED = 5
    ALARM_MISSING = 6


def person_overlaps_bbox(obj_bbox, persons, fw, fh) -> bool:
    if not persons:
        return False
    frame_area = float(max(1, fw) * max(1, fh)) if (fw > 0 and fh > 0) else 0.0
    for p in persons:
        if p.confidence < 0.30:
            continue
        if frame_area > 1.0 and bbox_area(p.bbox) / frame_area > 0.14:
            continue
        if iou_xyxy(p.bbox, obj_bbox) >= 0.052:
            return True
    return False


def is_person_near(obj_bbox, persons, prox_px, fw, fh) -> bool:
    if not persons:
        return False
    ox, oy = centroid(obj_bbox)
    frame_area = float(max(1, fw) * max(1, fh)) if (fw > 0 and fh > 0) else 0.0

    OWNER_MIN_CONF = 0.44
    OWNER_MAX_AREA_RATIO = 0.09
    OWNER_MAX_ASPECT = 1.42
    OWNER_OBJ_MIN_IOU = 0.055

    # Умеренная зона владельца: не слишком большая (иначе прохожие «усыновляют»
    # чужие треки и плодят object_left), не крошечная (иначе стоящий человек не
    # привязывается к предмету у ног). Дистанция ниже меряется и до ног.
    eff_prox = prox_px
    if fw > 0 and fh > 0:
        diag = math.hypot(fw, fh)
        eff_prox = min(prox_px, max(60.0, 0.045 * diag))

    for p in persons:
        if p.confidence < OWNER_MIN_CONF:
            continue
        if frame_area > 1.0:
            if bbox_area(p.bbox) / frame_area > OWNER_MAX_AREA_RATIO:
                continue
            pw = p.bbox[2] - p.bbox[0]
            ph = p.bbox[3] - p.bbox[1]
            if pw > 1.0 and ph > 1.0 and (pw / ph) > OWNER_MAX_ASPECT:
                continue
        # расстояние до центра ИЛИ до ног (низ bbox): предмет на лавке/полу ближе
        # к ногам стоящего человека, чем к его центру-торсу.
        px, py = centroid(p.bbox)
        fx, fy = 0.5 * (p.bbox[0] + p.bbox[2]), p.bbox[3]
        dist = min(math.hypot(px - ox, py - oy), math.hypot(fx - ox, fy - oy))
        if dist > eff_prox:
            continue
        iou_po = iou_xyxy(p.bbox, obj_bbox)
        if iou_po >= OWNER_OBJ_MIN_IOU:
            return True
        if dist <= 0.28 * eff_prox:
            return True
        if (p.bbox[0] < obj_bbox[2] and obj_bbox[0] < p.bbox[2]
                and p.bbox[1] < obj_bbox[3] and obj_bbox[1] < p.bbox[3] and iou_po >= 0.028):
            return True
    return False


@dataclass
class TH:
    track_id: int
    cls_id: int
    cls_name: str
    state: St = St.CANDIDATE
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    last_bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    last_conf: float = 0.0
    static_since_ts: float = 0.0
    unattended_since_ts: float = 0.0
    last_owner_near_ts: float = 0.0
    last_interaction_emit_ts: float = 0.0
    ever_owner_near: bool = False
    owner_near_prev: bool = False
    was_confirmed: bool = False
    centroid_history: list = field(default_factory=list)  # (ts, cx, cy)
    object_left_emitted: bool = False
    raised_unattended: bool = False
    raised_removed: bool = False
    raised_missing: bool = False

    def area(self) -> float:
        return bbox_area(self.last_bbox)

    def displacement_window(self, since_ts: float) -> float:
        xs = [e for e in self.centroid_history if e[0] >= since_ts]
        if len(xs) < 2:
            return 0.0
        dx = xs[-1][1] - xs[0][1]
        dy = xs[-1][2] - xs[0][2]
        return math.hypot(dx, dy)


def _event(kind, cam, video_pos_ms, t: TH, note: str) -> dict:
    import time
    return {
        "type": kind,
        "camera_id": cam,
        "track_id": int(t.track_id),
        "cls_id": int(t.cls_id),
        "cls_name": t.cls_name,
        "confidence": float(t.last_conf),
        "ts_wall_ms": time.time() * 1000.0,
        "video_pos_ms": float(video_pos_ms),
        "bbox": [float(t.last_bbox[0]), float(t.last_bbox[1]),
                 float(t.last_bbox[2]), float(t.last_bbox[3])],
        "note": note,
    }


class SceneAnalyzer:
    def __init__(self, p: AnalyzerCfg):
        self.p = p
        self.tracks: dict[int, TH] = {}

    def reset(self):
        self.tracks.clear()

    def ingest(self, ts, video_pos_ms, camera_id, objects, persons, fw, fh):
        p = self.p
        tracks = self.tracks
        events = []
        seen_ids = []
        mlen = max(8, p.centroid_history_maxlen)

        for det in objects:
            tid = det.track_id
            if tid < 0:
                continue
            seen_ids.append(tid)
            tr = tracks.get(tid)
            if tr is None:
                tr = TH(track_id=tid, cls_id=det.class_id, cls_name=det.cls_name,
                        first_seen_ts=ts, state=St.CANDIDATE)
                tracks[tid] = tr

            tr.last_seen_ts = ts
            tr.last_bbox = det.bbox
            cx, cy = centroid(det.bbox)
            tr.last_conf = det.confidence
            tr.centroid_history.append((ts, cx, cy))
            if len(tr.centroid_history) > mlen:
                tr.centroid_history = tr.centroid_history[-mlen:]

            if tr.area() < p.min_object_area_px:
                continue

            owner_near = is_person_near(det.bbox, persons, p.owner_proximity_px, fw, fh)
            if owner_near:
                tr.last_owner_near_ts = ts
                tr.ever_owner_near = True

            disp = tr.displacement_window(ts - p.static_window_sec)
            is_static = (disp <= p.static_displacement_px
                         and (ts - tr.first_seen_ts) >= p.static_window_sec)

            if tr.state == St.CANDIDATE and is_static:
                tr.state = St.STATIC
                tr.static_since_ts = ts
                tr.was_confirmed = True
            elif tr.state == St.STATIC and not is_static:
                tr.state = St.CANDIDATE
                tr.static_since_ts = 0.0
                tr.unattended_since_ts = 0.0
                tr.object_left_emitted = False
                tr.owner_near_prev = owner_near
                continue

            if (owner_near and not tr.owner_near_prev and tr.was_confirmed
                    and (ts - tr.last_interaction_emit_ts) > INTERACTION_THROTTLE_SEC):
                events.append(_event("person_interaction", camera_id, video_pos_ms, tr,
                                     "owner near object"))
                tr.last_interaction_emit_ts = ts
            tr.owner_near_prev = owner_near

            if owner_near and tr.state == St.UNATTENDED and not tr.raised_unattended:
                tr.state = St.STATIC
                tr.unattended_since_ts = 0.0
                tr.object_left_emitted = False

            if tr.state in (St.STATIC, St.UNATTENDED, St.ALARM_UNATTENDED):
                if not tr.ever_owner_near:
                    continue
                without_owner_for = (ts - tr.last_owner_near_ts) if tr.last_owner_near_ts > 0 \
                    else (ts - tr.first_seen_ts)
                if without_owner_for >= p.owner_left_sec and tr.state == St.STATIC:
                    # Переход в UNATTENDED — ВНУТРЕННИЙ (нужен для таймера тревоги).
                    # Событие object_left в ленту НЕ шлём (по просьбе: оставляем
                    # только реальные тревоги со снимками — unattended/removed/missing).
                    tr.state = St.UNATTENDED
                    tr.unattended_since_ts = ts
                    tr.object_left_emitted = True

            if tr.state == St.UNATTENDED:
                unattended_for = ts - tr.unattended_since_ts
                if unattended_for >= p.abandon_time_sec and not tr.raised_unattended:
                    tr.state = St.ALARM_UNATTENDED
                    tr.raised_unattended = True
                    events.append(_event("object_unattended", camera_id, video_pos_ms, tr,
                                         f"unattended_for={unattended_for:.6f}"))

        # ---- Исчезновения ----
        to_drop = []
        seen_set = set(seen_ids)
        for tid, tr in tracks.items():
            if tid in seen_set:
                continue
            if persons and person_overlaps_bbox(tr.last_bbox, persons, fw, fh):
                tr.last_seen_ts = ts
                continue
            silent_for = ts - tr.last_seen_ts

            # ТЗ: object_missing / object_removed срабатывают ТОЛЬКО когда трек
            # уже получил alarm-статус (красный, raised_unattended). Жёлтые
            # (static/unattended ещё не дошло до abandon_time) тихо исчезают.
            if (tr.raised_unattended and not tr.raised_removed and not tr.raised_missing
                    and silent_for >= p.disappear_grace_sec):
                recent_interaction = (tr.last_owner_near_ts > 0
                                      and (tr.last_seen_ts - tr.last_owner_near_ts) <= p.owner_left_sec)
                if recent_interaction:
                    tr.state = St.ALARM_REMOVED
                    tr.raised_removed = True
                    events.append(_event("object_removed", camera_id, video_pos_ms, tr,
                                         f"removed_after_interaction silent_for={silent_for:.6f}"))
                else:
                    tr.state = St.ALARM_MISSING
                    tr.raised_missing = True
                    events.append(_event("object_missing", camera_id, video_pos_ms, tr,
                                         f"silent_for={silent_for:.6f}"))

            if tr.state == St.CANDIDATE and silent_for >= CANDIDATE_DROP_SILENT_SEC:
                to_drop.append(tid)
            elif (tr.raised_removed or tr.raised_missing) and silent_for > p.disappear_grace_sec * 2.0:
                to_drop.append(tid)
            elif tr.was_confirmed and not tr.raised_unattended and silent_for > max(20.0, p.disappear_grace_sec * 4.0):
                # подтверждённый, но красным не стал — тихо отбрасываем
                to_drop.append(tid)
            elif not tr.was_confirmed and silent_for > max(22.0, p.disappear_grace_sec * 3.0):
                to_drop.append(tid)

        for tid in to_drop:
            tracks.pop(tid, None)

        # ---- Потолок активных треков ----
        max_tracks = max(64, p.max_active_tracks)
        if len(tracks) > max_tracks:
            evict = [(t.last_seen_ts, tid) for tid, t in tracks.items()
                     if not (t.raised_unattended or t.raised_removed or t.raised_missing)]
            evict.sort()
            overflow = len(tracks) - max_tracks
            for _, tid in evict:
                if overflow <= 0:
                    break
                tracks.pop(tid, None)
                overflow -= 1

        return events

    def tracks_snapshot(self, now_ts) -> list:
        out = []
        # static/unattended → жёлтый (замечен, ждём сценарий); alarm_* → красный.
        state_name = {
            St.STATIC: "static",
            St.UNATTENDED: "unattended",
            St.ALARM_UNATTENDED: "alarm_unattended",
            St.ALARM_REMOVED: "alarm_removed",
            St.ALARM_MISSING: "alarm_missing",
        }
        for t in self.tracks.values():
            if t.state in (St.CANDIDATE, St.NONE):
                continue
            out.append({
                "id": int(t.track_id),
                "cls": t.cls_name,
                "state": state_name.get(t.state, "candidate"),
                "bbox": [float(t.last_bbox[0]), float(t.last_bbox[1]),
                         float(t.last_bbox[2]), float(t.last_bbox[3])],
                "conf": float(t.last_conf),
                "static_for_sec": max(0.0, now_ts - t.static_since_ts) if t.static_since_ts > 0 else 0.0,
                "unattended_for_sec": max(0.0, now_ts - t.unattended_since_ts) if t.unattended_since_ts > 0 else 0.0,
                "alarm": t.state in (St.ALARM_UNATTENDED, St.ALARM_REMOVED, St.ALARM_MISSING),
            })
        return out
