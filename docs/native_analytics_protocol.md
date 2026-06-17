# Native analytics protocol (Python ↔ C++)

Version: **v1**

Transport: **TCP**, request/response per-frame, **JSON line + binary payload**.

This protocol is designed to be:

- debuggable (JSON),
- robust to forward-compatible fields,
- efficient for frames (raw BGR bytes).

## Connection lifecycle

1) Python connects to `native_analytics.addr` (e.g. `127.0.0.1:9909`).
2) Python sends `hello` once.
3) Server replies `hello_ok`.
4) For each frame Python sends `frame` + BGR payload.
5) Server replies `result` (same order).
6) Optional: Python may send `reset` to clear tracker/FSM state without reconnect.

All messages are one JSON object serialized as UTF-8 and terminated by `\n`.

Binary payload framing is:

- after the JSON line of type `frame`, client sends `len_u32_le` (4 bytes) and then `len` raw bytes.

## Messages

### hello (client → server)

```json
{"type":"hello","v":1,"camera_id":"main","engine":"tensorrt","model_path":"C:/.../yolo.engine","input_size":640,
 "postprocess":{"conf":0.22,"iou":0.5,"nms_iou":0.45,"max_det":300,"person_class_id":0},
 "analyzer":{"static_displacement_px":7.0,"static_window_sec":3.0,"abandon_time_sec":15.0,"owner_proximity_px":180.0,
            "owner_left_sec":5.0,"disappear_grace_sec":4.0,"min_object_area_px":100.0,"centroid_history_maxlen":72}}
```

Notes:
- Unknown fields must be ignored.
- Optional top-level numeric fields for `integra-analyticsd` IouTracker (defaults: iou 0.35, max_missed 10, soft centroid on): `tracker_iou`, `tracker_max_missed`, `tracker_soft_centroid` (0 or 1).

### hello_ok (server → client)

```json
{"type":"hello_ok","v":1,"ok":true,"message":"ready"}
```

### frame (client → server)

```json
{"type":"frame","v":1,"frame_id":123,"pos_ms":4567.0,"width":1920,"height":1080,"format":"bgr24"}
```

Then:
- `len_u32_le` = `width*height*3`
- payload: raw BGR bytes, row-major

### reset (client → server)

```json
{"type":"reset","v":1}
```

Purpose:
- reset tracker/analyzer state on seek/open,
- keep TCP connection and loaded inference engine alive.

### reset_ok (server → client)

```json
{"type":"reset_ok","v":1,"ok":true}
```

### result (server → client)

```json
{
  "type":"result","v":1,
  "frame_id":123,"pos_ms":4567.0,
  "metrics":{"preprocess_ms":1.2,"infer_ms":8.4,"postprocess_ms":0.7,"tracker_ms":0.2,"analyzer_ms":0.1},
  "detections":[{"track_id":7,"cls_id":-1,"cls_name":"object","confidence":0.42,"bbox":[10,20,30,40]}],
  "persons":[{"track_id":2,"cls_id":0,"cls_name":"person","confidence":0.88,"bbox":[...]}],
  "tracks":[{"id":7,"cls":"object","state":"alarm_unattended","bbox":[...],"conf":0.42,
             "static_for_sec":12.3,"unattended_for_sec":7.1,"alarm":true}],
  "events":[{"type":"object_unattended","track_id":7,"cls_id":-1,"cls_name":"object","confidence":0.42,
             "bbox":[...],"note":"unattended_for=..."}]
}
```

Notes:
- Два контура (ТЗ): `persons` — class-based (YOLO cls=0, ByteTrack, устойчивые ID);
  `detections` — class-agnostic объекты сцены из FrameDiffDetector (`cls_id=-1`, `cls_name="object"`).
- `events[].type` ∈ `person_interaction` | `object_left` | `object_unattended` |
  `object_removed` | `object_missing` — событие определяется поведением во времени, не классом.
- `tracks[].state` ∈ `unattended` | `alarm_unattended` | `alarm_removed` | `alarm_missing`
  (candidate/static на оверлей не выводятся). `tracks` — снимок для UI `/api/info.tracks`.
- Server may omit optional metric keys; client must treat missing as `0`.

