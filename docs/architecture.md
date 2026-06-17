# Архитектура видеоаналитики (YOLOv11, Full HD, real-time)

Система построена по ТЗ как набор независимых контуров: люди обрабатываются как
**class-based** сущности с устойчивыми track ID, остальные предметы сцены — как
**class-agnostic** объекты, выявляемые по пространственно-временным изменениям кадра.
Событие определяется **поведением во времени**, а не именем YOLO-класса.

## Конвейер

```
ИСТОЧНИК (file/RTSP) ──► video-bridge (Rust+OpenCV, TCP :9876, stream, без накопления)
                                  │ BGR кадр + pts_ms
                                  ▼
            ┌──────────────────────────────────────────────┐
            │ M1 YOLO perception core (integra_ffi)          │  весь кадр, imgsz=640, FP16
            │ IStreamContext::infer (TensorRT / OpenCV-DNN)  │
            └──────────────────────────────────────────────┘
              │ люди cls=0                     │ весь кадр (BGR)
              ▼                                ▼
   ┌───────────────────────┐      ┌──────────────────────────────────┐
   │ M2 person tracking     │      │ M3 class-agnostic candidates       │
   │ apply_frame_filter →    │      │ FrameDiffDetector (Sobel+pixel)   │
   │ ByteTracker (Kalman+    │      │ → подавление зон людей            │
   │ two-stage) устойчивые ID│      │ → IouTracker (регионы, cls=-1)    │
   └───────────────────────┘      └──────────────────────────────────┘
              │ persons (track_id)            │ objects (cls=-1, "object")
              └───────────────┬───────────────┘
                              ▼
            ┌──────────────────────────────────────────────┐
            │ M4 temporal event analytics (SceneAnalyzer)    │  поведенческая FSM
            └──────────────────────────────────────────────┘
                              │ AlarmEvent (callback "event")
                              ▼  ── ГРАНИЦА: mpsc StreamMessage::Event ──
            ┌──────────────────────────────────────────────┐
            │ M5 independent alarm processor (gateway)       │  dedup, snapshot, журнал, WS
            └──────────────────────────────────────────────┘
                              │
                              ▼
            ┌──────────────────────────────────────────────┐
            │ M6 visualization (backend_gateway + static/)   │  треки людей, ID, тревоги
            └──────────────────────────────────────────────┘
```

## Модули и файлы

| Модуль | Реализация |
|---|---|
| M1 perception | `native/src/integra_ffi.cpp` (`IStreamContext::infer`), движки `tensorrt_engine.cpp` / `opencv_dnn_engine.cpp` |
| M2 person tracking | `native/src/byte_track.cpp` (`ByteTracker`) — настоящий ByteTrack |
| M3 object candidates | `native/src/frame_diff_detector.cpp` + `iou_tracker.cpp` |
| M4 event analytics | `native/src/scene_analyzer.cpp` (`SceneAnalyzer`) |
| M5 alarm processor | `runtime-core/src/integra/stream.rs` (граница) + `bin/backend_gateway.rs` (consumer) |
| M6 visualization | `runtime-core/src/integra/preview_encode.rs` + `static/app.js` |

## M2 — ByteTrack (class-based, только люди)

`ByteTracker` (`byte_track.hpp/.cpp`) — каноничный ByteTrack:
- Kalman (модель xyah, постоянная скорость), предсказание боксов;
- двухстадийная ассоциация BYTE: сначала high-conf детекции, затем low-conf «спасают» треки;
- венгерское назначение (Jonker–Volgenant) по cost = 1 − IoU;
- жизненный цикл `New → Tracked → Lost → Removed`, реактивация Lost в пределах `track_buffer`.

Параметры — `config.json → model.tracker` (`high_thresh`, `low_thresh`, `new_thresh`,
`match_thresh`, `track_buffer`); `frame_rate` берётся из `pipeline.target_fps`.

## M3 — class-agnostic объекты

`FrameDiffDetector` сравнивает кадр с предыдущим (пиксельная разница + градиент Собеля),
находит изменившиеся регионы. Регионы, перекрытые людьми (IoU > 0.2), подавляются — это
зона контура M2. Оставшиеся → `Detection{ class_id = -1, cls_name = "object" }` →
`IouTracker` (устойчивость региона между кадрами). Имя YOLO-класса никогда не присваивается.

Параметры — `config.json → analyzer.frame_diff_*`, `use_frame_diff_detector`.

## M4 — состояния и события

Состояния региона-кандидата (FSM):

| Состояние | Условие |
|---|---|
| `candidate` | регион только появился |
| `static` | смещение центроида < `static_displacement_px` за окно `static_window_sec` |
| `unattended` | static + был владелец (`ever_owner_near`), но он ушёл (`owner_left_sec`) |
| `alarm_unattended` | unattended дольше `abandon_time_sec` |
| `alarm_removed` | подтверждённый регион исчез ≤ `owner_left_sec` после взаимодействия |
| `alarm_missing` | подтверждённый регион исчез без взаимодействия (`disappear_grace_sec`) |

События:

| Событие | Когда |
|---|---|
| `person_interaction` | человек вошёл в зону подтверждённого объекта (троттлинг) |
| `object_left` | static-объект с историей владельца перешёл в unattended |
| `object_unattended` | unattended дольше `abandon_time_sec` — тревога |
| `object_removed` | объект исчез вскоре после взаимодействия — «забрали» |
| `object_missing` | объект исчез без взаимодействия — пропал |

Все пороги — в **секундах от ts**, не в кадрах: снижение частоты аналитики не ломает события.

## M5 — независимый тревожный движок

`SceneAnalyzer::ingest` лишь возвращает `AlarmEvent`. FFI-callback → safe-Rust
(`FrameCollector`) → `mpsc` `StreamMessage::Event`. Это и есть граница: consumer
(`backend_gateway`) делает dedup/cooldown, прикрепляет JPEG-снимок **только** к тревожным
событиям (`object_unattended` / `object_removed` / `object_missing` — см.
`stream.rs::alarm_needs_snapshot`), пишет журнал и шлёт в WebSocket. Ядро аналитики не
зависит от обработки тревог.

## Бюджет ОЗУ ≤ 500 МБ

- YOLOv11n/s, `imgsz=640`, FP16; один FrameDiffDetector (дубль в SceneAnalyzer удалён);
- кольцевые буферы фиксированной длины (`centroid_history_maxlen`, `frame_diff_buffer_size`);
- zero-copy кадра (`cv::Mat` поверх буфера без копии); снапшот = crop bbox, не ринг кадров
  (`forensic_ring_max=0`); rate-limit превью (`preview_*`);
- `max_active_tracks` — потолок треков; пороги по времени → можно прореживать аналитику.
- Маркеры бюджета: `pipeline.memory_chart_warning_bytes` (400 МБ) / `..._critical_bytes` (500 МБ).
