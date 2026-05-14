# Integra Native

Детектор оставленных / пропавших предметов, целиком на C++ / Rust.
Python в рантайме не используется — `run.py` это только launcher двух
нативных процессов.

Два типа тревог:

- **`alarm_abandoned`** — предмет стал статичным, владелец рядом не появляется → тревога;
- **`alarm_disappeared`** — ранее тревожный предмет исчез из кадра.

---

## 1. Архитектура

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Браузер (static/index.html, app.js, styles.css)                         │
│       │  HTTP/REST + WebSocket + MJPEG                                   │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
        ┌────────────────────▼────────────────────────────────┐
        │ backend_gateway  (Rust / axum, runtime-core)        │
        │  • /api/info, /api/open, /api/metrics, /api/events  │
        │  • /video_feed (MJPEG c bbox-overlay)               │
        │  • /ws (broadcast событий и status тиков)           │
        └────┬──────────────────────────────┬─────────────────┘
             │ TCP :9876 (BGR кадры)         │ FFI (C ABI)
             │                                │
   ┌─────────▼──────────┐         ┌──────────▼──────────────────────┐
   │ video-bridge       │         │ integra_ffi  (C++ shared lib)   │
   │ (Rust + opencv)    │         │  • SharedEngine (TRT / OpenCV)  │
   │  HW-accel decode   │         │  • IouTracker + SceneAnalyzer   │
   │  → BGR frames      │         │  • frame_filter (anti-noise)    │
   └────────────────────┘         │  • IntegraPipeline (per-stream) │
                                  └─────────────────────────────────┘
```

Все три компонента — независимые бинарники / разделяемые библиотеки.
Pipeline вычислительной части (декод → preprocess → inference → NMS →
tracker → FSM) исполняется в C++ / CUDA, без переключений в Python.

## 2. Структура репозитория

```
.
├── native/              # C++ ядро: integra_core, integra_ffi, analyticsd, alarmd, pipeline
│   ├── include/integra/ # Публичные заголовки (frame_filter.hpp, inference_engine.hpp, ...)
│   ├── src/             # Реализации (shared_engine.cpp, tensorrt_engine.cpp, integra_ffi.cpp, ...)
│   ├── CMakeLists.txt
│   └── README.md        # Подробности по C++ ядру
├── runtime-core/        # Rust workspace member: backend_gateway + integra Rust bindings
│   ├── src/integra/     # FFI обёртки (ffi.rs, pipeline.rs, stream.rs, events.rs)
│   ├── src/bin/backend_gateway.rs
│   └── Cargo.toml
├── video-bridge/        # Rust workspace member: TCP-сервер с BGR-кадрами (opencv-rust)
├── static/              # Frontend (index.html, app.js, styles.css)
├── data/                # Видеофайлы (mkv, mp4, ...)
├── logs/snapshots/      # JPEG snapshots алармовых событий (создаётся автоматически)
├── models/              # Веса (yolo11n.onnx, yolo11n.engine, ...)
├── config.json          # Параметры pipeline (engine, conf/iou, analyzer thresholds)
├── run.py               # Python-launcher: запускает video-bridge + backend_gateway
└── README.md
```

## 3. Сборка

### 3.1. Нативная библиотека `integra_ffi`

**Windows (MSBuild):**
```powershell
cmake -S native -B native\build-msvc -G "Visual Studio 17 2022" -A x64
cmake --build native\build-msvc --config RelWithDebInfo --target integra_ffi
```
Результат: `native\build-msvc\RelWithDebInfo\integra_ffi.dll`.

**Linux:**
```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build native/build --target integra_ffi -j
```
Если **`integra_ffi`** собран с **`INTEGRA_WITH_TENSORRT=ON`**
(`INTEGRA_HAS_TENSORRT=1`), то для движка `kind="tensorrt"` используется
`SharedTRTEngine` + per-stream `IExecutionContext`. Иначе доступны `opencv` / `onnx` (OpenCV DNN и т.д.). См. **`native/scripts/build_engine_msvc.ps1`** для Windows.

Детали — в [`native/README.md`](native/README.md).

### 3.2. Rust workspace

```bash
cargo build --release --workspace
```

Для `video-bridge` нужен libclang (зависимость `opencv-rust` через
bindgen):
- Windows: `pip install libclang` → `LIBCLANG_PATH=.venv\Lib\site-packages\clang\native`;
- Linux: `apt install libclang-dev`.

### 3.3. TensorRT: сборка `.engine` из `.onnx` (C++ `integra_trt_bake`, без Python)

Рантайм ожидает **уже собранный** serialized engine (см. `native_analytics.model_path`).

- Утилита **`integra_trt_bake`** (сборка натива с **`INTEGRA_WITH_TENSORRT=ON`**):  
  `integra_trt_bake --onnx models/yolo11n.onnx --out models/yolo11n_fp16.engine --fp16 --workspace-mb 4096`
- Кеш (Linux): **`native/scripts/trt_bake_cached.sh`** (sha256 + compute capability).
- **Windows (MSVC):** **`native/scripts/build_engine_msvc.ps1`** — CMake + `integra_trt_bake` + `integra_ffi` (параметры `-OpenCvDir`, `-TensorRtRoot` или переменные `INTEGRA_OPENCV_DIR` / `TENSORRT_ROOT`).

Подробности — [native/README.md](native/README.md) (раздел «TensorRT» и «Windows + CUDA + TensorRT»).

## 4. Запуск

**Сборка всего (native TRT DLL + Rust gateway + video-bridge), из корня:**

```powershell
.\scripts\build_all.ps1
```

Требуется уже сконфигурированный CMake-каталог `native\build-msvc-trt-user` (см. `native\README.md`). Скрипт копирует `integra_ffi.dll` в `runtime-core\target\release\`. Для `video-bridge` выставляются `OPENCV_LINK_*` под `C:\build\opencv\x64\vc17\lib` (имя `opencv_world*.lib` ищется автоматически).

**Запуск gateway + bridge одним окном:**

```powershell
.\scripts\run_stack.ps1
```

**Windows (только gateway, TensorRT):** из корня репозитория:

```powershell
.\scripts\run_backend_gateway.ps1
```

Скрипт выставляет `INTEGRA_PROJECT_ROOT`, `INTEGRA_FFI_PATH`, дописывает в `PATH` каталоги **TensorRT `bin`**, **CUDA `bin`**, **OpenCV `bin`** (пути в начале `scripts/run_backend_gateway.ps1` при необходимости поправьте). Нужны собранные `integra_ffi.dll` и `models/yolo11n_fp16.engine` (см. §3).

**Через Python-launcher (как раньше):**

```bash
python run.py --release
```

`run.py` найдёт `integra_ffi.dll/.so`, выставит `INTEGRA_FFI_PATH` и
запустит:

1. `video-bridge` (Rust) — TCP :9876, отдаёт BGR кадры из видео;
2. `backend_gateway` (Rust) — HTTP :8000, грузит `integra_ffi`, рулит
   FFI Pipeline, отдаёт UI.

UI откроется на [http://127.0.0.1:8000](http://127.0.0.1:8000).

Полезные флаги:

| Флаг              | Что делает                                          |
|-------------------|-----------------------------------------------------|
| `--release`       | `cargo run --release` (рекомендуется для прод)       |
| `--port 8095`     | Сменить порт UI                                      |
| `--bridge-addr`   | Сменить адрес TCP video-bridge                       |
| `--no-bridge`     | Запустить gateway без bridge (smoke без видео)       |

Переменные среды:

| ENV                          | Назначение                                       |
|------------------------------|--------------------------------------------------|
| `INTEGRA_FFI_PATH`           | Путь к `integra_ffi.{dll,so}` (берётся run.py)   |
| `INTEGRA_BACKEND_HOST/PORT`  | Где слушает gateway                              |
| `INTEGRA_VIDEO_BRIDGE_ADDR`  | TCP-адрес video-bridge                           |
| `INTEGRA_ENGINE_KIND`        | `tensorrt` / `opencv` / `onnx` / `stub`          |
| `INTEGRA_PROJECT_ROOT`       | Корень проекта (для static/, data/, logs/)      |
| `RUST_LOG`                   | Уровень логов axum / runtime-core                |

## 5. REST / WS API (для фронтенда)

| Метод | Путь                          | Описание                                          |
|------:|-------------------------------|---------------------------------------------------|
|  GET  | `/`, `/static/*`              | Фронтенд (HTML/JS/CSS)                            |
|  GET  | `/health`                     | Liveness probe                                    |
|  GET  | `/api/info`                   | Снапшот стрима: tracks, stats, fps, ...           |
|  POST | `/api/open`                   | `{path}` — открыть видео, запустить pipeline     |
|  POST | `/api/play`, `/api/pause`     | Управление playback (форвардятся в video-bridge) |
|  POST | `/api/seek`                   | `{"frame":N}` — seek по номеру кадра             |
|  GET  | `/api/metrics`                | process RSS/CPU + system + pipeline EMA          |
|  GET  | `/api/files`                  | Список видео в `data/`                            |
|  GET  | `/api/streams`                | Список стримов                                    |
|  GET  | `/api/events?limit=N`         | Лог последних событий                             |
|  GET/PUT | `/api/settings`            | Чтение / deep-merge правка `config.json`         |
|  GET  | `/video_feed`                 | MJPEG c bbox-overlay (multipart/x-mixed-replace) |
|  GET  | `/video_snapshot`             | JPEG последнего кадра                            |
|  GET  | `/logs/snapshots/main/*.jpg`  | JPG алармовых событий                            |
|  WS   | `/ws`                         | hello → status (каждые 500мс) + event (по факту) |

WS-сообщения:

```jsonc
{"type":"hello",  "stream_id":"main", "info":{...full info...}}
{"type":"status", "stream_id":"main", "info":{...full info...}}
{"type":"event",  "stream_id":"main", "event":{"type":"alarm_abandoned", "snapshot_path":"/logs/snapshots/main/...jpg", ...}}
```

## 6. Конфигурация — `config.json`

Минимально необходимое:

```json
{
  "native_analytics": {
    "engine": "tensorrt",
    "model_path": "models/yolo11n_fp16.engine",
    "input_size": 640
  },
  "model": {
    "conf": 0.25,
    "iou": 0.45,
    "person_class": 0,
    "object_classes": [24, 26, 28],
    "min_box_size_px": 20
  },
  "analyzer": {
    "static_displacement_px": 12,
    "static_window_sec": 3.0,
    "abandon_time_sec": 12.0,
    "owner_proximity_px": 220.0,
    "owner_left_sec": 5.0,
    "disappear_grace_sec": 5.0,
    "min_object_area_px": 400
  }
}
```

`gateway` мёрджит `PUT /api/settings` deep-merge'ом и атомарно записывает
файл (`config.json.tmp` → rename).

## 7. Подмодули — отдельные README

- [`native/README.md`](native/README.md) — C++ ядро, разбиение по бинарям, расширение.
- [`runtime-core/README.md`](runtime-core/README.md) — Rust workspace member, FFI обёртки.

## 8. Состояние шагов рефакторинга

| # | Шаг                                                                   | Статус |
|---|-----------------------------------------------------------------------|:------:|
| 1 | Shared TRT engine + per-stream IExecutionContext                      | done   |
| 2 | Модуль `frame_filter` (антишум, общий для analyticsd и FFI)           | done   |
| 3 | `integra_ffi.h/.cpp` + Rust bindings (`runtime-core::integra::*`)     | done   |
| 4 | `backend_gateway` поверх FFI + video-bridge (UI / WS / MJPEG)         | done   |
| 5 | Snapshot writer + удаление Python `app/`/`services/`                  | done   |
| 6 | Seek / pause / play через video-bridge (расширение протокола)         | done   |
| 7 | TensorRT: `integra_trt_bake` + кеш `trt_bake_cached.sh` (ONNX→FP16 engine)  | done   |
| 8 | Бенчмарки + parity-тесты C++↔ожидаемый поведение                      | todo   |
| 9 | CI (Windows/Linux) + docker compose                                   | todo   |
