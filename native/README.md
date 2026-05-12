# Integra Native (C++ / CUDA)

Нативный рантайм пайплайна детекции: **декод (OpenCV) → препроцесс (CUDA/CPU) → инференс (TensorRT / ONNX Runtime / OpenCV DNN / stub) → IoU-трекер → FSM `SceneAnalyzer` → события в `integra-alarmd` или FFI / TCP-клиенты**.

Состоит из библиотеки `integra_core` и исполняемых утилит (см. таблицу ниже + опционально `integra_trt_bake` при `INTEGRA_WITH_TENSORRT=ON`).

| Бинарник              | Источник                  | Назначение                                                                 |
| --------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `integra-pipeline`    | `src/main.cpp`            | End-to-end CLI: видеофайл → детекция → JSON-line alarm sink.               |
| `integra-analyticsd`  | `src/analyticsd_main.cpp` | TCP-сервис (порт 9909): протокол `docs/native_analytics_protocol.md` (Python UI убран; можно использовать для отладки). |
| `integra-alarmd`      | `alarmd/main.cpp`         | Приёмник тревог (TCP JSON-lines → файл / stdout, без зависимостей).        |
| `integra_trt_bake`    | `src/trt_bake_main.cpp`   | **С TensorRT:** офлайн сборка `.engine` из `.onnx` (FP16, workspace).      |

## Структура каталога

```
native/
├── CMakeLists.txt              # Сборка integra_core + бинарники + integra_ffi (+ integra_trt_bake при TRT)
│                               # Опции: INTEGRA_ENABLE_CUDA, INTEGRA_WITH_ONNXRUNTIME,
│                               #        INTEGRA_ORT_CUDA, INTEGRA_WITH_TENSORRT
├── build_integra.sh            # Linux helper: auto-detect TensorRT/ONNXRT + cmake + build
├── scripts/
│   └── trt_bake_cached.sh      # Кеш .engine: sha256 (16 hex) + compute capability + FP16
├── README.md                   # этот файл
│
├── include/integra/            # Публичный C++ API (используется analyticsd и pipeline)
│   ├── types.hpp               # FrameMeta, BBoxXYXY, Detection, DetectionBatch, AlarmEvent
│   ├── geom.hpp                # Геометрия (IoU и пр.)
│   ├── inference_engine.hpp    # IInferenceEngine + make_inference_engine() (фабрика)
│   ├── yolo_postprocess.hpp    # PostprocessParams, decode + NMS, selftest
│   ├── gpu_preprocess.hpp      # GpuLetterboxPrep: BGR→NCHW float (CUDA path / CPU fallback)
│   ├── iou_tracker.hpp         # Лёгкий IoU-трекер (временная замена BoT-SORT)
│   ├── scene_analyzer.hpp      # AnalyzerParams + FSM abandoned / unattended / disappeared
│   ├── alarm_sink.hpp          # TCP-клиент JSON-lines → integra-alarmd
│   ├── video_source.hpp        # OpenCV VideoCapture wrapper
│   └── pipeline.hpp            # Склейка всех этапов в один проход
│
├── src/                        # Реализации API + точки входа бинарников
│   ├── main.cpp                # integra-pipeline (CLI: --video, --engine, --alarm-sink, ...)
│   ├── analyticsd_main.cpp     # integra-analyticsd (TCP-протокол: hello / frame / reset)
│   │                           # Здесь же anti-noise FrameFilterConfig
│   │                           # (min_conf, min_conf_person, min_box_px, max_detections, ...)
│   ├── pipeline.cpp            # decode → preprocess → infer → tracker → analyzer → sink
│   ├── video_source.cpp        # Чтение кадров OpenCV-ом (file / stream)
│   ├── gpu_preprocess.cpp      # CUDA-вариант + CPU fallback (letterbox + нормализация)
│   ├── yolo_postprocess.cpp    # Декод выхода YOLO + class-aware NMS
│   ├── iou_tracker.cpp         # Сопоставление треков по IoU
│   ├── scene_analyzer.cpp      # FSM abandoned / unattended / disappeared (порт логики analyzer)
│   ├── alarm_sink.cpp          # TCP-клиент к integra-alarmd
│   │
│   ├── trt_bake_main.cpp       # integra_trt_bake: ONNX → serialized engine (nvonnxparser)
│   ├── inference_stub.cpp      # Заглушка движка (никогда не падает, для smoke-tests)
│   ├── opencv_dnn_engine.cpp   # --engine opencv : OpenCV DNN + ONNX-веса
│   ├── onnx_engine.cpp         # --engine onnx   : ONNX Runtime (опц. CUDA EP)
│   └── tensorrt_engine.cpp     # --engine tensorrt: сериализованный .engine
│
├── cuda/
│   └── preprocess.cu           # CUDA kernel: BGR→NCHW float, letterbox на устройстве
│
├── alarmd/
│   └── main.cpp                # integra-alarmd: TCP JSON-lines → лог-файл + stdout
│
└── build-msvc/                 # Артефакты MSVC (cmake -G "Visual Studio ..."); в VCS не входит
    └── RelWithDebInfo/         # → integra-analyticsd.exe и пр.
```

## Точки расширения

| Куда подключать                | Файл                                                              |
| ------------------------------ | ----------------------------------------------------------------- |
| Новый inference backend        | `include/integra/inference_engine.hpp` + новый `src/*_engine.cpp`, регистрация в `make_inference_engine()` |
| Замена IoU-трекера на BoT-SORT | `include/integra/iou_tracker.hpp` / `src/iou_tracker.cpp`         |
| Бизнес-правила тревог          | `include/integra/scene_analyzer.hpp` / `src/scene_analyzer.cpp`   |
| Anti-noise фильтр на кадр      | `include/integra/frame_filter.hpp` / `src/frame_filter.cpp` (общий для analyticsd и FFI) |
| Транспорт событий              | `src/alarm_sink.cpp` (текущий формат — JSON-lines поверх TCP)     |

## Зависимости

- **CMake ≥ 3.20**, **C++17**, **OpenCV** (`core imgproc videoio dnn`), **pthreads** (Linux) / `ws2_32` (Windows).
- Опционально: **CUDA Toolkit** (`INTEGRA_ENABLE_CUDA=ON`), **TensorRT** (`INTEGRA_WITH_TENSORRT=ON`), **ONNX Runtime** (`INTEGRA_WITH_ONNXRUNTIME=ON`, GPU EP — `INTEGRA_ORT_CUDA=ON`).

## Сборка

Linux (с CUDA, TensorRT / ONNXRT по наличию SDK):

```bash
bash native/build_integra.sh
# артефакты: native/build/integra-pipeline, integra-analyticsd, integra-alarmd,
#            integra_trt_bake (если INTEGRA_WITH_TENSORRT=ON), integra_ffi
```

Минимально (без GPU, только stub / OpenCV DNN):

```bash
cd native
cmake -B build -DINTEGRA_ENABLE_CUDA=OFF
cmake --build build -j
```

Windows (MSVC, как сейчас собирается `integra-analyticsd.exe`):

```powershell
cd native
cmake -B build-msvc -G "Visual Studio 17 2022" -A x64
cmake --build build-msvc --config RelWithDebInfo --target integra-analyticsd
```

## TensorRT: ONNX → `.engine` (production)

Рантайм (`integra_ffi`, `--engine tensorrt`) загружает **сериализованный** engine. ONNX → plan собирайте **офлайн** на машине с TensorRT и GPU.

**Прямой вызов:**

```bash
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH}"
./build/integra_trt_bake --onnx models/yolo11n.onnx --out models/yolo11n_fp16.engine \
  --fp16 --workspace-mb 4096
```

**Кеш по sha256 (16 hex) + compute capability:**

```bash
chmod +x native/scripts/trt_bake_cached.sh
ENGINE="$(./native/scripts/trt_bake_cached.sh "$(pwd)/models/yolo11n.onnx" 4096)"
echo "engine: ${ENGINE}"
```

Артефакты: `models/.integra_trt_cache/`. Укажите путь в `config.json` → `native_analytics.model_path`.

**Альтернатива:** `trtexec --onnx=... --saveEngine=... --fp16`.

## Запуск

`integra-analyticsd` (опционально, для отладки TCP-протокола; основной UI — Rust `backend_gateway`):

```bash
./build/integra-analyticsd --listen 127.0.0.1:9909 \
  --engine opencv --model models/yolo11n.onnx --imgsz 640
# или --engine tensorrt --model models/yolo11n_fp16.engine
```

`integra-alarmd` + `integra-pipeline` (автономная демонстрация без Python):

```bash
./build/integra-alarmd   --listen 0.0.0.0:9090 --log logs/native_alerts.jsonl
./build/integra-pipeline --video file.mp4 --alarm-sink 127.0.0.1:9090 \
  --engine opencv --model models/yolo11n.onnx --imgsz 640 --stats
# Smoke-test без модели:
./build/integra-pipeline --video file.mp4 --alarm-sink 127.0.0.1:9090 --synth-detect
```

Полезные флаги `integra-pipeline`: `--conf`, `--nms-iou`, `--num-classes`, `--person-class`, `--target-fps`, `--max-frames`, `--selftest-yolo`.

## Протокол `integra-analyticsd`

TCP, `\n`-разделённые JSON-сообщения + бинарные блоки кадров. Спецификация: `docs/native_analytics_protocol.md` в корне репо.

Кратко:

1. Клиент шлёт `hello` (`engine`, `model_path`, `input_size`, `postprocess.{conf,iou,nms_iou,...}`, `object_classes`, `min_box_size_px`, `analyzer.*`).
2. Сервер отвечает `hello_ok` (с `real_engine: true/false`).
3. На каждый кадр клиент шлёт `frame` (метаданные) + бинарный BGR-блок (`<u32 length><payload>`); сервер возвращает JSON c `detections`, `persons`, `events`.
4. `reset` сбрасывает трекер и `SceneAnalyzer`.
