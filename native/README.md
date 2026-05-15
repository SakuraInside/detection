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
│   ├── trt_bake_cached.sh      # Кеш .engine: sha256 (16 hex) + compute capability + FP16 (Linux)
│   └── build_engine_msvc.ps1   # Windows: cmake + integra_trt_bake (пути OpenCV/TRT внутри скрипта)
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

### Windows + CUDA + TensorRT: `integra_trt_bake` и `integra_ffi` (C++), Rust без Python

Цепочка production: **офлайн** утилита **`integra_trt_bake`** (C++, `src/trt_bake_main.cpp`) → файл **`.engine`** → рантайм **`integra_ffi.dll`** с **`INTEGRA_HAS_TENSORRT`** → **`backend_gateway`** (Rust) грузит DLL и передаёт `engine_kind=tensorrt` + путь к `.engine` через существующий FFI (`IntegraConfig`).

1. Нужны **OpenCV** (с `dnn`) и **TensorRT SDK** (каталог с `include/NvInfer.h` и `lib`).
2. CMake **обязан** видеть OpenCV: `-DOpenCV_DIR=...` (каталог, где лежит `OpenCVConfig.cmake`).
3. Сборка с флагами:

```powershell
cd native
$OpenCVDir = "C:\build\opencv"          # замените на свой
$TrtRoot   = "C:\path\TensorRT-10.x"    # замените на свой

cmake -B build-msvc-trt -G "Visual Studio 17 2022" -A x64 `
  "-DOpenCV_DIR=$OpenCVDir" `
  -DINTEGRA_ENABLE_CUDA=ON `
  -DINTEGRA_WITH_TENSORRT=ON `
  "-DTENSORRT_ROOT=$TrtRoot"

cmake --build build-msvc-trt --config Release --target integra_trt_bake --target integra_ffi
```

4. Перед запуском **`integra_trt_bake.exe`** на Windows добавьте в **`PATH`** каталог **`%TENSORRT_ROOT%\bin`** (там `nvinfer_10.dll`, `nvonnxparser_10.dll`). Каталог **`lib`** нужен линкеру при сборке. При необходимости — также **`CUDA\v13.x\bin`**.

5. Сборка `.engine` из корня репозитория (`models/yolo11n.onnx` → тот же путь, что в `config.json`):

```powershell
cd ..
.\native\build-msvc-trt\Release\integra_trt_bake.exe `
  --onnx models\yolo11n.onnx --out models\yolo11n_fp16.engine --fp16 --workspace-mb 4096
```

Готовый скрипт-обёртка (пути вверху файла): `native/scripts/build_engine_msvc.ps1`.

## TensorRT: ONNX → `.engine` (production)

Рантайм (`integra_ffi`, `--engine tensorrt`) загружает **сериализованный** engine. ONNX → plan собирайте **офлайн** на машине с TensorRT и GPU — **только через `integra_trt_bake`** (C++); Python в этом репозитории для bake не используется.

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

### YOLO11s (или другой `.pt`): экспорт ONNX + bake одной командой

На Debian/Ubuntu **PEP 668** запрещает `pip install` в системный Python — используйте **venv** и пересоберите натив после обновления CMake (PIC для `libintegra_ffi.so`):

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r tools/requirements-trt-export.txt
rm -rf native/build
./native/build_integra.sh
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH}"
chmod +x native/scripts/export_yolo_trt_engine.sh
./native/scripts/export_yolo_trt_engine.sh models/yolo11s.pt 640 4096
# если нет models/yolo11s.pt — Ultralytics скачает: ./native/scripts/export_yolo_trt_engine.sh yolo11s.pt 640 4096
# → models/yolo11s_fp16.engine (как в config.json → native_analytics.model_path)
```

Аргументы: `[weights.pt] [imgsz] [workspace_mb]`. Переменная **`INTEGRA_EXPORT_PYTHON`** переопределяет интерпретатор (по умолчанию берётся `.venv/bin/python`, если есть).

`imgsz` должен совпадать с `native_analytics.input_size` и желательно с `model.imgsz` в Python.

### Ошибка `libnvinfer_builder_resource_smXX.so... cannot open shared object file`

При **`buildSerializedNetwork`** TensorRT подгружает **`libnvinfer_builder_resource_sm<ваш_CC>.so`** из каталога **`${TENSORRT_ROOT}/lib`** (например **sm89** для RTX 40xx). Скрипты **`trt_bake_cached.sh`** / **`export_yolo_trt_engine.sh`** сами дописывают в **`LD_LIBRARY_PATH`** пути **`lib`** и **`lib/x86_64-linux-gnu`**.

Если ошибка остаётся:

1. Убедитесь, что переменная **`TENSORRT_ROOT`** указывает на **полный** распакованный Linux Tarball с **developer.nvidia.com** (не обрезанный каталог).
2. Проверьте наличие файла:  
   `ls "${TENSORRT_ROOT}/lib"/libnvinfer_builder_resource_sm*.so`
3. Версия TensorRT должна совпадать с той, с которой собран **`integra_trt_bake`** (например **10.16.1**).

## Запуск

`integra-analyticsd` (опционально, для отладки TCP-протокола; основной UI — Rust `backend_gateway`):

```bash
./build/integra-analyticsd --listen 127.0.0.1:9909 \
  --engine opencv --model models/yolo11s.onnx --imgsz 640
# или --engine tensorrt --model models/yolo11s_fp16.engine
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
