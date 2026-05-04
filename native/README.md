# Integra native (C++ / CUDA)

Целевой рантайм без Python: **декод (OpenCV) → преподготовка на GPU (`.cu`) → (TensorRT/ONNX) → тревоги в отдельный процесс**.

## Сборка

Требования: **CMake ≥ 3.20**, **OpenCV** (dev), **NVIDIA CUDA Toolkit** (для `INTEGRA_ENABLE_CUDA=ON`).

```bash
cd native
cmake -B build -DINTEGRA_ENABLE_CUDA=ON
cmake --build build -j
# бинарники: build/integra-pipeline, build/integra-alarmd
```

Без GPU:

```bash
cmake -B build -DINTEGRA_ENABLE_CUDA=OFF
cmake --build build -j
```

## Запуск демо

Терминал 1 — приём тревог (JSON lines):

```bash
./build/integra-alarmd --listen 0.0.0.0:9090 --log logs/native_alerts.jsonl
```

Терминал 2 — пайплайн:

```bash
./build/integra-pipeline --video /path/file.mp4 --alarm-sink 127.0.0.1:9090 --demo-alarm
```

Параметры инференса (пока **stub**, без весов):

```bash
./build/integra-pipeline --video file.mp4 --engine stub --imgsz 640 --model /path/model.onnx
```

Проверка цепочки **IoU-трекер → SceneAnalyzer → alarmd** без весов:

```bash
./build/integra-alarmd --listen 127.0.0.1:9090 &
./build/integra-pipeline --video data/clips/foo.mp4 --alarm-sink 127.0.0.1:9090 \
  --synth-detect --imgsz 640
```

Флаг **`--synth-detect`** подставляет один статический bbox класса `bottle`, если модель ничего не вернула; через реальное время сработает FSM abandoned/disappeared (как в Python `analyzer.py`). **`--person-class`** — id класса «человек» (COCO: 0).

- `--imgsz 0` — без resize, полный кадр в сеть (дороже по VRAM/времени).
- `--engine tensorrt|onnx` — зарезервировано; сейчас используется тот же stub до подключения runtime.

## Куда встраивать детектор

- **`IInferenceEngine`** (`include/integra/inference_engine.hpp`) — реализация `infer()` получает NCHW float (GPU или CPU).
- **`src/inference_stub.cpp`** — заменить на `tensorrt_engine.cpp` / `onnx_engine.cpp` и зарегистрировать в `make_inference_engine()`.
- **`IouTracker`** — временная замена BoT-SORT; затем можно подставить BYTETrack/BoT-SORT на C++.
- **`SceneAnalyzer`** — порт FSM из `app/analyzer.py` (abandoned / disappeared).
- **`src/pipeline.cpp`** — масштабирование bbox → трекер → разделение person/object → `SceneAnalyzer::ingest` → TCP в **integra-alarmd**.

## Старый Python-стек

Каталог `app/` остаётся для миграции и веб-прототипа; продуктивный путь — **integra-pipeline + integra-alarmd** и внешние интеграции по JSONL/TCP.
