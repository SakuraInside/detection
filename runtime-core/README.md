# runtime-core (Rust)

Первый каркас Rust runtime для поэтапного выноса orchestration из Python.

Что уже заложено:

- централизованный scheduler с bounded очередями;
- микробатчинг запросов на inference;
- compact per-camera state (ring buffers, без хранения сырых кадров);
- backpressure/drop policy;
- разделение ingest/inference/result контуров.

Что пока демо:

- ingest и inference эмулируются внутри процесса;
- для Rust↔Python control-plane добавлен probe JSON-RPC сервиса:
  - задайте `INFERENCE_RPC_ADDR=127.0.0.1:7788`;
  - при старте `runtime-core` выполнит health-check Python inference service.

Запуск:

```bash
cd runtime-core
cargo run
```

Логи покажут работу scheduler и сигналы backpressure.

Параллельно можно поднять Python inference service:

```bash
python run_inference_service.py --host 127.0.0.1 --port 7788
```

Чтобы принимать runtime-метаданные из текущего Python pipeline:

```bash
set RUNTIME_INGEST_ADDR=127.0.0.1:7878
cargo run
```

И в `config.json` выставить:

- `pipeline.runtime_core_addr = "127.0.0.1:7878"`
- `pipeline.runtime_core_timeout_sec = 0.05`

Чтобы runtime-core отдавал решения `should_infer` в Python pipeline:

```bash
set RUNTIME_CONTROL_ADDR=127.0.0.1:7879
cargo run
```

И в `config.json` выставить:

- `pipeline.runtime_control_addr = "127.0.0.1:7879"`
- `pipeline.runtime_control_timeout_sec = 0.01`

Control policy сейчас адаптивная:

- базовый интервал берется из `default_interval` (Python `detect_every_n_frames`);
- добавляется boost по EMA latency камеры;
- добавляется boost при перегрузе (pending/dropped в scheduler).
- для активных камер (detections/events) дается activity relief и выше `priority`.
- в ответе control-plane идут hints: `target_interval`, `priority`, `max_roi_count`.
- Python pipeline применяет эти hints в реальном infer path (не только телеметрия).

Для удобства эксплуатации эти значения автоматически выставляются через:

```bash
python run_stack.py --profile hybrid
python run_stack.py --profile external
```

Адреса и таймауты можно переопределять через env:

- `RUNTIME_INGEST_ADDR`
- `RUNTIME_CONTROL_ADDR`
- `INFERENCE_RPC_ADDR`
- `RUNTIME_CORE_TIMEOUT_SEC`
- `RUNTIME_CONTROL_TIMEOUT_SEC`
