# Service Split Runbook

## Topology

- `services/backend`: FastAPI backend API/WS/video stream, native analytics orchestration.
- `services/frontend`: standalone UI service that consumes backend over HTTP.
- Native analytics path is primary (`native_analytics.enabled=true`, `fallback_to_python=false`).
- Real-native guardrail: `native_analytics.require_real_engine=true` must stay enabled in production.
  Backend will reject `stub` daemon handshake and fail fast.

## Start Services

### 1) Native analytics daemon

Run `integra-analyticsd` on `127.0.0.1:9909` before backend start.
For production-grade quality/memory profile, daemon must be built with a real engine:
- `-DINTEGRA_WITH_TENSORRT=ON` and `--engine tensorrt --model <file.engine>`, or
- `-DINTEGRA_WITH_ONNXRUNTIME=ON` and `--engine onnx --model <file.onnx>`.

### 2) Backend

```bash
python services/backend/run_backend.py --host 127.0.0.1 --port 8000
```

Health checks:

- `GET /health`
- `GET /health/sla?stream_id=main`

### 3) Frontend

```bash
set INTEGRA_BACKEND_URL=http://127.0.0.1:8000
python -m services.frontend.server --host 127.0.0.1 --port 8080
```

Frontend health:

- `GET /health`

## Low-RAM FHD Profile

`config.json` enforces:

- `pipeline.decode_queue=1`
- `pipeline.result_queue=1`
- `pipeline.frame_pool_size=1`
- capped preview/snapshot resolution and jpeg quality
- memory thresholds: warn at 768MiB, fail at 960MiB

## Stability and Parity Validation

Run 10-15 minute probe:

```bash
python stability_probe.py --backend http://127.0.0.1:8000 --stream-id main --minutes 15
```

Parity check:

```bash
python parity_native_vs_python.py --python-events logs/python_events.json --native-events logs/native_events.json
```

