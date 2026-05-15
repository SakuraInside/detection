#!/usr/bin/env bash
# Экспорт Ultralytics YOLO11s (или другого *.pt) → ONNX → TensorRT FP16 .engine для integra_trt_bake.
#
# Требуется:
#   • собранный native/build/integra_trt_bake (INTEGRA_WITH_TENSORRT=ON), см. native/build_integra.sh
#   • venv + pip (см. tools/requirements-trt-export.txt) — системный pip на Debian часто заблокирован (PEP 668)
#   • LD_LIBRARY_PATH с TensorRT lib (как в native/README.md)
#
# Использование (из корня репозитория):
#   ./native/scripts/export_yolo_trt_engine.sh [путь/к/weights.pt | yolo11s.pt] [imgsz] [workspace_mb]
#
# Если файла нет, но имя похоже на чекпойнт Ultralytics (например yolo11s.pt), библиотека может скачать веса.
#
# TensorRT: integra_trt_bake ищет libnvinfer_builder_resource_sm*.so — задайте TENSORRT_ROOT или
# export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:..."
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -n "${TENSORRT_ROOT:-}" ]]; then
  export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${TENSORRT_ROOT}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

WEIGHTS="${1:-models/yolo11s.pt}"
IMGSZ="${2:-640}"
WS_MB="${3:-4096}"

if [[ -n "${INTEGRA_EXPORT_PYTHON:-}" ]]; then
  PYTHON="${INTEGRA_EXPORT_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "error: Python not found: ${PYTHON}" >&2
  exit 1
fi

if ! "${PYTHON}" -c "import ultralytics" 2>/dev/null; then
  echo "error: пакет ultralytics не установлен для ${PYTHON}" >&2
  echo "  PEP 668: создайте venv и поставьте зависимости:" >&2
  echo "    python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-trt-export.txt" >&2
  exit 1
fi

if [[ -f "${WEIGHTS}" ]]; then
  echo "[export] веса: ${WEIGHTS}" >&2
else
  echo "[export] файл не найден (${WEIGHTS}) — пробуем загрузку/разрешение имени через Ultralytics…" >&2
fi

echo "[export] Ultralytics → ONNX (${WEIGHTS}, imgsz=${IMGSZ}), python=${PYTHON}" >&2
ONNX_PATH="$(WEIGHTS="${WEIGHTS}" IMGSZ="${IMGSZ}" "${PYTHON}" - <<'PY'
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from ultralytics import YOLO

raw = os.environ["WEIGHTS"]
sz = int(os.environ["IMGSZ"])
p = Path(raw)
key = str(p.resolve()) if p.exists() else raw

# Ultralytics печатает сводку в stdout — иначе bash $(...) забирает весь лог вместо пути к .onnx.
buf = io.StringIO()
with redirect_stdout(buf):
    m = YOLO(key)
    out = m.export(format="onnx", imgsz=sz, simplify=True, opset=12, verbose=False)
onnx_path = Path(out).resolve()
sys.stderr.write(buf.getvalue())
print(str(onnx_path))
PY
)"
ONNX_PATH="$(echo -n "${ONNX_PATH}" | tr -d '\r' | tail -n 1)"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "error: ONNX export failed: ${ONNX_PATH}" >&2
  exit 1
fi

echo "[export] ONNX: ${ONNX_PATH}" >&2
echo "[export] TensorRT bake (workspace ${WS_MB} MB)..." >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_CACHE="$(bash "${SCRIPT_DIR}/trt_bake_cached.sh" "${ONNX_PATH}" "${WS_MB}")"
ENGINE_CACHE="$(echo -n "${ENGINE_CACHE}" | tr -d '\r' | tail -n 1)"

STEM="$(basename "${ONNX_PATH}" .onnx)"
OUT_ENGINE="${ROOT}/models/${STEM}_fp16.engine"
mkdir -p "${ROOT}/models"
cp -f "${ENGINE_CACHE}" "${OUT_ENGINE}"
echo "[export] Готово: ${OUT_ENGINE}" >&2
echo "${OUT_ENGINE}"
