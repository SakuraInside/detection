#!/usr/bin/env bash
# Один запуск веб-интерфейса: опционально integra-alarmd + uvicorn (run.py).
# Использование:
#   ./start_ui.sh
#   ./start_ui.sh --host 0.0.0.0 --port 8080
# Доп. библиотеки (TensorRT / ONNX для нативных бинарников): см. .env.native.example

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

if [[ -f "${ROOT}/.env.native" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ROOT}/.env.native"
  set +a
fi

_append_ld() {
  local d="$1"
  [[ -d "$d" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$d:"*) ;;
    *) export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

# TensorRT: типичная распаковка в Downloads
for cand in "${HOME}/Downloads"/TensorRT-*/*/lib \
            "${HOME}/Downloads"/TensorRT-*/lib \
            "${TENSORRT_ROOT:-}"/lib; do
  [[ -e "$cand" ]] && _append_ld "$cand" && break
done

# ONNX Runtime C++ (если распакован в Downloads)
for cand in "${HOME}/Downloads"/onnxruntime-linux-*/lib \
            "${INTEGRA_ONNXRUNTIME_ROOT:-}"/lib; do
  [[ -e "$cand" ]] && _append_ld "$cand" && break
done

ALARMD_PID=""
cleanup() {
  if [[ -n "${ALARMD_PID}" ]] && kill -0 "${ALARMD_PID}" 2>/dev/null; then
    kill "${ALARMD_PID}" 2>/dev/null || true
    wait "${ALARMD_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Фоновый приём JSON-тревог (нужен только если пользуетесь native integra-pipeline → alarm-sink)
if [[ "${INTEGRA_START_ALARMD:-1}" != "0" ]] && [[ -x "${ROOT}/native/build/integra-alarmd" ]]; then
  mkdir -p "${ROOT}/logs"
  "${ROOT}/native/build/integra-alarmd" \
    --listen "${INTEGRA_ALARMD_LISTEN:-127.0.0.1:9090}" \
    --log "${ROOT}/logs/native_alerts.jsonl" &
  ALARMD_PID=$!
  echo "[start_ui] integra-alarmd → ${INTEGRA_ALARMD_LISTEN:-127.0.0.1:9090} (pid ${ALARMD_PID})"
fi

echo "[start_ui] запуск веб-приложения (python run.py)…"
python "${ROOT}/run.py" "$@"
rc=$?
cleanup
trap - EXIT INT TERM
exit "${rc}"
