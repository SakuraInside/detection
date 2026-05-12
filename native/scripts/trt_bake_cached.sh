#!/usr/bin/env bash
# Кешированная сборка .engine из ONNX для Integra (Linux + TensorRT).
#
# Имя артефакта: <stem>__sha<16>__cc<major>_<minor>__fp16.engine
# в каталоге <dir-of-onnx>/.integra_trt_cache/
#
# Требуется: integra_trt_bake в PATH (из build/) или задайте INTEGRA_TRT_BAKE.
#
# Usage:
#   ./native/scripts/trt_bake_cached.sh /path/to/model.onnx [workspace_mb]
#
set -euo pipefail

ONNX_PATH="${1:?usage: $0 /path/to/model.onnx [workspace_mb]}"
WS_MB="${2:-4096}"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "error: ONNX not found: ${ONNX_PATH}" >&2
  exit 1
fi

BAKER="${INTEGRA_TRT_BAKE:-integra_trt_bake}"
if [[ -x "${BAKER}" ]]; then
  :
elif command -v "${BAKER}" >/dev/null 2>&1; then
  BAKER="$(command -v "${BAKER}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  NATIVE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  CAND="${NATIVE_DIR}/build/integra_trt_bake"
  if [[ -x "${CAND}" ]]; then
    BAKER="${CAND}"
  else
    echo "error: integra_trt_bake not found. Build with INTEGRA_WITH_TENSORRT=ON or set INTEGRA_TRT_BAKE" >&2
    exit 1
  fi
fi

if command -v sha256sum >/dev/null 2>&1; then
  HASH="$(sha256sum "${ONNX_PATH}" | awk '{print substr($1,1,16)}')"
elif command -v shasum >/dev/null 2>&1; then
  HASH="$(shasum -a 256 "${ONNX_PATH}" | awk '{print substr($1,1,16)}')"
else
  echo "error: need sha256sum or shasum" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  CC_TAG="${CC_RAW//./_}"
else
  CC_TAG="unknown"
fi

BASE="$(basename "${ONNX_PATH}" .onnx)"
CACHE_DIR="$(dirname "${ONNX_PATH}")/.integra_trt_cache"
mkdir -p "${CACHE_DIR}"

OUT="${CACHE_DIR}/${BASE}__sha${HASH}__cc${CC_TAG}__fp16.engine"

if [[ -f "${OUT}" ]]; then
  echo "Reusing cached engine: ${OUT}" >&2
  echo "${OUT}"
  exit 0
fi

TMP="${OUT}.tmp.$$"
"${BAKER}" --onnx "${ONNX_PATH}" --out "${TMP}" --fp16 --workspace-mb "${WS_MB}"
mv -f "${TMP}" "${OUT}"
echo "Wrote ${OUT}" >&2
echo "${OUT}"
