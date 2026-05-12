#!/usr/bin/env bash
# Сборка integra-pipeline / integra-alarmd с CUDA и (по возможности) TensorRT / ONNX Runtime.
# Запуск из любого каталога:  bash native/build_integra.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# TensorRT 10.x (путь после распаковки архива с developer.nvidia.com)
: "${TENSORRT_ROOT:=${HOME}/Downloads/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-13.2/TensorRT-10.16.1.11}"

# ONNX Runtime C++ SDK — только из архива GitHub (pip-колесо заголовков не даёт):
# https://github.com/microsoft/onnxruntime/releases
# Ищем распакованный onnxruntime-linux-* в Downloads.
INTEGRA_ONNXRUNTIME_ROOT="${INTEGRA_ONNXRUNTIME_ROOT:-}"
if [[ -z "${INTEGRA_ONNXRUNTIME_ROOT}" ]]; then
  for cand in "${HOME}/Downloads"/onnxruntime-linux-*; do
    if [[ -f "${cand}/include/onnxruntime_cxx_api.h" ]]; then
      INTEGRA_ONNXRUNTIME_ROOT="${cand}"
      echo "Найден ONNX Runtime SDK: ${INTEGRA_ONNXRUNTIME_ROOT}"
      break
    fi
  done
fi

CMAKE_OPTS=(
  -DINTEGRA_ENABLE_CUDA=ON
  -DINTEGRA_WITH_TENSORRT=ON
  "-DTENSORRT_ROOT=${TENSORRT_ROOT}"
)

if [[ -n "${INTEGRA_ONNXRUNTIME_ROOT}" && -f "${INTEGRA_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" ]]; then
  CMAKE_OPTS+=(
    -DINTEGRA_WITH_ONNXRUNTIME=ON
    "-DINTEGRA_ONNXRUNTIME_ROOT=${INTEGRA_ONNXRUNTIME_ROOT}"
  )
  # Раскомментируйте, если нужен CUDA Execution Provider внутри ORT (нужен GPU-пакет ORT):
  # CMAKE_OPTS+=(-DINTEGRA_ORT_CUDA=ON)
else
  echo "Внимание: C++ ONNX Runtime SDK не найден — сборка без --engine onnx (останется stub)."
  echo "Скачайте «onnxruntime-linux-x64-gpu» или «onnxruntime-linux-x64» с GitHub Releases,"
  echo "распакуйте в ~/Downloads и при необходимости задайте:"
  echo "  export INTEGRA_ONNXRUNTIME_ROOT=/полный/путь/к/onnxruntime-linux-..."
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" "${CMAKE_OPTS[@]}" "$@"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo
echo "Готово:"
echo "  ${BUILD_DIR}/integra-pipeline"
echo "  ${BUILD_DIR}/integra-analyticsd"
echo "  ${BUILD_DIR}/integra-alarmd"
echo "  ${BUILD_DIR}/integra_trt_bake   # ONNX → .engine (FP16), INTEGRA_WITH_TENSORRT=ON"
echo "  ${BUILD_DIR}/integra_ffi"
echo
echo "Если при запуске ругается на libnvinfer / libonnxruntime:"
echo "  export LD_LIBRARY_PATH=\"${TENSORRT_ROOT}/lib:\${LD_LIBRARY_PATH}\""
if [[ -n "${INTEGRA_ONNXRUNTIME_ROOT}" ]]; then
  echo "  export LD_LIBRARY_PATH=\"${INTEGRA_ONNXRUNTIME_ROOT}/lib:\${LD_LIBRARY_PATH}\""
fi
