#!/usr/bin/env bash
# Сборка integra-pipeline / integra-alarmd с CUDA и (по возможности) TensorRT / ONNX Runtime.
# Запуск из любого каталога:  bash native/build_integra.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# TensorRT: задайте export TENSORRT_ROOT=... при нестандартном пути.
# Иначе ищем: ~/var/TensorRT-* (часто распаковывают в ~/var), затем дефолтный путь из Downloads.
if [[ -z "${TENSORRT_ROOT:-}" ]]; then
  _trt_auto=""
  shopt -s nullglob
  for d in "${HOME}/var"/TensorRT-* "${HOME}/var"; do
    if [[ -f "${d}/include/NvInfer.h" ]]; then
      _trt_auto="$d"
      break
    fi
  done
  shopt -u nullglob
  if [[ -n "${_trt_auto}" ]]; then
    export TENSORRT_ROOT="${_trt_auto}"
    echo "TensorRT: TENSORRT_ROOT не задан — используем ${_trt_auto}"
  else
    export TENSORRT_ROOT="${HOME}/Downloads/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-13.2/TensorRT-10.16.1.11"
  fi
fi
unset _trt_auto 2>/dev/null || true

tensorrt_sdk_ok() {
  # Тарбол NVIDIA: TENSORRT_ROOT/include, TENSORRT_ROOT/lib
  if [[ -f "${TENSORRT_ROOT}/include/NvInfer.h" ]]; then
    local shopt_save
    shopt_save=$(shopt -p nullglob)
    shopt -s nullglob
    local libs=("${TENSORRT_ROOT}/lib"/libnvinfer.so*)
    eval "$shopt_save"
    if ((${#libs[@]} > 0)); then
      return 0
    fi
  fi
  # Пакеты Ubuntu/Debian (nv-tensorrt-local-repo → apt install libnvinfer-dev …)
  if [[ -f /usr/include/x86_64-linux-gnu/NvInfer.h ]]; then
    shopt -s nullglob
    local libs=(/usr/lib/x86_64-linux-gnu/libnvinfer.so*)
    shopt -u nullglob
    if ((${#libs[@]} > 0)); then
      export TENSORRT_ROOT=/usr
      return 0
    fi
  fi
  return 1
}

# INTEGRA_REQUIRE_TENSORRT=1 — упасть, если SDK не найден (вместо сборки без TRT).
if tensorrt_sdk_ok; then
  WITH_TENSORRT=ON
  echo "TensorRT SDK: ${TENSORRT_ROOT}"
else
  if [[ "${INTEGRA_REQUIRE_TENSORRT:-0}" == "1" ]]; then
    echo "Ошибка: TensorRT SDK не найден." >&2
    echo "  Тарбол: export TENSORRT_ROOT=.../TensorRT-... (include/NvInfer.h, lib/libnvinfer.so*)" >&2
    echo "  Или apt: sudo apt install libnvinfer-dev libnvonnxparser-dev (после apt update)." >&2
    shopt -s nullglob
    _trt_repo=(/var/nv-tensorrt-local-repo-*)
    shopt -u nullglob
    if ((${#_trt_repo[@]} > 0)); then
      echo "  Репозиторий TensorRT в ${_trt_repo[0]} — установите из него dev-пакеты (см. выше)." >&2
    fi
    exit 1
  fi
  WITH_TENSORRT=OFF
  echo "Внимание: TensorRT SDK не найден (TENSORRT_ROOT=${TENSORRT_ROOT}) — сборка без INTEGRA_WITH_TENSORRT."
  echo "  Для --engine tensorrt: apt (libnvinfer-dev) или тарбол + export TENSORRT_ROOT=..."
  echo "  Либо в config.json: engine onnx + YOLO11 .onnx (нужен ONNX Runtime SDK и пересборка), см. native/README.md."
  shopt -s nullglob
  _trt_repo=(/var/nv-tensorrt-local-repo-*)
  shopt -u nullglob
  if ((${#_trt_repo[@]} > 0)); then
    echo "  Обнаружен каталог ${_trt_repo[0]} — это локальный репозиторий apt (кэш .deb), не корень SDK."
    echo "  Дальше: sudo apt update && sudo apt install libnvinfer-dev libnvonnxparser-dev"
    echo "  (точные имена пакетов — в документации NVIDIA / apt search nvinfer после подключения репо)."
  fi
fi

# ONNX Runtime C++ SDK — только из архива GitHub (pip-колесо заголовков не даёт):
# https://github.com/microsoft/onnxruntime/releases
# Ищем распакованный onnxruntime-linux-* в Downloads.
INTEGRA_ONNXRUNTIME_ROOT="${INTEGRA_ONNXRUNTIME_ROOT:-}"
if [[ -z "${INTEGRA_ONNXRUNTIME_ROOT}" ]]; then
  shopt -s nullglob
  for cand in "${HOME}/Downloads"/onnxruntime-linux-* "${HOME}/Desktop"/onnxruntime-linux-* "${HOME}/var"/onnxruntime-linux-*; do
    if [[ -f "${cand}/include/onnxruntime_cxx_api.h" ]]; then
      INTEGRA_ONNXRUNTIME_ROOT="${cand}"
      echo "Найден ONNX Runtime SDK: ${INTEGRA_ONNXRUNTIME_ROOT}"
      break
    fi
  done
  shopt -u nullglob
fi

CMAKE_OPTS=(
  -DINTEGRA_ENABLE_CUDA=ON
  "-DINTEGRA_WITH_TENSORRT=${WITH_TENSORRT}"
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
  echo "  YOLO11 .onnx с OpenCV DNN не загружается — для YOLO11 нужен ONNX Runtime: архив с"
  echo "  https://github.com/microsoft/onnxruntime/releases → распаковать в ~/Downloads, ~/Desktop или ~/var,"
  echo "  затем: export INTEGRA_ONNXRUNTIME_ROOT=/полный/путь && bash native/build_integra.sh"
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" "${CMAKE_OPTS[@]}" "$@"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo
echo "Готово:"
echo "  ${BUILD_DIR}/integra-pipeline"
echo "  ${BUILD_DIR}/integra-analyticsd"
echo "  ${BUILD_DIR}/integra-alarmd"
if [[ "${WITH_TENSORRT}" == "ON" ]]; then
  echo "  ${BUILD_DIR}/integra_trt_bake   # ONNX → .engine (FP16)"
fi
echo "  ${BUILD_DIR}/integra_ffi"
echo
if [[ "${WITH_TENSORRT}" == "ON" ]]; then
  echo "Если при запуске ругается на libnvinfer / libonnxruntime:"
  if [[ -n "${TENSORRT_ROOT}" && "${TENSORRT_ROOT}" != "/usr" && -d "${TENSORRT_ROOT}/lib" ]]; then
    echo "  export LD_LIBRARY_PATH=\"${TENSORRT_ROOT}/lib:\${LD_LIBRARY_PATH}\""
  fi
  shopt -s nullglob
  _trt_libs=(/usr/lib/x86_64-linux-gnu/libnvinfer.so*)
  shopt -u nullglob
  if ((${#_trt_libs[@]} > 0)); then
    echo "  export LD_LIBRARY_PATH=\"/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}\"   # типично для apt"
  fi
fi
if [[ -n "${INTEGRA_ONNXRUNTIME_ROOT}" ]]; then
  echo "  export LD_LIBRARY_PATH=\"${INTEGRA_ONNXRUNTIME_ROOT}/lib:\${LD_LIBRARY_PATH}\""
fi
