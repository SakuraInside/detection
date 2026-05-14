# Integra native pipeline: видео (OpenCV) + CUDA kernels + отдельный alarm-сервис.
# Python-стек остаётся опционально для миграции; целевой рантайм — этот бинарник.
cmake_minimum_required(VERSION 3.20)
project(integra-native VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(INTEGRA_ENABLE_CUDA "Сборка CUDA kernels (.cu)" ON)
option(INTEGRA_WITH_ONNXRUNTIME "ONNX Runtime: реальный --engine onnx" OFF)
option(INTEGRA_ORT_CUDA "ONNX: AppendExecutionProvider_CUDA (нужен GPU-пакет ORT + CuDNN)" OFF)
option(INTEGRA_WITH_TENSORRT "TensorRT: сериализованный .engine (--engine tensorrt)" OFF)
set(INTEGRA_ONNXRUNTIME_ROOT "" CACHE PATH "Корень ONNX Runtime: include/, lib/")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT: include/ (NvInfer.h), lib/ (libnvinfer.so)")

if(INTEGRA_ORT_CUDA AND NOT INTEGRA_WITH_ONNXRUNTIME)
  message(FATAL_ERROR "INTEGRA_ORT_CUDA=ON требует INTEGRA_WITH_ONNXRUNTIME=ON")
endif()
if(INTEGRA_WITH_TENSORRT AND NOT INTEGRA_ENABLE_CUDA)
  message(FATAL_ERROR "INTEGRA_WITH_TENSORRT=ON требует INTEGRA_ENABLE_CUDA=ON")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn)
find_package(Threads REQUIRED)

if(INTEGRA_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA arch list")
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(integra_core STATIC
  src/video_source.cpp
  src/gpu_preprocess.cpp
  src/inference_stub.cpp
  src/shared_engine.cpp
  src/opencv_dnn_engine.cpp
  src/yolo_postprocess.cpp
  src/yolo_letterbox.cpp
  src/iou_tracker.cpp
  src/scene_analyzer.cpp
  src/alarm_sink.cpp
  src/frame_filter.cpp
  src/pipeline.cpp
)

target_include_directories(integra_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(integra_core PUBLIC ${OpenCV_LIBS})

if(INTEGRA_WITH_ONNXRUNTIME)
  find_path(INTEGRA_ONNX_INCLUDE NAMES onnxruntime_cxx_api.h
    HINTS "${INTEGRA_ONNXRUNTIME_ROOT}/include" "$ENV{ONNXRUNTIME_ROOT}/include"
    PATHS /usr/local/include /usr/include)
  find_library(INTEGRA_ONNX_LIB NAMES onnxruntime
    HINTS "${INTEGRA_ONNXRUNTIME_ROOT}/lib" "$ENV{ONNXRUNTIME_ROOT}/lib"
    PATHS /usr/local/lib /usr/lib)
  if(NOT INTEGRA_ONNX_INCLUDE OR NOT INTEGRA_ONNX_LIB)
    message(FATAL_ERROR "INTEGRA_WITH_ONNXRUNTIME=ON: не найден onnxruntime (include onnxruntime_cxx_api.h / libonnxruntime). Задайте INTEGRA_ONNXRUNTIME_ROOT.")
  endif()
  message(STATUS "ONNX Runtime include: ${INTEGRA_ONNX_INCLUDE}")
  message(STATUS "ONNX Runtime lib: ${INTEGRA_ONNX_LIB}")
  target_sources(integra_core PRIVATE src/onnx_engine.cpp)
  target_include_directories(integra_core PRIVATE "${INTEGRA_ONNX_INCLUDE}")
  target_link_libraries(integra_core PRIVATE "${INTEGRA_ONNX_LIB}")
  target_compile_definitions(integra_core PUBLIC INTEGRA_HAS_ONNXRUNTIME=1)
  if(INTEGRA_ORT_CUDA)
    message(STATUS "ONNX Runtime: CUDA execution provider (INTEGRA_ORT_CUDA)")
    target_compile_definitions(integra_core PRIVATE INTEGRA_ORT_CUDA=1)
  endif()
endif()

if(INTEGRA_WITH_TENSORRT)
  find_path(TENSORRT_INCLUDE NAMES NvInfer.h
    HINTS "${TENSORRT_ROOT}/include" "$ENV{TENSORRT_ROOT}/include"
    PATHS /usr/include/x86_64-linux-gnu /usr/local/include)
  find_library(TENSORRT_NVINFER NAMES nvinfer nvinfer_10
    HINTS "${TENSORRT_ROOT}/lib" "$ENV{TENSORRT_ROOT}/lib"
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
  find_library(TENSORRT_NVONNXPARSER NAMES nvonnxparser nvonnxparser_10
    HINTS "${TENSORRT_ROOT}/lib" "$ENV{TENSORRT_ROOT}/lib"
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
  if(NOT TENSORRT_INCLUDE OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR "INTEGRA_WITH_TENSORRT=ON: не найден TensorRT (NvInfer.h, libnvinfer). Задайте TENSORRT_ROOT.")
  endif()
  if(NOT TENSORRT_NVONNXPARSER)
    message(FATAL_ERROR "INTEGRA_WITH_TENSORRT=ON: не найден libnvonnxparser (нужен для integra_trt_bake). Проверьте TENSORRT_ROOT/lib.")
  endif()
  message(STATUS "TensorRT include: ${TENSORRT_INCLUDE}")
  message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
  message(STATUS "TensorRT nvonnxparser: ${TENSORRT_NVONNXPARSER}")
  target_sources(integra_core PRIVATE src/tensorrt_engine.cpp)
  target_include_directories(integra_core PRIVATE "${TENSORRT_INCLUDE}")
  target_link_libraries(integra_core PRIVATE "${TENSORRT_NVINFER}")
  target_compile_definitions(integra_core PUBLIC INTEGRA_HAS_TENSORRT=1)

  # Утилита: ONNX → serialized engine (off-line, перед запуском gateway).
  add_executable(integra_trt_bake src/trt_bake_main.cpp)
  target_include_directories(integra_trt_bake PRIVATE "${TENSORRT_INCLUDE}")
  target_link_libraries(integra_trt_bake PRIVATE
    "${TENSORRT_NVINFER}"
    "${TENSORRT_NVONNXPARSER}"
  )
  if(INTEGRA_ENABLE_CUDA)
    target_link_libraries(integra_trt_bake PRIVATE CUDA::cudart)
  endif()
endif()

if(INTEGRA_ENABLE_CUDA)
  target_sources(integra_core PRIVATE cuda/preprocess.cu)
  target_compile_definitions(integra_core PUBLIC INTEGRA_HAS_CUDA=1)
  set_property(TARGET integra_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET integra_core PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(integra_core PUBLIC CUDA::cudart)
else()
  target_compile_definitions(integra_core PUBLIC INTEGRA_HAS_CUDA=0)
endif()

add_executable(integra-pipeline src/main.cpp)
target_link_libraries(integra-pipeline PRIVATE integra_core)

add_executable(integra-analyticsd src/analyticsd_main.cpp)
target_link_libraries(integra-analyticsd PRIVATE integra_core Threads::Threads)

if(WIN32)
  target_link_libraries(integra-analyticsd PRIVATE ws2_32)
endif()

add_executable(integra-alarmd alarmd/main.cpp)
target_link_libraries(integra-alarmd PRIVATE Threads::Threads)
if(WIN32)
  target_link_libraries(integra-alarmd PRIVATE ws2_32)
endif()

# integra_ffi: SHARED library — C ABI для Rust/любого другого языка.
add_library(integra_ffi SHARED src/integra_ffi.cpp)
target_compile_definitions(integra_ffi PRIVATE INTEGRA_FFI_BUILD=1)
target_link_libraries(integra_ffi PRIVATE integra_core Threads::Threads)
if(WIN32)
  set_target_properties(integra_ffi PROPERTIES PREFIX "")
  target_link_libraries(integra_ffi PRIVATE ws2_32)
endif()
set_target_properties(integra_ffi PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS integra-pipeline integra-analyticsd integra-alarmd RUNTIME DESTINATION bin)
if(INTEGRA_WITH_TENSORRT)
  install(TARGETS integra_trt_bake RUNTIME DESTINATION bin)
endif()
install(TARGETS integra_ffi
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES include/integra/integra_ffi.h DESTINATION include/integra)
