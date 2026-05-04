#pragma once

#include "integra/types.hpp"
#include "integra/yolo_postprocess.hpp"

#include <memory>
#include <string>

namespace integra {

/// Вход в движок: NCHW float [1,3,H,W] (как после `integra_cuda_bgr_to_nchw_float`).
struct InferenceInput {
  const float* nchw = nullptr;  // host или device — см. on_device
  int width = 0;
  int height = 0;
  bool on_device = false;       // true: CUDA pointer (TensorRT); false: CPU
  void* cuda_stream = nullptr;  // cudaStream_t при on_device; иначе nullptr
};

struct InferenceEngineConfig {
  std::string model_path;
  int input_size = 640;
  PostprocessParams postprocess;
};

/// Плагин к TensorRT / ONNX Runtime / OpenCV DNN. Реализация по умолчанию — stub.
class IInferenceEngine {
 public:
  virtual ~IInferenceEngine() = default;

  /// model_path: .engine / .onnx / пусто для stub. input_size: квадрат 640, 960, …
  virtual bool init(const InferenceEngineConfig& cfg) = 0;
  virtual bool infer(const InferenceInput& in, DetectionBatch& out) = 0;
};

/// Фабрика: onnx / tensorrt при соответствующих CMake-опциях (см. native/CMakeLists.txt).
std::unique_ptr<IInferenceEngine> make_inference_engine(const std::string& kind);

}  // namespace integra
