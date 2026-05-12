#include "integra/inference_engine.hpp"

#include <iostream>
#include <memory>

#if INTEGRA_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace integra {

#if INTEGRA_HAS_ONNXRUNTIME
std::unique_ptr<IInferenceEngine> make_onnx_engine();
#endif
#if INTEGRA_HAS_TENSORRT
std::unique_ptr<IInferenceEngine> make_tensorrt_engine();
#endif
std::unique_ptr<IInferenceEngine> make_opencv_dnn_engine();

namespace {

class StubEngine final : public IInferenceEngine {
 public:
  bool init(const InferenceEngineConfig& /*cfg*/) override { return true; }

  bool infer(const InferenceInput& in, DetectionBatch& out) override {
    out.items.clear();
    out.inference_ms = 0.f;
#if INTEGRA_HAS_CUDA
    if (in.on_device && in.cuda_stream != nullptr) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(in.cuda_stream));
    }
#endif
    return true;
  }

  bool is_stub() const override { return true; }
};

}  // namespace

std::unique_ptr<IInferenceEngine> make_inference_engine(const std::string& kind) {
  if (kind == "onnx") {
#if INTEGRA_HAS_ONNXRUNTIME
    return make_onnx_engine();
#else
    std::cerr << "integra: пересоберите с -DINTEGRA_WITH_ONNXRUNTIME=ON (и ONNX Runtime в "
                 "INTEGRA_ONNXRUNTIME_ROOT); пока используется stub.\n";
    return std::make_unique<StubEngine>();
#endif
  }
  if (kind == "tensorrt") {
#if INTEGRA_HAS_TENSORRT
    return make_tensorrt_engine();
#else
    std::cerr << "integra: пересоберите с -DINTEGRA_WITH_TENSORRT=ON (TensorRT SDK, TENSORRT_ROOT) "
                 "и укажите --model file.engine\n";
    return std::make_unique<StubEngine>();
#endif
  }
  if (kind == "opencv") {
    return make_opencv_dnn_engine();
  }
  if (kind.empty() || kind == "stub") {
    return std::make_unique<StubEngine>();
  }
  return nullptr;
}

}  // namespace integra
