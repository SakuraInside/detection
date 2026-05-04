#include "integra/inference_engine.hpp"
#include "integra/yolo_postprocess.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace integra {

namespace {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "tensorrt: " << msg << "\n";
    }
  }
};

static size_t volume_dims(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    const int di = d.d[i];
    if (di > 0) {
      v *= static_cast<size_t>(di);
    }
  }
  return v;
}

class TensorRTEngine final : public IInferenceEngine {
 public:
  TensorRTEngine() = default;

  ~TensorRTEngine() override {
    if (d_output_) {
      cudaFree(d_output_);
    }
    if (d_input_owned_) {
      cudaFree(d_input_owned_);
    }
#if NV_TENSORRT_MAJOR >= 10
    delete ctx_;
    delete engine_;
    delete runtime_;
#else
    if (ctx_) {
      ctx_->destroy();
    }
    if (engine_) {
      engine_->destroy();
    }
    if (runtime_) {
      runtime_->destroy();
    }
#endif
  }

  TensorRTEngine(const TensorRTEngine&) = delete;
  TensorRTEngine& operator=(const TensorRTEngine&) = delete;

  bool init(const InferenceEngineConfig& cfg) override {
    cfg_ = cfg;
    if (cfg.model_path.empty()) {
      std::cerr << "integra: --engine tensorrt требует непустой --model (путь к .engine)\n";
      return false;
    }

    std::ifstream f(cfg.model_path, std::ios::binary);
    if (!f) {
      std::cerr << "integra: не удалось открыть .engine: " << cfg.model_path << "\n";
      return false;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty()) {
      std::cerr << "integra: пустой файл движка\n";
      return false;
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
      std::cerr << "integra: createInferRuntime failed\n";
      return false;
    }
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) {
      std::cerr << "integra: deserializeCudaEngine failed\n";
      return false;
    }

    ctx_ = engine_->createExecutionContext();
    if (!ctx_) {
      std::cerr << "integra: createExecutionContext failed\n";
      return false;
    }

    const int32_t nio = engine_->getNbIOTensors();
    if (nio < 2) {
      std::cerr << "integra: ожидается хотя бы вход и выход (I/O tensors >= 2)\n";
      return false;
    }

    for (int32_t i = 0; i < nio; ++i) {
      const char* nm = engine_->getIOTensorName(i);
      const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(nm);
      if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
        input_name_ = nm;
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_name_.empty()) {
        output_name_ = nm;
      }
    }

    if (input_name_.empty() || output_name_.empty()) {
      std::cerr << "integra: не найдены имена входного/выходного тензора\n";
      return false;
    }

    const nvinfer1::DataType itype = engine_->getTensorDataType(input_name_.c_str());
    const nvinfer1::DataType otype = engine_->getTensorDataType(output_name_.c_str());
    if (itype != nvinfer1::DataType::kFLOAT || otype != nvinfer1::DataType::kFLOAT) {
      std::cerr << "integra: пока поддерживается только FP32 I/O\n";
      return false;
    }

    return true;
  }

  bool infer(const InferenceInput& in, DetectionBatch& out) override {
    out.items.clear();
    out.inference_ms = 0.f;

    const int H = in.height;
    const int W = in.width;
    if (H <= 0 || W <= 0 || in.nchw == nullptr) {
      return false;
    }

    const std::size_t n_in = static_cast<std::size_t>(3) * static_cast<std::size_t>(H) *
                             static_cast<std::size_t>(W);

    nvinfer1::Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = 1;
    in_dims.d[1] = 3;
    in_dims.d[2] = H;
    in_dims.d[3] = W;

    if (!ctx_->setInputShape(input_name_.c_str(), in_dims)) {
      std::cerr << "integra: TensorRT setInputShape failed (проверьте размер кадра и .engine)\n";
      return false;
    }

    const float* d_in = nullptr;
    if (in.on_device) {
      d_in = in.nchw;
    } else {
      if (d_input_owned_bytes_ < n_in * sizeof(float)) {
        if (d_input_owned_) {
          cudaFree(d_input_owned_);
        }
        if (cudaMalloc(&d_input_owned_, n_in * sizeof(float)) != cudaSuccess) {
          std::cerr << "integra: cudaMalloc input failed\n";
          return false;
        }
        d_input_owned_bytes_ = n_in * sizeof(float);
      }
      if (cudaMemcpy(d_input_owned_, in.nchw, n_in * sizeof(float), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        std::cerr << "integra: cudaMemcpy H2D input failed\n";
        return false;
      }
      d_in = d_input_owned_;
    }

    ctx_->setTensorAddress(input_name_.c_str(),
                           const_cast<void*>(static_cast<const void*>(d_in)));

    const nvinfer1::Dims od = ctx_->getTensorShape(output_name_.c_str());
    for (int i = 0; i < od.nbDims; ++i) {
      if (od.d[i] <= 0) {
        std::cerr << "integra: выходной тензор имеет неполную размерность (динамика?) после "
                     "setInputShape\n";
        return false;
      }
    }
    const size_t out_elems = volume_dims(od);
    const size_t need_o = out_elems * sizeof(float);
    if (d_output_cap_bytes_ < need_o) {
      if (d_output_) {
        cudaFree(d_output_);
      }
      if (cudaMalloc(&d_output_, need_o) != cudaSuccess) {
        std::cerr << "integra: cudaMalloc output failed\n";
        return false;
      }
      d_output_cap_bytes_ = need_o;
    }

    ctx_->setTensorAddress(output_name_.c_str(), d_output_);

    cudaStream_t stream{};
    if (in.cuda_stream != nullptr) {
      stream = static_cast<cudaStream_t>(in.cuda_stream);
    }

    auto t0 = std::chrono::steady_clock::now();
    const bool ok = ctx_->enqueueV3(stream);
    if (!ok) {
      std::cerr << "integra: TensorRT enqueueV3 failed\n";
      return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << "integra: cudaStreamSynchronize failed\n";
      return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    out.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    host_decode_.resize(out_elems);
    if (cudaMemcpy(host_decode_.data(), d_output_, need_o, cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::cerr << "integra: cudaMemcpy D2H output failed\n";
      return false;
    }

    if (od.nbDims != 3 || od.d[0] != 1) {
      std::cerr << "integra: ожидается выход rank 3 с batch 1\n";
      return false;
    }

    const std::int64_t d1 = od.d[1];
    const std::int64_t d2 = od.d[2];
    const bool chw = yolo_output_channel_first(d1, d2);
    PostprocessParams pp = cfg_.postprocess;
    std::vector<Detection> decoded;
    decode_yolov8_flat(host_decode_.data(), static_cast<int>(d1), static_cast<int>(d2), chw, pp,
                       static_cast<float>(W), static_cast<float>(H), transpose_scratch_, decoded);
    nms_greedy_xyxy(decoded, pp.nms_iou_threshold);
    out.items = std::move(decoded);
    return true;
  }

 private:
  TrtLogger logger_{};
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* ctx_ = nullptr;

  std::string input_name_;
  std::string output_name_;

  float* d_output_ = nullptr;
  size_t d_output_cap_bytes_ = 0;

  float* d_input_owned_ = nullptr;
  size_t d_input_owned_bytes_ = 0;

  std::vector<float> host_decode_;
  std::vector<float> transpose_scratch_;
  InferenceEngineConfig cfg_;
};

}  // namespace

std::unique_ptr<IInferenceEngine> make_tensorrt_engine() {
  return std::make_unique<TensorRTEngine>();
}

}  // namespace integra
