#include "integra/inference_engine.hpp"
#include "integra/yolo_postprocess.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if INTEGRA_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace integra {

namespace {

class OnnxEngine final : public IInferenceEngine {
 public:
  bool init(const InferenceEngineConfig& cfg) override {
    cfg_ = cfg;
    if (cfg.model_path.empty()) {
      std::cerr << "integra: --engine onnx требует непустой --model (путь к .onnx)\n";
      return false;
    }

    try {
      Ort::SessionOptions so;
      unsigned hc = std::thread::hardware_concurrency();
      if (hc == 0) {
        hc = 1;
      }
      so.SetIntraOpNumThreads(static_cast<int>(std::min<unsigned>(hc, 4u)));
#if defined(INTEGRA_ORT_CUDA)
      {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        so.AppendExecutionProvider_CUDA(cuda_opts);
      }
#endif
      session_ = std::make_unique<Ort::Session>(env_, cfg.model_path.c_str(), so);
    } catch (const Ort::Exception& e) {
      std::cerr << "integra: ONNX Runtime: " << e.what() << "\n";
      return false;
    }

    try {
      Ort::AllocatorWithDefaultOptions alloc;
      auto in_name = session_->GetInputNameAllocated(0, alloc);
      auto out_name = session_->GetOutputNameAllocated(0, alloc);
      input_name_str_.assign(in_name.get());
      output_name_str_.assign(out_name.get());

      Ort::TypeInfo in_info = session_->GetInputTypeInfo(0);
      auto ts = in_info.GetTensorTypeAndShapeInfo();
      input_template_shape_ = ts.GetShape();
      if (input_template_shape_.size() != 4) {
        std::cerr << "integra: ожидается вход 4D NCHW [1,3,H,W], получено rank "
                  << input_template_shape_.size() << "\n";
        return false;
      }
    } catch (const Ort::Exception& e) {
      std::cerr << "integra: ONNX метаданные: " << e.what() << "\n";
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

    const std::size_t n_elem = static_cast<std::size_t>(3) * static_cast<std::size_t>(H) *
                               static_cast<std::size_t>(W);
    if (host_nchw_.size() < n_elem) {
      host_nchw_.resize(n_elem);
    }

#if INTEGRA_HAS_CUDA
    if (in.on_device) {
      cudaStream_t stream = static_cast<cudaStream_t>(in.cuda_stream);
      cudaError_t e;
      if (stream != nullptr) {
        e = cudaMemcpyAsync(host_nchw_.data(), in.nchw, n_elem * sizeof(float), cudaMemcpyDeviceToHost,
                            stream);
        if (e == cudaSuccess) {
          e = cudaStreamSynchronize(stream);
        }
      } else {
        e = cudaMemcpy(host_nchw_.data(), in.nchw, n_elem * sizeof(float), cudaMemcpyDeviceToHost);
      }
      if (e != cudaSuccess) {
        std::cerr << "integra: cudaMemcpy (ORT вход) failed\n";
        return false;
      }
    } else
#endif
    {
      std::memcpy(host_nchw_.data(), in.nchw, n_elem * sizeof(float));
    }

    std::vector<int64_t> in_shape = input_template_shape_;
    for (std::size_t i = 0; i < in_shape.size(); ++i) {
      if (in_shape[i] < 0) {
        if (i == 0) {
          in_shape[i] = 1;
        } else if (i == 1) {
          in_shape[i] = 3;
        } else if (i == 2) {
          in_shape[i] = H;
        } else if (i == 3) {
          in_shape[i] = W;
        }
      }
    }
    if (in_shape.size() >= 4) {
      if (in_shape[2] <= 0) {
        in_shape[2] = H;
      }
      if (in_shape[3] <= 0) {
        in_shape[3] = W;
      }
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, host_nchw_.data(), n_elem, in_shape.data(), in_shape.size());

    const char* in_names[] = {input_name_str_.c_str()};
    const char* out_names[] = {output_name_str_.c_str()};

    auto t0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs;
    try {
      outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);
    } catch (const Ort::Exception& e) {
      std::cerr << "integra: ONNX Run: " << e.what() << "\n";
      return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    out.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (outputs.empty()) {
      return true;
    }

    float* raw = outputs[0].GetTensorMutableData<float>();
    if (raw == nullptr) {
      return false;
    }

    Ort::TensorTypeAndShapeInfo oinfo = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> oshape = oinfo.GetShape();
    if (oshape.size() == 3 && oshape[0] == 1) {
      const int64_t d1 = oshape[1];
      const int64_t d2 = oshape[2];
      const bool chw = yolo_output_channel_first(d1, d2);
      PostprocessParams pp = cfg_.postprocess;
      std::vector<Detection> decoded;
      decode_yolov8_flat(raw, static_cast<int>(d1), static_cast<int>(d2), chw, pp,
                         static_cast<float>(W), static_cast<float>(H), transpose_scratch_, decoded);
      nms_greedy_xyxy(decoded, pp.nms_iou_threshold);
      out.items = std::move(decoded);
      return true;
    }

    std::cerr << "integra: неподдерживаемая размерность выхода ONNX (rank=" << oshape.size()
              << ")\n";
    return false;
  }

 private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "integra"};
  std::unique_ptr<Ort::Session> session_;
  InferenceEngineConfig cfg_;
  std::string input_name_str_;
  std::string output_name_str_;
  std::vector<int64_t> input_template_shape_;
  std::vector<float> host_nchw_;
  std::vector<float> transpose_scratch_;
};

}  // namespace

std::unique_ptr<IInferenceEngine> make_onnx_engine() {
  return std::make_unique<OnnxEngine>();
}

}  // namespace integra
