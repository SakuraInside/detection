#include "integra/inference_engine.hpp"
#include "integra/yolo_postprocess.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if INTEGRA_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace integra {

namespace {

class OpenCvDnnEngine final : public IInferenceEngine {
 public:
  bool init(const InferenceEngineConfig& cfg) override {
    cfg_ = cfg;
    if (cfg.model_path.empty()) {
      std::cerr << "integra: --engine opencv требует непустой --model (путь к .onnx)\n";
      return false;
    }
    try {
      net_ = cv::dnn::readNet(cfg.model_path);
    } catch (const cv::Exception& e) {
      std::cerr << "integra: OpenCV DNN readNet failed: " << e.what() << "\n";
      return false;
    }
    if (net_.empty()) {
      std::cerr << "integra: OpenCV DNN net is empty\n";
      return false;
    }
    // По умолчанию выбираем CPU-бэкенд для минимального RAM-футпринта daemon.
    // CUDA можно включить явно: INTEGRA_OPENCV_DNN_CUDA=1
    const char* cuda_env = std::getenv("INTEGRA_OPENCV_DNN_CUDA");
    const bool want_cuda = (cuda_env != nullptr && cuda_env[0] == '1');
#if INTEGRA_HAS_CUDA
    if (want_cuda) {
      try {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
      } catch (...) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      }
    } else {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
#else
    (void)want_cuda;
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif
    out_names_ = net_.getUnconnectedOutLayersNames();
    if (out_names_.empty()) {
      out_names_.push_back(net_.getLayerNames().back());
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
        std::cerr << "integra: cudaMemcpy (OpenCV DNN input) failed\n";
        return false;
      }
    } else
#endif
    {
      std::memcpy(host_nchw_.data(), in.nchw, n_elem * sizeof(float));
    }

    int sz[] = {1, 3, H, W};
    cv::Mat blob(4, sz, CV_32F, host_nchw_.data());
    net_.setInput(blob);

    std::vector<cv::Mat> outs;
    auto t0 = std::chrono::steady_clock::now();
    try {
      net_.forward(outs, out_names_);
    } catch (const cv::Exception& e) {
      std::cerr << "integra: OpenCV DNN forward failed: " << e.what() << "\n";
      return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    out.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (outs.empty() || outs[0].empty()) {
      return true;
    }

    const cv::Mat& y = outs[0];
    if (y.dims != 3 || y.size[0] != 1) {
      std::cerr << "integra: OpenCV DNN unexpected output rank, expected [1,C,N] or [1,N,C]\n";
      return false;
    }
    const int d1 = y.size[1];
    const int d2 = y.size[2];
    const bool chw = yolo_output_channel_first(d1, d2);
    const float* raw = y.ptr<float>();

    PostprocessParams pp = cfg_.postprocess;
    std::vector<Detection> decoded;
    decode_yolov8_flat(raw, d1, d2, chw, pp, static_cast<float>(W), static_cast<float>(H),
                       transpose_scratch_, decoded);
    nms_greedy_xyxy(decoded, pp.nms_iou_threshold);
    out.items = std::move(decoded);
    return true;
  }

 private:
  InferenceEngineConfig cfg_;
  cv::dnn::Net net_;
  std::vector<std::string> out_names_;
  std::vector<float> host_nchw_;
  std::vector<float> transpose_scratch_;
};

}  // namespace

std::unique_ptr<IInferenceEngine> make_opencv_dnn_engine() {
  return std::make_unique<OpenCvDnnEngine>();
}

}  // namespace integra

