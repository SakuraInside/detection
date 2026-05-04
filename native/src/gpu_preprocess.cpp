#include "integra/gpu_preprocess.hpp"

#include <opencv2/opencv.hpp>

#if INTEGRA_HAS_CUDA
#include <cuda_runtime.h>

extern "C" cudaError_t integra_cuda_bgr_to_nchw_float(const unsigned char* d_bgr_hwc,
                                                         float* d_nchw,
                                                         int width,
                                                         int height,
                                                         cudaStream_t stream);
#endif

namespace integra {

#if INTEGRA_HAS_CUDA

GpuLetterboxPrep::GpuLetterboxPrep() {
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
}

GpuLetterboxPrep::~GpuLetterboxPrep() {
  free_buffers();
  cudaStreamDestroy(stream_);
}

void GpuLetterboxPrep::free_buffers() {
  if (d_bgr_) {
    cudaFree(d_bgr_);
    d_bgr_ = nullptr;
  }
  if (d_nchw_) {
    cudaFree(d_nchw_);
    d_nchw_ = nullptr;
  }
  cap_bgr_ = cap_nchw_ = nchw_bytes_ = 0;
  last_w_ = last_h_ = 0;
}

bool GpuLetterboxPrep::upload_and_preprocess(const cv::Mat& bgr_host, int& out_w, int& out_h) {
  cv::Mat cont;
  if (!bgr_host.isContinuous() || bgr_host.type() != CV_8UC3) {
    if (bgr_host.type() != CV_8UC3) {
      return false;
    }
    bgr_host.copyTo(cont);
  } else {
    cont = bgr_host;
  }
  CV_Assert(cont.type() == CV_8UC3 && cont.isContinuous());
  const int w = cont.cols;
  const int h = cont.rows;
  out_w = w;
  out_h = h;
  const std::size_t need_bgr = static_cast<std::size_t>(w * h * 3);
  const std::size_t need_nchw = static_cast<std::size_t>(3 * w * h) * sizeof(float);
  if (w != last_w_ || h != last_h_) {
    free_buffers();
    last_w_ = w;
    last_h_ = h;
  }
  if (cap_bgr_ < need_bgr) {
    if (d_bgr_) {
      cudaFree(d_bgr_);
    }
    cudaMalloc(&d_bgr_, need_bgr);
    cap_bgr_ = need_bgr;
  }
  if (cap_nchw_ < need_nchw) {
    if (d_nchw_) {
      cudaFree(d_nchw_);
    }
    cudaMalloc(&d_nchw_, need_nchw);
    cap_nchw_ = need_nchw;
  }
  nchw_bytes_ = need_nchw;
  cudaError_t e =
      cudaMemcpyAsync(d_bgr_, cont.ptr<unsigned char>(), need_bgr, cudaMemcpyHostToDevice, stream_);
  if (e != cudaSuccess) {
    return false;
  }
  e = integra_cuda_bgr_to_nchw_float(d_bgr_, d_nchw_, w, h, stream_);
  if (e != cudaSuccess) {
    return false;
  }
  // Не синхронизируем здесь: инференс должен выполняться на том же cuda_stream (см. InferenceInput)
  // или вызвать sync() перед следующим кадром. Stub/tensorrt/onnx ждут поток в infer().
  return true;
}

#else

GpuLetterboxPrep::GpuLetterboxPrep() = default;
GpuLetterboxPrep::~GpuLetterboxPrep() = default;

bool GpuLetterboxPrep::upload_and_preprocess(const cv::Mat& bgr_host, std::vector<float>& out_cpu_nchw) {
  cv::Mat cont;
  if (!bgr_host.isContinuous()) {
    bgr_host.copyTo(cont);
  } else {
    cont = bgr_host;
  }
  CV_Assert(cont.type() == CV_8UC3);
  const int h = cont.rows;
  const int w = cont.cols;
  out_cpu_nchw.resize(static_cast<std::size_t>(3 * h * w));
  const float inv = 1.0f / 255.0f;
  for (int y = 0; y < h; ++y) {
    const unsigned char* row = cont.ptr<unsigned char>(y);
    for (int x = 0; x < w; ++x) {
      unsigned char b = row[x * 3 + 0];
      unsigned char g = row[x * 3 + 1];
      unsigned char r = row[x * 3 + 2];
      int i = y * w + x;
      out_cpu_nchw[0 * h * w + i] = static_cast<float>(r) * inv;
      out_cpu_nchw[1 * h * w + i] = static_cast<float>(g) * inv;
      out_cpu_nchw[2 * h * w + i] = static_cast<float>(b) * inv;
    }
  }
  return true;
}

#endif

}  // namespace integra
