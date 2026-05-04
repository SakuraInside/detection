#pragma once

#include <cstddef>
#include <vector>

#ifndef INTEGRA_HAS_CUDA
#define INTEGRA_HAS_CUDA 0
#endif

#if INTEGRA_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace cv {
class Mat;
}

namespace integra {

/// Держит device-буферы под один размер кадра; resize при смене геометрии.
class GpuLetterboxPrep {
 public:
  GpuLetterboxPrep();
  ~GpuLetterboxPrep();

  GpuLetterboxPrep(const GpuLetterboxPrep&) = delete;
  GpuLetterboxPrep& operator=(const GpuLetterboxPrep&) = delete;

#if INTEGRA_HAS_CUDA
  /// Копирует BGR с host в GPU и заполняет float NCHW [1,3,H,W] на device.
  bool upload_and_preprocess(const cv::Mat& bgr_host, int& out_w, int& out_h);
  /// Указатель на device float NCHW.
  float* device_nchw() const { return d_nchw_; }
  std::size_t nchw_bytes() const { return nchw_bytes_; }
  /// Поток, на котором выполнялся преподготовка (для `InferenceInput::cuda_stream` / TensorRT).
  void* cuda_stream() const { return static_cast<void*>(stream_); }
  /// Явная блокировка до завершения работы на preprocess-потоке (ошибка / shutdown).
  void sync() const { cudaStreamSynchronize(stream_); }
#else
  bool upload_and_preprocess(const cv::Mat& bgr_host, std::vector<float>& out_cpu_nchw);
#endif

 private:
#if INTEGRA_HAS_CUDA
  void free_buffers();
  unsigned char* d_bgr_ = nullptr;
  float* d_nchw_ = nullptr;
  std::size_t cap_bgr_ = 0;
  std::size_t cap_nchw_ = 0;
  std::size_t nchw_bytes_ = 0;
  int last_w_ = 0;
  int last_h_ = 0;
  cudaStream_t stream_{};
#endif
};

}  // namespace integra
