#pragma once

#include "integra/types.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace cv {
class Mat;
}

namespace integra {

/// Захват из файла (CAP_FFMPEG). RTSP/камера — тот же API OpenCV.
class VideoFileSource {
 public:
  VideoFileSource();
  ~VideoFileSource();

  VideoFileSource(const VideoFileSource&) = delete;
  VideoFileSource& operator=(const VideoFileSource&) = delete;

  bool open(const std::string& path);
  void close();

  /// Последний кадр BGR uint8, continuous.
  bool read_next(cv::Mat& out_bgr, FrameMeta& meta);

  double fps() const { return fps_; }
  std::int64_t frame_count() const { return frame_count_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  double fps_ = 30.0;
  std::int64_t frame_count_ = 0;
};

}  // namespace integra
