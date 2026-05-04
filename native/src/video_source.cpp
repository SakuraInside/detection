#include "integra/video_source.hpp"

#include <memory>
#include <opencv2/opencv.hpp>

namespace integra {

struct VideoFileSource::Impl {
  cv::VideoCapture cap;
  std::uint64_t next_frame_id = 0;
};

VideoFileSource::VideoFileSource() : impl_(std::make_unique<Impl>()) {}

VideoFileSource::~VideoFileSource() { close(); }

bool VideoFileSource::open(const std::string& path) {
  close();
#if CV_VERSION_MAJOR >= 4
  impl_->cap.open(path, cv::CAP_FFMPEG);
#else
  impl_->cap.open(path);
#endif
  if (!impl_->cap.isOpened()) {
    return false;
  }
  fps_ = impl_->cap.get(cv::CAP_PROP_FPS);
  if (fps_ <= 1e-3) {
    fps_ = 30.0;
  }
  frame_count_ = static_cast<std::int64_t>(impl_->cap.get(cv::CAP_PROP_FRAME_COUNT));
  return true;
}

void VideoFileSource::close() {
  if (impl_->cap.isOpened()) {
    impl_->cap.release();
  }
}

bool VideoFileSource::read_next(cv::Mat& out_bgr, FrameMeta& meta) {
  if (!impl_->cap.isOpened()) {
    return false;
  }
  cv::Mat tmp;
  if (!impl_->cap.read(tmp) || tmp.empty()) {
    return false;
  }
  if (!tmp.isContinuous()) {
    tmp.copyTo(out_bgr);
  } else {
    out_bgr = std::move(tmp);
  }
  meta.width = out_bgr.cols;
  meta.height = out_bgr.rows;
  meta.pos_ms = impl_->cap.get(cv::CAP_PROP_POS_MSEC);
  meta.frame_id = ++impl_->next_frame_id;
  return true;
}

}  // namespace integra
