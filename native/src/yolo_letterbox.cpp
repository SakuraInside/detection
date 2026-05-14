#include "integra/yolo_letterbox.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace integra {

bool yolo_letterbox_bgr(const cv::Mat& src_bgr, int dst_edge, cv::Mat& dst_bgr, LetterboxMeta* meta) {
  if (!meta || src_bgr.empty() || dst_edge <= 0) {
    return false;
  }
  const int w = src_bgr.cols;
  const int h = src_bgr.rows;
  if (w <= 0 || h <= 0) {
    return false;
  }
  meta->r = std::min(static_cast<float>(dst_edge) / static_cast<float>(w),
                     static_cast<float>(dst_edge) / static_cast<float>(h));
  const int new_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * meta->r)));
  const int new_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * meta->r)));
  meta->pad_left = (dst_edge - new_w) / 2;
  const int pad_right = dst_edge - new_w - meta->pad_left;
  meta->pad_top = (dst_edge - new_h) / 2;
  const int pad_bottom = dst_edge - new_h - meta->pad_top;

  cv::Mat resized;
  cv::resize(src_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  cv::copyMakeBorder(resized, dst_bgr, meta->pad_top, pad_bottom, meta->pad_left, pad_right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  return true;
}

void yolo_unletterbox_xyxy(BBoxXYXY& b, const LetterboxMeta& meta, int orig_w, int orig_h) {
  const float inv_r = 1.f / std::max(meta.r, 1e-6f);
  const float pl = static_cast<float>(meta.pad_left);
  const float pt = static_cast<float>(meta.pad_top);
  b.x1 = (b.x1 - pl) * inv_r;
  b.x2 = (b.x2 - pl) * inv_r;
  b.y1 = (b.y1 - pt) * inv_r;
  b.y2 = (b.y2 - pt) * inv_r;

  const float mx = static_cast<float>(std::max(0, orig_w - 1));
  const float my = static_cast<float>(std::max(0, orig_h - 1));
  b.x1 = std::max(0.f, std::min(b.x1, mx));
  b.x2 = std::max(0.f, std::min(b.x2, mx));
  b.y1 = std::max(0.f, std::min(b.y1, my));
  b.y2 = std::max(0.f, std::min(b.y2, my));
  if (b.x2 <= b.x1) {
    b.x2 = std::min(mx, b.x1 + 1.f);
  }
  if (b.y2 <= b.y1) {
    b.y2 = std::min(my, b.y1 + 1.f);
  }
}

void yolo_unletterbox_dets(std::vector<Detection>& dets, const LetterboxMeta& meta, int orig_w,
                           int orig_h) {
  for (auto& d : dets) {
    yolo_unletterbox_xyxy(d.bbox, meta, orig_w, orig_h);
  }
}

}  // namespace integra
