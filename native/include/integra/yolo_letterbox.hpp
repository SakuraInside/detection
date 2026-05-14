#pragma once

#include "integra/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace integra {

/// Параметры letterbox (Ultralytics): r = min(dst/w, dst/h), паддинг до квадрата dst×dst.
struct LetterboxMeta {
  float r = 1.f;
  int pad_left = 0;
  int pad_top = 0;
};

/// BGR → квадрат dst×dst с сохранением пропорций и серой окантовкой (114,114,114).
bool yolo_letterbox_bgr(const cv::Mat& src_bgr, int dst_edge, cv::Mat& dst_bgr, LetterboxMeta* meta);

/// Координаты из пространства letterbox-квадрата → исходный кадр src_bgr.cols × rows.
void yolo_unletterbox_xyxy(BBoxXYXY& box, const LetterboxMeta& meta, int orig_w, int orig_h);

void yolo_unletterbox_dets(std::vector<Detection>& dets, const LetterboxMeta& meta, int orig_w,
                           int orig_h);

}  // namespace integra
