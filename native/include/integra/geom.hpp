#pragma once

#include "integra/types.hpp"

#include <algorithm>
#include <cmath>

namespace integra {

inline float iou_xyxy(const BBoxXYXY& a, const BBoxXYXY& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.f, xx2 - xx1);
  const float h = std::max(0.f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  const float u = area_a + area_b - inter;
  return u > 1e-6f ? inter / u : 0.f;
}

inline void centroid_xyxy(const BBoxXYXY& b, float* cx, float* cy) {
  *cx = 0.5f * (b.x1 + b.x2);
  *cy = 0.5f * (b.y1 + b.y2);
}

inline float bbox_area(const BBoxXYXY& b) {
  return std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
}

inline BBoxXYXY scale_bbox_xyxy(const BBoxXYXY& b, float sx, float sy) {
  return {b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy};
}

}  // namespace integra
