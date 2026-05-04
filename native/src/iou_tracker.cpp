#include "integra/iou_tracker.hpp"

#include "integra/geom.hpp"

#include <cstddef>
#include <limits>

namespace integra {

IouTracker::IouTracker(float iou_match_threshold) : thr_(iou_match_threshold) {}

void IouTracker::reset() {
  active_.clear();
  next_id_ = 1;
}

void IouTracker::update(std::vector<Detection>& dets) {
  std::vector<Track> next_active;
  next_active.reserve(dets.size());
  std::vector<char> det_used(dets.size(), 0);

  for (const auto& prev : active_) {
    float best_iou = 0.f;
    std::size_t best_j = std::numeric_limits<std::size_t>::max();
    for (std::size_t j = 0; j < dets.size(); ++j) {
      if (det_used[j]) {
        continue;
      }
      if (dets[j].class_id != prev.class_id) {
        continue;
      }
      const float i = iou_xyxy(prev.bbox, dets[j].bbox);
      if (i > best_iou) {
        best_iou = i;
        best_j = j;
      }
    }
    if (best_j != std::numeric_limits<std::size_t>::max() && best_iou >= thr_) {
      dets[best_j].track_id = prev.id;
      Track t;
      t.id = prev.id;
      t.class_id = dets[best_j].class_id;
      t.bbox = dets[best_j].bbox;
      next_active.push_back(t);
      det_used[best_j] = 1;
    }
  }

  for (std::size_t j = 0; j < dets.size(); ++j) {
    if (det_used[j]) {
      continue;
    }
    Track t;
    t.id = next_id_++;
    t.class_id = dets[j].class_id;
    t.bbox = dets[j].bbox;
    dets[j].track_id = t.id;
    next_active.push_back(t);
  }

  active_.swap(next_active);
}

}  // namespace integra
