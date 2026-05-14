#include "integra/iou_tracker.hpp"

#include "integra/geom.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace integra {

namespace {

constexpr float kCenterIouFloor = 0.018f;

float center_dist_xyxy(const BBoxXYXY& a, const BBoxXYXY& b) {
  float ax = 0.f;
  float ay = 0.f;
  float bx = 0.f;
  float by = 0.f;
  centroid_xyxy(a, &ax, &ay);
  centroid_xyxy(b, &bx, &by);
  const float dx = ax - bx;
  const float dy = ay - by;
  return std::sqrt(dx * dx + dy * dy);
}

float max_bbox_side(const BBoxXYXY& b) {
  const float w = std::max(0.f, b.x2 - b.x1);
  const float h = std::max(0.f, b.y2 - b.y1);
  return std::max(w, h);
}

float center_match_cap_px(int frame_w, int frame_h) {
  if (frame_w > 0 && frame_h > 0) {
    const float diag =
        std::hypot(static_cast<float>(frame_w), static_cast<float>(frame_h));
    return std::max(36.f, 0.07f * diag);
  }
  return 64.f;
}

/// Вторичный матч при дрожащем bbox: небольшой IoU, но центры близко относительно размера бокса.
bool secondary_centroid_match(float iou,
                              float dist_px,
                              float cap_px,
                              const BBoxXYXY& prev,
                              const BBoxXYXY& det) {
  if (iou < kCenterIouFloor || dist_px > cap_px) {
    return false;
  }
  const float scale = std::max(8.f, std::max(max_bbox_side(prev), max_bbox_side(det)));
  return dist_px <= 0.45f * scale;
}

float pair_match_score(float iou,
                       bool hard_iou,
                       bool soft_centroid,
                       float dist_px,
                       float cap_px) {
  if (hard_iou) {
    return iou;
  }
  if (soft_centroid) {
    return 0.12f + 0.35f * (1.f - std::min(1.f, dist_px / std::max(1.f, cap_px)));
  }
  return -1.f;
}

struct Cand {
  float score = -1.f;
  float iou = 0.f;
  std::size_t ti = 0;
  std::size_t dj = 0;
};

}  // namespace

IouTracker::IouTracker(float iou_match_threshold, int max_missed_frames, bool soft_centroid_match)
    : thr_(iou_match_threshold),
      max_missed_(max_missed_frames > 0 ? max_missed_frames : 1),
      soft_centroid_(soft_centroid_match) {}

void IouTracker::reset() {
  active_.clear();
  next_id_ = 1;
}

void IouTracker::update(std::vector<Detection>& dets, int frame_w, int frame_h) {
  const float cap_px = center_match_cap_px(frame_w, frame_h);

  std::vector<Cand> pairs;
  pairs.reserve(active_.size() * dets.size() + 8);

  for (std::size_t ti = 0; ti < active_.size(); ++ti) {
    const Track& tr = active_[ti];
    for (std::size_t dj = 0; dj < dets.size(); ++dj) {
      if (dets[dj].class_id != tr.class_id) {
        continue;
      }
      const float iou = iou_xyxy(tr.bbox, dets[dj].bbox);
      const float dist = center_dist_xyxy(tr.bbox, dets[dj].bbox);
      const bool hard = iou >= thr_;
      const bool soft = soft_centroid_ && !hard &&
                         secondary_centroid_match(iou, dist, cap_px, tr.bbox, dets[dj].bbox);
      const float score = pair_match_score(iou, hard, soft, dist, cap_px);
      if (score < 0.f) {
        continue;
      }
      Cand c;
      c.score = score;
      c.iou = iou;
      c.ti = ti;
      c.dj = dj;
      pairs.push_back(c);
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const Cand& a, const Cand& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    if (a.iou != b.iou) {
      return a.iou > b.iou;
    }
    if (a.ti != b.ti) {
      return a.ti < b.ti;
    }
    return a.dj < b.dj;
  });

  std::vector<char> t_used(active_.size(), 0);
  std::vector<char> d_used(dets.size(), 0);
  std::vector<Track> next;
  next.reserve(active_.size() + dets.size());

  for (const auto& p : pairs) {
    const std::size_t ti = p.ti;
    const std::size_t dj = p.dj;
    if (ti >= active_.size() || dj >= dets.size()) {
      continue;
    }
    if (t_used[ti] || d_used[dj]) {
      continue;
    }
    t_used[ti] = 1;
    d_used[dj] = 1;
    Track nt;
    nt.id = active_[ti].id;
    nt.class_id = dets[dj].class_id;
    nt.bbox = dets[dj].bbox;
    nt.missed = 0;
    dets[dj].track_id = nt.id;
    next.push_back(std::move(nt));
  }

  for (std::size_t ti = 0; ti < active_.size(); ++ti) {
    if (t_used[ti]) {
      continue;
    }
    Track t = active_[ti];
    t.missed += 1;
    if (t.missed <= max_missed_) {
      next.push_back(std::move(t));
    }
  }

  for (std::size_t dj = 0; dj < dets.size(); ++dj) {
    if (d_used[dj]) {
      continue;
    }
    Track nt;
    nt.id = next_id_++;
    nt.class_id = dets[dj].class_id;
    nt.bbox = dets[dj].bbox;
    nt.missed = 0;
    dets[dj].track_id = nt.id;
    next.push_back(std::move(nt));
  }

  active_.swap(next);
}

}  // namespace integra
