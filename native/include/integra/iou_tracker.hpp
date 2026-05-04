#pragma once

#include "integra/types.hpp"

#include <vector>

namespace integra {

/// Простой межкадровый трекер по IoU с разделением по class_id (замена BoT-SORT на первом этапе).
class IouTracker {
 public:
  explicit IouTracker(float iou_match_threshold = 0.35f);

  /// Присваивает detection[].track_id; координаты в одном пространстве (полный кадр).
  void update(std::vector<Detection>& frame_dets);

  void reset();

 private:
  struct Track {
    int id = 0;
    int class_id = 0;
    BBoxXYXY bbox{};
  };

  float thr_;
  int next_id_ = 1;
  std::vector<Track> active_;
};

}  // namespace integra
