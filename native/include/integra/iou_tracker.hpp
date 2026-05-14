#pragma once

#include "integra/types.hpp"

#include <vector>

namespace integra {

/// Межкадровый трекер: глобальное сопоставление по IoU + запас по центроиду,
/// краткая память трека при пропуске детекции (аналог max_age в SORT).
class IouTracker {
 public:
  /// @param iou_match_threshold минимальный IoU для «жёсткого» матча
  /// @param max_missed_frames сколько кадров подряд держать трек без детекции (последний bbox)
  /// @param soft_centroid_match вторичное сопоставление по центроиду при низком IoU (дрожащий bbox)
  explicit IouTracker(float iou_match_threshold = 0.35f,
                      int max_missed_frames = 10,
                      bool soft_centroid_match = true);

  /// Присваивает detection[].track_id; координаты в одном пространстве (полный кадр).
  /// frame_width/height > 0 — масштабировать допустимое смещение центроида под разрешение.
  void update(std::vector<Detection>& frame_dets, int frame_width = 0, int frame_height = 0);

  void reset();

 private:
  struct Track {
    int id = 0;
    int class_id = 0;
    BBoxXYXY bbox{};
    int missed = 0;
  };

  float thr_;
  int max_missed_;
  bool soft_centroid_;
  int next_id_ = 1;
  std::vector<Track> active_;
};

}  // namespace integra
