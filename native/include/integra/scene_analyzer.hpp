#pragma once

#include "integra/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace integra {

struct AnalyzerParams {
  double static_displacement_px = 8.0;
  double static_window_sec = 3.0;
  double abandon_time_sec = 15.0;
  double owner_proximity_px = 180.0;
  double owner_left_sec = 5.0;
  double disappear_grace_sec = 4.0;
  double min_object_area_px = 600.0;
  int centroid_history_maxlen = 64;
  int person_class_id = 0;
};

/// FSM abandoned / disappeared — порт логики `app/analyzer.py` (без Web/UI).
class SceneAnalyzer {
 public:
  explicit SceneAnalyzer(AnalyzerParams p);
  ~SceneAnalyzer();
  SceneAnalyzer(const SceneAnalyzer&) = delete;
  SceneAnalyzer& operator=(const SceneAnalyzer&) = delete;

  void reset();
  void set_params(AnalyzerParams p);

  /// `objects` — не-person детекции с заполненным track_id; `persons` — люди.
  std::vector<AlarmEvent> ingest(double ts, double video_pos_ms, const std::string& camera_id,
                                 const std::vector<Detection>& objects,
                                 const std::vector<Detection>& persons);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace integra
