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
  double owner_proximity_px = 115.0;
  double owner_left_sec = 5.0;
  double disappear_grace_sec = 9.0;
  double min_object_area_px = 600.0;
  int centroid_history_maxlen = 64;
  int person_class_id = 0;
  // Жесткий предел активных треков в памяти (защита от всплеска ложных детекций).
  int max_active_tracks = 256;
};

struct TrackSnapshot {
  int id = 0;
  std::string cls;
  std::string state;  // candidate|static|unattended|alarm_abandoned|alarm_disappeared
  float bbox[4] = {0, 0, 0, 0};
  float conf = 0.f;
  double static_for_sec = 0.0;
  double unattended_for_sec = 0.0;
  bool alarm = false;
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
  /// Переход static→unattended только после `ever_owner_near` (человек хотя бы раз был рядом);
  /// иначе весь стационарный фон превращается в «бесхозный». `frame_w` / `frame_h` — отсев
  /// гигантских FP person в `is_person_near` (площадь/аспект бокса).
  std::vector<AlarmEvent> ingest(double ts, double video_pos_ms, const std::string& camera_id,
                                 const std::vector<Detection>& objects,
                                 const std::vector<Detection>& persons,
                                 int frame_w = 0, int frame_h = 0);

  /// Снимок треков для UI: без `candidate` и без `static` (только unattended / тревоги).
  std::vector<TrackSnapshot> tracks_snapshot(double now_ts) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace integra
