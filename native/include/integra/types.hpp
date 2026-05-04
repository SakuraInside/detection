#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace integra {

struct FrameMeta {
  std::uint64_t frame_id = 0;
  double pos_ms = 0.0;
  int width = 0;
  int height = 0;
};

struct BBoxXYXY {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;
};

/// Одна детекция (после масштабирования в координаты полного кадра).
struct Detection {
  int track_id = -1;
  int class_id = 0;
  std::string cls_name;
  float confidence = 0.f;
  BBoxXYXY bbox{};
};

/// Пакет для постобработки / трекера / FSM (аналог Python DetectionResult).
struct DetectionBatch {
  std::vector<Detection> items;
  float inference_ms = 0.f;
};

/// Событие для внешней системы (JSON line → integra-alarmd или SIEM).
struct AlarmEvent {
  std::string type;       // abandoned | disappeared | ...
  std::string camera_id;
  std::int64_t track_id = -1;
  int cls_id = 0;
  std::string cls_name;
  double confidence = 0.0;
  double ts_wall_ms = 0.0;
  double video_pos_ms = 0.0;
  float bbox[4] = {0, 0, 0, 0};
  std::string note;
};

}  // namespace integra
