#include "integra/scene_analyzer.hpp"

#include "integra/geom.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace integra {

namespace {

enum class St {
  kNone = 0,
  kCandidate,
  kStatic,
  kUnattended,
  kAlarmAbandoned,
  kAlarmDisappeared,
};

struct CentroidEntry {
  double ts = 0;
  float cx = 0.f;
  float cy = 0.f;
};

struct TH {
  int track_id = 0;
  int cls_id = 0;
  std::string cls_name;
  St state = St::kNone;
  double first_seen_ts = 0;
  double last_seen_ts = 0;
  BBoxXYXY last_bbox{};
  float last_cx = 0.f;
  float last_cy = 0.f;
  float last_conf = 0.f;
  double static_since_ts = 0;
  double unattended_since_ts = 0;
  double abandoned_at_ts = 0;
  double last_owner_near_ts = 0;
  std::vector<CentroidEntry> centroid_history;
  int presence_count = 0;
  int frames_seen = 0;
  bool raised_abandoned = false;
  bool raised_disappeared = false;

  float area() const { return bbox_area(last_bbox); }

  double displacement_window(double since_ts) const {
    std::vector<CentroidEntry> xs;
    for (const auto& e : centroid_history) {
      if (e.ts >= since_ts) {
        xs.push_back(e);
      }
    }
    if (xs.size() < 2) {
      return 0.0;
    }
    const float dx = xs.back().cx - xs.front().cx;
    const float dy = xs.back().cy - xs.front().cy;
    return std::hypot(static_cast<double>(dx), static_cast<double>(dy));
  }
};

bool is_person_near(const Detection& obj, const std::vector<Detection>& persons, double prox_px) {
  if (persons.empty()) {
    return false;
  }
  float ox, oy;
  centroid_xyxy(obj.bbox, &ox, &oy);
  const float ox1 = obj.bbox.x1, oy1 = obj.bbox.y1, ox2 = obj.bbox.x2, oy2 = obj.bbox.y2;
  for (const auto& person : persons) {
    float px, py;
    centroid_xyxy(person.bbox, &px, &py);
    if (std::hypot(static_cast<double>(px - ox), static_cast<double>(py - oy)) <= prox_px) {
      return true;
    }
    const float px1 = person.bbox.x1, py1 = person.bbox.y1, px2 = person.bbox.x2, py2 = person.bbox.y2;
    if (px1 < ox2 && ox1 < px2 && py1 < oy2 && oy1 < py2) {
      return true;
    }
  }
  return false;
}

AlarmEvent make_ev(const std::string& type, const std::string& cam, double video_pos_ms, const TH& t,
                   const std::string& note) {
  AlarmEvent ev;
  ev.type = type;
  ev.camera_id = cam;
  ev.track_id = t.track_id;
  ev.cls_id = t.cls_id;
  ev.cls_name = t.cls_name;
  ev.confidence = t.last_conf;
  ev.ts_wall_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  ev.video_pos_ms = video_pos_ms;
  ev.bbox[0] = t.last_bbox.x1;
  ev.bbox[1] = t.last_bbox.y1;
  ev.bbox[2] = t.last_bbox.x2;
  ev.bbox[3] = t.last_bbox.y2;
  ev.note = note;
  return ev;
}

}  // namespace

struct SceneAnalyzer::Impl {
  AnalyzerParams p;
  std::unordered_map<int, TH> tracks;
};

SceneAnalyzer::SceneAnalyzer(AnalyzerParams p) : impl_(std::make_unique<Impl>()) { impl_->p = p; }

SceneAnalyzer::~SceneAnalyzer() = default;

void SceneAnalyzer::reset() { impl_->tracks.clear(); }

void SceneAnalyzer::set_params(AnalyzerParams p) { impl_->p = p; }

std::vector<AlarmEvent> SceneAnalyzer::ingest(double ts, double video_pos_ms,
                                               const std::string& camera_id,
                                               const std::vector<Detection>& objects,
                                               const std::vector<Detection>& persons) {
  auto& p = impl_->p;
  auto& tracks = impl_->tracks;
  std::vector<AlarmEvent> events;
  std::vector<int> seen_ids;
  seen_ids.reserve(objects.size());

  const int mlen = std::max(8, p.centroid_history_maxlen);

  for (const auto& det : objects) {
    const int tid = det.track_id;
    if (tid < 0) {
      continue;
    }
    seen_ids.push_back(tid);

    auto it = tracks.find(tid);
    if (it == tracks.end()) {
      TH nt;
      nt.track_id = tid;
      nt.cls_id = det.class_id;
      nt.cls_name = det.cls_name;
      nt.first_seen_ts = ts;
      nt.state = St::kCandidate;
      nt.centroid_history.reserve(static_cast<std::size_t>(mlen));
      tracks.emplace(tid, std::move(nt));
      it = tracks.find(tid);
    }

    TH& tr = it->second;
    tr.last_seen_ts = ts;
    tr.last_bbox = det.bbox;
    centroid_xyxy(det.bbox, &tr.last_cx, &tr.last_cy);
    tr.last_conf = det.confidence;
    tr.presence_count += 1;
    tr.frames_seen += 1;
    CentroidEntry ce;
    ce.ts = ts;
    ce.cx = tr.last_cx;
    ce.cy = tr.last_cy;
    tr.centroid_history.push_back(ce);
    while (static_cast<int>(tr.centroid_history.size()) > mlen) {
      tr.centroid_history.erase(tr.centroid_history.begin());
    }

    if (tr.area() < static_cast<float>(p.min_object_area_px)) {
      continue;
    }

    if (is_person_near(det, persons, p.owner_proximity_px)) {
      tr.last_owner_near_ts = ts;
    }

    const double disp = tr.displacement_window(ts - p.static_window_sec);
    const bool is_static =
        disp <= p.static_displacement_px && (ts - tr.first_seen_ts) >= p.static_window_sec;

    if (tr.state == St::kCandidate && is_static) {
      tr.state = St::kStatic;
      tr.static_since_ts = ts;
    } else if (tr.state == St::kStatic && !is_static) {
      tr.state = St::kCandidate;
      tr.static_since_ts = 0;
      tr.unattended_since_ts = 0;
      continue;
    }

    if (tr.state == St::kStatic || tr.state == St::kUnattended || tr.state == St::kAlarmAbandoned) {
      const double without_owner_for =
          tr.last_owner_near_ts > 0 ? (ts - tr.last_owner_near_ts) : (ts - tr.first_seen_ts);
      if (without_owner_for >= p.owner_left_sec) {
        if (tr.state == St::kStatic) {
          tr.state = St::kUnattended;
          tr.unattended_since_ts = ts;
        }
      }
    }

    if (tr.state == St::kUnattended) {
      const double unattended_for = ts - tr.unattended_since_ts;
      if (unattended_for >= p.abandon_time_sec && !tr.raised_abandoned) {
        tr.state = St::kAlarmAbandoned;
        tr.abandoned_at_ts = ts;
        tr.raised_abandoned = true;
        std::string note = "unattended_for=" + std::to_string(unattended_for);
        events.push_back(make_ev("abandoned", camera_id, video_pos_ms, tr, note));
      }
    }
  }

  std::vector<int> to_drop;
  for (auto& kv : tracks) {
    const int tid = kv.first;
    TH& tr = kv.second;
    bool seen = false;
    for (int s : seen_ids) {
      if (s == tid) {
        seen = true;
        break;
      }
    }
    if (seen) {
      continue;
    }
    const double silent_for = ts - tr.last_seen_ts;
    if (tr.raised_abandoned && !tr.raised_disappeared) {
      if (silent_for >= p.disappear_grace_sec) {
        tr.state = St::kAlarmDisappeared;
        tr.raised_disappeared = true;
        std::string note = "silent_for=" + std::to_string(silent_for);
        events.push_back(make_ev("disappeared", camera_id, video_pos_ms, tr, note));
      }
    }
    if (!tr.raised_abandoned && silent_for > std::max(10.0, p.disappear_grace_sec * 2.0)) {
      to_drop.push_back(tid);
    }
  }
  for (int tid : to_drop) {
    tracks.erase(tid);
  }

  return events;
}

}  // namespace integra
