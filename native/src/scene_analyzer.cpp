#include "integra/scene_analyzer.hpp"

#include "integra/geom.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace integra {

namespace {

// Кандидаты/статика не попадают в tracks_snapshot (оверлей): показываем только
// unattended и тревоги. FSM по-прежнему ведёт кандидатов внутри ingest.
constexpr double kCandidateDropSilentSec = 2.0;
// Троттлинг повторных person_interaction по одному региону.
constexpr double kInteractionThrottleSec = 1.5;

enum class St {
  kNone = 0,
  kCandidate,
  kStatic,
  kUnattended,
  kAlarmUnattended,
  kAlarmRemoved,
  kAlarmMissing,
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
  double last_owner_near_ts = 0;
  double last_interaction_emit_ts = 0;
  bool ever_owner_near = false;
  bool owner_near_prev = false;
  bool was_confirmed = false;  // достигал статики хотя бы раз (реальный объект)
  std::vector<CentroidEntry> centroid_history;
  int presence_count = 0;
  int frames_seen = 0;
  bool object_left_emitted = false;
  bool raised_unattended = false;
  bool raised_removed = false;
  bool raised_missing = false;

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

/// Человек перекрывает bbox объекта (окклюзия): не копим silent / не спешим с missing.
bool person_overlaps_bbox(const BBoxXYXY& obj_bbox, const std::vector<Detection>& persons,
                          int frame_w, int frame_h) {
  if (persons.empty()) {
    return false;
  }
  const float frame_area =
      (frame_w > 0 && frame_h > 0)
          ? static_cast<float>(std::max(1, frame_w) * std::max(1, frame_h))
          : 0.f;
  for (const auto& person : persons) {
    if (person.confidence < 0.30f) {
      continue;
    }
    if (frame_area > 1.f) {
      const float pa = bbox_area(person.bbox);
      if (pa / frame_area > 0.14f) {
        continue;
      }
    }
    if (iou_xyxy(person.bbox, obj_bbox) >= 0.052f) {
      return true;
    }
  }
  return false;
}

bool is_person_near(const Detection& obj, const std::vector<Detection>& persons, double prox_px,
                     int frame_w, int frame_h) {
  if (persons.empty()) {
    return false;
  }
  float ox, oy;
  centroid_xyxy(obj.bbox, &ox, &oy);
  const float ox1 = obj.bbox.x1, oy1 = obj.bbox.y1, ox2 = obj.bbox.x2, oy2 = obj.bbox.y2;

  const float frame_area =
      (frame_w > 0 && frame_h > 0)
          ? static_cast<float>(std::max(1, frame_w) * std::max(1, frame_h))
          : 0.f;

  constexpr float kOwnerPersonMinConf = 0.44f;
  // Не считать «владельцем» гигантские FP (стол/стена как person) — иначе цепляются все
  // объекты в радиусе и потом уходят в unattended по всему кадру.
  constexpr float kOwnerPersonMaxAreaRatio = 0.09f;
  // Силуэт человека для proximity: слишком широкий бокс ≈ мебель/FP.
  constexpr float kOwnerPersonMaxAspect = 1.42f;
  // Минимальное пересечение person–object: иначе «рядом по пикселям» цепляет объекты
  // на столе при проходе человека у лестницы / у стойки.
  constexpr float kOwnerObjectMinIou = 0.055f;

  double eff_prox = prox_px;
  if (frame_w > 0 && frame_h > 0) {
    const double diag =
        std::hypot(static_cast<double>(frame_w), static_cast<double>(frame_h));
    eff_prox = std::min(prox_px, std::max(48.0, 0.036 * diag));
  }

  for (const auto& person : persons) {
    if (person.confidence < kOwnerPersonMinConf) {
      continue;
    }
    if (frame_area > 1.f) {
      const float pa = bbox_area(person.bbox);
      if (pa / frame_area > kOwnerPersonMaxAreaRatio) {
        continue;
      }
      const float pw = person.bbox.x2 - person.bbox.x1;
      const float ph = person.bbox.y2 - person.bbox.y1;
      if (pw > 1.f && ph > 1.f) {
        const float par = pw / ph;
        if (par > kOwnerPersonMaxAspect) {
          continue;
        }
      }
    }
    float px, py;
    centroid_xyxy(person.bbox, &px, &py);
    const double dist =
        std::hypot(static_cast<double>(px - ox), static_cast<double>(py - oy));
    if (dist > eff_prox) {
      continue;
    }
    const float iou_po = iou_xyxy(person.bbox, obj.bbox);
    if (iou_po >= kOwnerObjectMinIou) {
      return true;
    }
    // Только явная близость по центроиду (не полный диск eff_prox при нулевом IoU).
    if (dist <= 0.28 * eff_prox) {
      return true;
    }
    const float px1 = person.bbox.x1, py1 = person.bbox.y1, px2 = person.bbox.x2, py2 = person.bbox.y2;
    if (px1 < ox2 && ox1 < px2 && py1 < oy2 && oy1 < py2 && iou_po >= 0.028f) {
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

SceneAnalyzer::SceneAnalyzer(AnalyzerParams p) : impl_(std::make_unique<Impl>()) {
  impl_->p = p;
}

SceneAnalyzer::~SceneAnalyzer() = default;

void SceneAnalyzer::reset() { impl_->tracks.clear(); }

void SceneAnalyzer::set_params(AnalyzerParams p) { impl_->p = p; }

std::vector<AlarmEvent> SceneAnalyzer::ingest(double ts, double video_pos_ms,
                                               const std::string& camera_id,
                                               const std::vector<Detection>& objects,
                                               const std::vector<Detection>& persons,
                                               int frame_w, int frame_h,
                                               const cv::Mat& /*bgr_frame*/) {
  auto& p = impl_->p;
  auto& tracks = impl_->tracks;
  std::vector<AlarmEvent> events;
  std::vector<int> seen_ids;
  seen_ids.reserve(objects.size());

  const int mlen = std::max(8, p.centroid_history_maxlen);

  // ---- Обновление регионов-кандидатов (class-agnostic объекты сцены) ----
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

    // Близость / взаимодействие человека.
    const bool owner_near =
        is_person_near(det, persons, p.owner_proximity_px, frame_w, frame_h);
    if (owner_near) {
      tr.last_owner_near_ts = ts;
      tr.ever_owner_near = true;
    }

    // Статичность по окну.
    const double disp = tr.displacement_window(ts - p.static_window_sec);
    const bool is_static =
        disp <= p.static_displacement_px && (ts - tr.first_seen_ts) >= p.static_window_sec;

    if (tr.state == St::kCandidate && is_static) {
      tr.state = St::kStatic;
      tr.static_since_ts = ts;
      tr.was_confirmed = true;
    } else if (tr.state == St::kStatic && !is_static) {
      // Объект снова задвигался — назад в кандидаты, сбрасываем «оставленность».
      tr.state = St::kCandidate;
      tr.static_since_ts = 0;
      tr.unattended_since_ts = 0;
      tr.object_left_emitted = false;
      tr.owner_near_prev = owner_near;
      continue;
    }

    // person_interaction: вход человека в зону подтверждённого объекта (false→true).
    if (owner_near && !tr.owner_near_prev && tr.was_confirmed &&
        (ts - tr.last_interaction_emit_ts) > kInteractionThrottleSec) {
      events.push_back(
          make_ev("person_interaction", camera_id, video_pos_ms, tr, "owner near object"));
      tr.last_interaction_emit_ts = ts;
    }
    tr.owner_near_prev = owner_near;

    // Владелец вернулся к ещё не-тревожному unattended — отменяем «оставленность».
    if (owner_near && tr.state == St::kUnattended && !tr.raised_unattended) {
      tr.state = St::kStatic;
      tr.unattended_since_ts = 0;
      tr.object_left_emitted = false;
    }

    // object_left → object_unattended.
    if (tr.state == St::kStatic || tr.state == St::kUnattended ||
        tr.state == St::kAlarmUnattended) {
      // Без хотя бы одного «владелец рядом» постоянный фон не считаем объектом сцены.
      if (!tr.ever_owner_near) {
        continue;
      }
      const double without_owner_for =
          tr.last_owner_near_ts > 0 ? (ts - tr.last_owner_near_ts) : (ts - tr.first_seen_ts);
      if (without_owner_for >= p.owner_left_sec && tr.state == St::kStatic) {
        tr.state = St::kUnattended;
        tr.unattended_since_ts = ts;
        if (!tr.object_left_emitted) {
          events.push_back(make_ev("object_left", camera_id, video_pos_ms, tr,
                                    "owner left object"));
          tr.object_left_emitted = true;
        }
      }
    }

    if (tr.state == St::kUnattended) {
      const double unattended_for = ts - tr.unattended_since_ts;
      if (unattended_for >= p.abandon_time_sec && !tr.raised_unattended) {
        tr.state = St::kAlarmUnattended;
        tr.raised_unattended = true;
        std::string note = "unattended_for=" + std::to_string(unattended_for);
        events.push_back(make_ev("object_unattended", camera_id, video_pos_ms, tr, note));
      }
    }
  }

  // ---- Исчезновения: object_removed (после взаимодействия) / object_missing ----
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
    // Окклюзия человеком — держим трек, не считаем исчезнувшим.
    if (!persons.empty() && person_overlaps_bbox(tr.last_bbox, persons, frame_w, frame_h)) {
      tr.last_seen_ts = ts;
      continue;
    }
    const double silent_for = ts - tr.last_seen_ts;

    if (tr.was_confirmed && !tr.raised_removed && !tr.raised_missing &&
        silent_for >= p.disappear_grace_sec) {
      // Недавнее взаимодействие перед исчезновением → «забрали».
      const bool recent_interaction =
          tr.last_owner_near_ts > 0 &&
          (tr.last_seen_ts - tr.last_owner_near_ts) <= p.owner_left_sec;
      if (recent_interaction) {
        tr.state = St::kAlarmRemoved;
        tr.raised_removed = true;
        std::string note = "removed_after_interaction silent_for=" + std::to_string(silent_for);
        events.push_back(make_ev("object_removed", camera_id, video_pos_ms, tr, note));
      } else {
        tr.state = St::kAlarmMissing;
        tr.raised_missing = true;
        std::string note = "silent_for=" + std::to_string(silent_for);
        events.push_back(make_ev("object_missing", camera_id, video_pos_ms, tr, note));
      }
    }

    // Уборка треков.
    if (tr.state == St::kCandidate && silent_for >= kCandidateDropSilentSec) {
      to_drop.push_back(tid);
    } else if ((tr.raised_removed || tr.raised_missing) &&
               silent_for > p.disappear_grace_sec * 2.0) {
      to_drop.push_back(tid);
    } else if (!tr.was_confirmed &&
               silent_for > std::max(22.0, p.disappear_grace_sec * 3.0)) {
      to_drop.push_back(tid);
    }
  }
  for (int tid : to_drop) {
    tracks.erase(tid);
  }

  // ---- Жёсткий потолок активных треков (защита от всплеска FP) ----
  const int max_tracks = std::max(64, p.max_active_tracks);
  if (static_cast<int>(tracks.size()) > max_tracks) {
    std::vector<std::tuple<double, int>> evict;
    evict.reserve(tracks.size());
    for (const auto& kv : tracks) {
      const TH& t = kv.second;
      if (t.raised_unattended || t.raised_removed || t.raised_missing) {
        continue;
      }
      evict.emplace_back(t.last_seen_ts, kv.first);
    }
    std::sort(evict.begin(), evict.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    int overflow = static_cast<int>(tracks.size()) - max_tracks;
    for (const auto& it : evict) {
      if (overflow <= 0) {
        break;
      }
      tracks.erase(std::get<1>(it));
      --overflow;
    }
  }

  return events;
}

std::vector<TrackSnapshot> SceneAnalyzer::tracks_snapshot(double now_ts) const {
  std::vector<TrackSnapshot> out;
  out.reserve(impl_->tracks.size());
  for (const auto& kv : impl_->tracks) {
    const TH& t = kv.second;
    // На оверлей выводим только unattended и тревоги (без candidate/static/none).
    if (t.state == St::kCandidate || t.state == St::kStatic || t.state == St::kNone) {
      continue;
    }
    TrackSnapshot s;
    s.id = t.track_id;
    s.cls = t.cls_name;
    switch (t.state) {
      case St::kNone:
      case St::kCandidate:
        s.state = "candidate";
        break;
      case St::kStatic:
        s.state = "static";
        break;
      case St::kUnattended:
        s.state = "unattended";
        break;
      case St::kAlarmUnattended:
        s.state = "alarm_unattended";
        break;
      case St::kAlarmRemoved:
        s.state = "alarm_removed";
        break;
      case St::kAlarmMissing:
        s.state = "alarm_missing";
        break;
    }
    s.bbox[0] = t.last_bbox.x1;
    s.bbox[1] = t.last_bbox.y1;
    s.bbox[2] = t.last_bbox.x2;
    s.bbox[3] = t.last_bbox.y2;
    s.conf = t.last_conf;
    s.static_for_sec = t.static_since_ts > 0 ? std::max(0.0, now_ts - t.static_since_ts) : 0.0;
    s.unattended_for_sec =
        t.unattended_since_ts > 0 ? std::max(0.0, now_ts - t.unattended_since_ts) : 0.0;
    s.alarm = (t.state == St::kAlarmUnattended || t.state == St::kAlarmRemoved ||
               t.state == St::kAlarmMissing);
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace integra
