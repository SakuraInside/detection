// integra_ffi.cpp — реализация C ABI поверх integra_core.
//
// Архитектура одного pipeline:
//
//   bgr → IStreamContext::infer        (ISharedEngine — кэш для tensorrt)
//       → filter by class whitelist     (frame_filter.hpp)
//       → apply_frame_filter (anti-noise)
//       → IouTracker::update
//       → SceneAnalyzer::ingest          (FSM abandoned / disappeared)
//       → callback("event", ...) × N
//       → callback("frame_result", ...)  (tracks + persons + stats)
//
// JSON формируется руками через ostringstream (нет внешних зависимостей).

#include "integra/integra_ffi.h"

#include "integra/frame_filter.hpp"
#include "integra/inference_engine.hpp"
#include "integra/iou_tracker.hpp"
#include "integra/scene_analyzer.hpp"
#include "integra/types.hpp"
#include "integra/yolo_postprocess.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// JSON helpers (тот же стиль, что в analyticsd_main.cpp).
// ---------------------------------------------------------------------------
std::string json_esc(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned int>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

double now_wall_sec() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

// ---------------------------------------------------------------------------
// IntegraPipeline (внутренняя реализация opaque-структуры).
// ---------------------------------------------------------------------------
struct IntegraPipeline {
  std::string camera_id;
  std::shared_ptr<integra::ISharedEngine> shared;
  std::unique_ptr<integra::IStreamContext> ctx;
  integra::IouTracker tracker;
  std::unique_ptr<integra::SceneAnalyzer> analyzer;
  integra::FrameFilterConfig ff;
  std::vector<int> object_classes;
  int person_class_id = 0;
  std::uint64_t frame_seq = 0;
  // single-stream guard: ловим случай, когда caller нарушает контракт
  // (одновременный push_frame на один pipeline из двух потоков).
  std::atomic<int> in_flight{0};

  // IoU + центроид + max_age: параметры из IntegraConfig (дефолты безопаснее для CCTV).
  IntegraPipeline() = default;
};

namespace {

// ---------------------------------------------------------------------------
// JSON builders.
// ---------------------------------------------------------------------------
std::string event_to_json(const integra::AlarmEvent& ev) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "{\"type\":\"" << json_esc(ev.type) << "\""
     << ",\"camera_id\":\"" << json_esc(ev.camera_id) << "\""
     << ",\"track_id\":" << ev.track_id
     << ",\"cls_id\":" << ev.cls_id
     << ",\"cls_name\":\"" << json_esc(ev.cls_name) << "\""
     << ",\"confidence\":" << ev.confidence
     << ",\"ts_wall_ms\":" << ev.ts_wall_ms
     << ",\"video_pos_ms\":" << ev.video_pos_ms
     << ",\"bbox\":[" << ev.bbox[0] << "," << ev.bbox[1] << "," << ev.bbox[2] << ","
     << ev.bbox[3] << "]"
     << ",\"note\":\"" << json_esc(ev.note) << "\""
     << "}";
  return os.str();
}

std::string tracks_array_json(const std::vector<integra::TrackSnapshot>& tracks) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "[";
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const auto& t = tracks[i];
    if (i) os << ",";
    os << "{\"id\":" << t.id
       << ",\"cls\":\"" << json_esc(t.cls) << "\""
       << ",\"state\":\"" << json_esc(t.state) << "\""
       << ",\"bbox\":[" << t.bbox[0] << "," << t.bbox[1] << "," << t.bbox[2] << ","
       << t.bbox[3] << "]"
       << ",\"conf\":" << t.conf
       << ",\"static_for_sec\":" << t.static_for_sec
       << ",\"unattended_for_sec\":" << t.unattended_for_sec
       << ",\"alarm\":" << (t.alarm ? "true" : "false") << "}";
  }
  os << "]";
  return os.str();
}

std::string persons_array_json(const std::vector<integra::Detection>& persons) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "[";
  for (std::size_t i = 0; i < persons.size(); ++i) {
    const auto& d = persons[i];
    if (i) os << ",";
    os << "{\"track_id\":" << d.track_id
       << ",\"confidence\":" << d.confidence
       << ",\"bbox\":[" << d.bbox.x1 << "," << d.bbox.y1 << "," << d.bbox.x2 << ","
       << d.bbox.y2 << "]}";
  }
  os << "]";
  return os.str();
}

std::string frame_result_json(std::uint64_t frame_id, double pts_ms,
                              std::size_t detections_n, std::size_t persons_n,
                              double infer_ms, double preprocess_ms,
                              double tracker_ms, double analyzer_ms,
                              const std::string& tracks_arr,
                              const std::string& persons_arr) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "{\"frame_id\":" << frame_id
     << ",\"pts_ms\":" << pts_ms
     << ",\"stats\":{"
     << "\"detections\":" << detections_n
     << ",\"persons\":" << persons_n
     << ",\"infer_ms\":" << infer_ms
     << ",\"preprocess_ms\":" << preprocess_ms
     << ",\"tracker_ms\":" << tracker_ms
     << ",\"analyzer_ms\":" << analyzer_ms
     << "}"
     << ",\"tracks\":" << tracks_arr
     << ",\"persons\":" << persons_arr
     << "}";
  return os.str();
}

// ---------------------------------------------------------------------------
// Маппинг IntegraConfig → внутренние параметры с разумными дефолтами.
// ---------------------------------------------------------------------------
void apply_defaults(integra::InferenceEngineConfig& ec,
                    integra::AnalyzerParams& ap,
                    integra::FrameFilterConfig& ff,
                    std::vector<int>& object_classes,
                    std::string& kind,
                    std::string& camera_id,
                    int& person_class_id,
                    const IntegraConfig& cfg) {
  // engine
  kind = (cfg.engine_kind && cfg.engine_kind[0]) ? cfg.engine_kind : "tensorrt";
  ec.model_path = (cfg.model_path && cfg.model_path[0]) ? cfg.model_path : "";
  ec.input_size = cfg.input_size > 0 ? cfg.input_size : 640;

  // postprocess
  ec.postprocess.conf_threshold =
      cfg.conf_threshold > 0.f ? cfg.conf_threshold : 0.25f;
  ec.postprocess.nms_iou_threshold =
      cfg.nms_iou_threshold > 0.f ? cfg.nms_iou_threshold : 0.45f;
  ec.postprocess.num_classes = cfg.num_classes > 0 ? cfg.num_classes : 80;
  ec.postprocess.num_anchors = cfg.num_anchors > 0 ? cfg.num_anchors : 8400;

  // class filter
  person_class_id = cfg.person_class_id;
  object_classes.clear();
  if (cfg.object_classes && cfg.object_classes_len > 0) {
    object_classes.assign(cfg.object_classes,
                          cfg.object_classes + cfg.object_classes_len);
    object_classes.erase(
        std::remove(object_classes.begin(), object_classes.end(), person_class_id),
        object_classes.end());
  }

  // frame filter
  ff.min_box_px = std::max(20, cfg.min_box_size_px > 0 ? cfg.min_box_size_px : 20);
  ff.use_regional_class_conf = (cfg.use_regional_class_conf != 0);
  if (ff.use_regional_class_conf) {
    ff.upper_region_y_ratio =
        cfg.upper_region_y_ratio > 1e-6f ? cfg.upper_region_y_ratio : 0.62f;
    ff.min_conf_upper = cfg.min_conf_upper > 1e-6f ? cfg.min_conf_upper : 0.22f;
    ff.min_conf_lower = cfg.min_conf_lower > 1e-6f ? cfg.min_conf_lower : 0.30f;
    ff.bottom_region_y_ratio =
        cfg.bottom_region_y_ratio > 1e-6f ? cfg.bottom_region_y_ratio : 0.88f;
    ff.min_conf_bottom = cfg.min_conf_bottom > 1e-6f ? cfg.min_conf_bottom : 0.26f;
    ff.conf_border_relax_px = std::max(0, cfg.border_relax_px);
    ff.min_conf_border = cfg.min_conf_border > 1e-6f ? cfg.min_conf_border : 0.20f;
    ff.person_min_conf_border =
        cfg.person_min_conf_border > 1e-6f ? cfg.person_min_conf_border : 0.18f;
  }
  ff.tune_from_postprocess(ec.postprocess.conf_threshold);
  // 0: иначе на широком угле CCTV почти всё у края — детекции исчезают с превью/трекера.
  ff.border_px = 0;
  // Чуть выше дефолта — режет совсем мелкие пятна; не душит узкие бутылки/кружки.
  ff.min_area_ratio = 0.00055f;
  ff.max_detections = 34;
  ff.ignore_det_norm_x1 = static_cast<float>(cfg.ignore_det_norm_x1);
  ff.ignore_det_norm_y1 = static_cast<float>(cfg.ignore_det_norm_y1);
  ff.ignore_det_norm_x2 = static_cast<float>(cfg.ignore_det_norm_x2);
  ff.ignore_det_norm_y2 = static_cast<float>(cfg.ignore_det_norm_y2);
  if (ff.ignore_det_norm_x2 <= ff.ignore_det_norm_x1 ||
      ff.ignore_det_norm_y2 <= ff.ignore_det_norm_y1) {
    ff.ignore_det_norm_x1 = 0.f;
    ff.ignore_det_norm_y1 = 0.f;
    ff.ignore_det_norm_x2 = 0.f;
    ff.ignore_det_norm_y2 = 0.f;
  }

  ff.class_min_conf_count = 0;
  if (cfg.class_min_conf_count > 0) {
    const int n = std::min(cfg.class_min_conf_count, INTEGRA_CLASS_MIN_CONF_MAX);
    ff.class_min_conf_count = n;
    for (int i = 0; i < n; ++i) {
      ff.class_min_conf_ids[i] = cfg.class_min_conf_class_ids[i];
      ff.class_min_conf_thresholds[i] = cfg.class_min_conf_thresholds[i];
    }
  }

  // analyzer
  ap.static_displacement_px =
      cfg.static_displacement_px > 0.0 ? cfg.static_displacement_px : 7.0;
  ap.static_window_sec = cfg.static_window_sec > 0.0 ? cfg.static_window_sec : 3.0;
  ap.abandon_time_sec = cfg.abandon_time_sec > 0.0 ? cfg.abandon_time_sec : 15.0;
  ap.owner_proximity_px =
      cfg.owner_proximity_px > 0.0 ? cfg.owner_proximity_px : 180.0;
  ap.owner_left_sec = cfg.owner_left_sec > 0.0 ? cfg.owner_left_sec : 5.0;
  ap.disappear_grace_sec =
      cfg.disappear_grace_sec > 0.0 ? cfg.disappear_grace_sec : 9.0;
  ap.min_object_area_px =
      cfg.min_object_area_px > 0.0 ? cfg.min_object_area_px : 100.0;
  ap.centroid_history_maxlen =
      cfg.centroid_history_maxlen > 0 ? cfg.centroid_history_maxlen : 72;
  ap.max_active_tracks =
      cfg.max_active_tracks > 0 ? cfg.max_active_tracks : 256;
  ap.person_class_id = person_class_id;

  // identity
  camera_id = (cfg.camera_id && cfg.camera_id[0]) ? cfg.camera_id : "main";
}

}  // namespace

// ===========================================================================
// FFI entry points.
// ===========================================================================

extern "C" {

INTEGRA_API IntegraPipeline* integra_pipeline_create(int abi_version,
                                                      const IntegraConfig* cfg) {
  if (abi_version != INTEGRA_FFI_ABI_VERSION) {
    std::cerr << "integra_ffi: ABI mismatch (caller=" << abi_version
              << ", lib=" << INTEGRA_FFI_ABI_VERSION << ")\n";
    return nullptr;
  }
  if (!cfg) {
    std::cerr << "integra_ffi: cfg == NULL\n";
    return nullptr;
  }

  auto p = std::make_unique<IntegraPipeline>();

  integra::InferenceEngineConfig ec;
  integra::AnalyzerParams ap;
  std::string kind;
  apply_defaults(ec, ap, p->ff, p->object_classes, kind, p->camera_id,
                 p->person_class_id, *cfg);

  const float trk_iou =
      cfg->tracker_iou_match_threshold > 1e-6f ? cfg->tracker_iou_match_threshold : 0.35f;
  const int trk_miss =
      cfg->tracker_max_missed_frames > 0 ? cfg->tracker_max_missed_frames : 10;
  const bool trk_soft = cfg->tracker_soft_centroid_match != 0;
  p->tracker = integra::IouTracker(trk_iou, trk_miss, trk_soft);

  // Создаём (или достаём из кэша) shared engine.
  p->shared = integra::make_shared_engine(kind, ec);
  if (!p->shared) {
    std::cerr << "integra_ffi: make_shared_engine('" << kind << "') failed\n";
    return nullptr;
  }

  // Лёгкий контекст для одного видеопотока.
  p->ctx = p->shared->create_stream_context();
  if (!p->ctx) {
    std::cerr << "integra_ffi: create_stream_context() failed\n";
    return nullptr;
  }

  p->analyzer = std::make_unique<integra::SceneAnalyzer>(ap);
  return p.release();
}

INTEGRA_API void integra_pipeline_destroy(IntegraPipeline* p) {
  if (!p) return;
  delete p;
}

INTEGRA_API int integra_pipeline_push_frame(IntegraPipeline* p,
                                             const uint8_t* bgr_data,
                                             int width,
                                             int height,
                                             int64_t pts_ms,
                                             IntegraEventCb cb,
                                             void* user_data) {
  if (!p) return -1;
  if (!bgr_data || width <= 0 || height <= 0) return -2;
  if (!p->ctx) return -4;

  // Проверка single-stream контракта (диагностика, без блокировки).
  if (p->in_flight.fetch_add(1, std::memory_order_acq_rel) != 0) {
    std::cerr << "integra_ffi: WARN concurrent push_frame on one pipeline "
                 "(caller violates contract)\n";
  }

  const auto t_pre0 = std::chrono::steady_clock::now();

  integra::StreamFrameInput sin{};
  sin.bgr = bgr_data;
  sin.width = width;
  sin.height = height;
  sin.row_stride_bytes = width * 3;
  sin.on_device = false;
  sin.cuda_stream = nullptr;

  integra::DetectionBatch batch;
  const bool ok = p->ctx->infer(sin, batch);
  const auto t_pre1 = std::chrono::steady_clock::now();
  if (!ok) {
    p->in_flight.fetch_sub(1, std::memory_order_acq_rel);
    return -3;
  }

  // Class whitelist → anti-noise filter.
  std::vector<integra::Detection> filtered;
  filtered.reserve(batch.items.size());
  for (const auto& d : batch.items) {
    if (integra::is_relevant_class(d.class_id, p->person_class_id,
                                    p->object_classes)) {
      filtered.push_back(d);
    }
  }
  filtered = integra::apply_frame_filter(filtered, width, height,
                                          p->person_class_id, p->ff);
  // IoU-дедуп: высокий порог — не склеивать две соседние канистры; затем NMS по bottle (39)
  // только при сильном перекрытии (дубли модели на одной ёмкости).
  integra::merge_same_class_objects_only(filtered, p->person_class_id, 0.48f);
  integra::merge_luggage_cross_class_by_iou(filtered, p->person_class_id, 0.38f);
  integra::merge_tableware_cross_iou(filtered, p->person_class_id, 0.25f);
  integra::suppress_duplicate_bottles_by_iou(filtered, p->person_class_id, 0.40f);

  const auto t_post = std::chrono::steady_clock::now();

  // Tracker.
  p->tracker.update(filtered, width, height);
  const auto t_track = std::chrono::steady_clock::now();

  // Split persons / objects.
  std::vector<integra::Detection> persons;
  std::vector<integra::Detection> objects;
  persons.reserve(filtered.size());
  objects.reserve(filtered.size());
  for (const auto& d : filtered) {
    if (d.class_id == p->person_class_id) {
      persons.push_back(d);
    } else {
      objects.push_back(d);
    }
  }

  // Analyzer (FSM) → events.
  const double ts = now_wall_sec();
  const double pts_sec = static_cast<double>(pts_ms);  // pts_ms keeps ms; analyzer wants ms-context
  const auto t_anal0 = std::chrono::steady_clock::now();
  std::vector<integra::AlarmEvent> events = p->analyzer->ingest(
      ts, pts_sec, p->camera_id, objects, persons, width, height);
  const auto t_anal1 = std::chrono::steady_clock::now();
  std::vector<integra::TrackSnapshot> tracks = p->analyzer->tracks_snapshot(ts);

  ++p->frame_seq;

  // Эмитим события.
  if (cb) {
    for (const auto& ev : events) {
      const std::string js = event_to_json(ev);
      cb("event", js.c_str(), user_data);
    }

    const double infer_ms = batch.inference_ms;
    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count() - infer_ms;
    const double tracker_ms =
        std::chrono::duration<double, std::milli>(t_track - t_post).count();
    const double analyzer_ms =
        std::chrono::duration<double, std::milli>(t_anal1 - t_anal0).count();

    const std::string tracks_arr = tracks_array_json(tracks);
    const std::string persons_arr = persons_array_json(persons);
    const std::string fr = frame_result_json(
        p->frame_seq, pts_sec, objects.size(), persons.size(),
        infer_ms, std::max(0.0, preprocess_ms), tracker_ms, analyzer_ms,
        tracks_arr, persons_arr);
    cb("frame_result", fr.c_str(), user_data);
  }

  p->in_flight.fetch_sub(1, std::memory_order_acq_rel);
  return 0;
}

INTEGRA_API void integra_pipeline_reset(IntegraPipeline* p) {
  if (!p) return;
  p->tracker.reset();
  if (p->analyzer) p->analyzer->reset();
  p->frame_seq = 0;
}

INTEGRA_API const char* integra_ffi_version(void) {
  return "integra-ffi 0.1.0";
}

}  // extern "C"
