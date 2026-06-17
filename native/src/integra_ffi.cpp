// integra_ffi.cpp — реализация C ABI поверх integra_core.
//
// Два независимых контура восприятия (по ТЗ):
//
//   bgr → IStreamContext::infer            (YOLO по всему кадру)
//       ├─[M2 class-based]  люди cls=0 → apply_frame_filter → ByteTracker (устойчивые ID)
//       └─[M3 class-agnostic] FrameDiffDetector → регионы-кандидаты (class_id=-1)
//                              → подавление пересечений с людьми → IouTracker
//       → SceneAnalyzer::ingest            (поведенческая FSM: person_interaction /
//                                           object_left / object_unattended /
//                                           object_removed / object_missing)
//       → callback("event", ...) × N
//       → callback("frame_result", ...)    (tracks + persons + stats)
//
// JSON формируется руками через ostringstream (нет внешних зависимостей).

#include "integra/integra_ffi.h"

#include "integra/byte_track.hpp"
#include "integra/frame_diff_detector.hpp"
#include "integra/frame_filter.hpp"
#include "integra/geom.hpp"
#include "integra/inference_engine.hpp"
#include "integra/iou_tracker.hpp"
#include "integra/scene_analyzer.hpp"
#include "integra/types.hpp"
#include "integra/yolo_postprocess.hpp"

#include <opencv2/core.hpp>

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
  // M2 (class-based): трекер людей — настоящий ByteTrack, устойчивые ID только у cls=0.
  std::unique_ptr<integra::ByteTracker> person_tracker;
  // M3 (class-agnostic): трекер регионов-кандидатов сцены (объекты без классов YOLO).
  integra::IouTracker object_tracker;
  std::unique_ptr<integra::SceneAnalyzer> analyzer;
  std::unique_ptr<integra::FrameDiffDetector> diff_detector;
  integra::FrameFilterConfig ff;
  std::vector<int> object_classes;
  int person_class_id = 0;
  // Параметры контура class-agnostic объектов (M3).
  bool object_candidates_enabled = true;
  float diff_pixel_threshold = 20.f;
  float diff_gradient_threshold = 15.f;
  int diff_min_region_area_px = 100;
  std::uint64_t frame_seq = 0;
  // single-stream guard: ловим случай, когда caller нарушает контракт
  // (одновременный push_frame на один pipeline из двух потоков).
  std::atomic<int> in_flight{0};

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
                    integra::ByteTrackParams& bt,
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
  // Контур class-agnostic объектов живёт в integra_ffi (один FrameDiffDetector);
  // внутренний детектор SceneAnalyzer выключаем, чтобы не дублировать diff.
  ap.use_frame_diff_detector = false;
  ap.track_only_persons = (cfg.track_only_persons != 0);

  // ByteTrack (контур людей)
  bt.track_high_thresh =
      cfg.bytetrack_high_thresh > 0.f ? cfg.bytetrack_high_thresh : 0.50f;
  bt.track_low_thresh =
      cfg.bytetrack_low_thresh > 0.f ? cfg.bytetrack_low_thresh : 0.10f;
  bt.new_track_thresh =
      cfg.bytetrack_new_thresh > 0.f ? cfg.bytetrack_new_thresh : 0.60f;
  bt.match_thresh =
      cfg.bytetrack_match_thresh > 0.f ? cfg.bytetrack_match_thresh : 0.80f;
  bt.track_buffer = cfg.bytetrack_buffer > 0 ? cfg.bytetrack_buffer : 30;
  bt.frame_rate = cfg.bytetrack_frame_rate > 0.f ? cfg.bytetrack_frame_rate : 30.f;

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
  integra::ByteTrackParams bt;
  std::string kind;
  apply_defaults(ec, ap, bt, p->ff, p->object_classes, kind, p->camera_id,
                 p->person_class_id, *cfg);

  // M2: ByteTrack для контура людей (устойчивые ID).
  p->person_tracker = std::make_unique<integra::ByteTracker>(bt);

  // M3: трекер class-agnostic регионов сцены (IoU + центроид + max_age).
  const float trk_iou =
      cfg->tracker_iou_match_threshold > 1e-6f ? cfg->tracker_iou_match_threshold : 0.35f;
  const int trk_miss =
      cfg->tracker_max_missed_frames > 0 ? cfg->tracker_max_missed_frames : 10;
  const bool trk_soft = cfg->tracker_soft_centroid_match != 0;
  p->object_tracker = integra::IouTracker(trk_iou, trk_miss, trk_soft);

  // M3 параметры источника кандидатов (FrameDiffDetector).
  p->object_candidates_enabled = (cfg->object_candidates_enabled != 0);
  p->diff_pixel_threshold =
      cfg->frame_diff_pixel_threshold > 0.f ? cfg->frame_diff_pixel_threshold : 20.f;
  p->diff_gradient_threshold =
      cfg->frame_diff_gradient_threshold > 0.f ? cfg->frame_diff_gradient_threshold : 15.f;
  p->diff_min_region_area_px =
      cfg->frame_diff_min_region_area_px > 0 ? cfg->frame_diff_min_region_area_px : 100;
  const int diff_buf =
      cfg->frame_diff_buffer_size > 0 ? cfg->frame_diff_buffer_size : 10;

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

  // Источник class-agnostic кандидатов (M3): единственный FrameDiffDetector.
  if (p->object_candidates_enabled) {
    p->diff_detector = std::make_unique<integra::FrameDiffDetector>(diff_buf);
  }

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
  
  // Создаём cv::Mat обёртку для BGR данных один раз (без копирования)
  cv::Mat bgr_mat(height, width, CV_8UC3, const_cast<uint8_t*>(bgr_data), width * 3);

  integra::DetectionBatch batch;
  const bool ok = p->ctx->infer(sin, batch);
  const auto t_pre1 = std::chrono::steady_clock::now();
  if (!ok) {
    p->in_flight.fetch_sub(1, std::memory_order_acq_rel);
    return -3;
  }

  // ===== Два независимых контура восприятия (ТЗ) =====

  // M2 (class-based): люди из YOLO cls=0 → фильтр → ByteTrack (устойчивые ID).
  std::vector<integra::Detection> persons;
  for (const auto& d : batch.items) {
    if (d.class_id == p->person_class_id) {
      persons.push_back(d);
    }
  }
  persons = integra::apply_frame_filter(persons, width, height,
                                         p->person_class_id, p->ff);
  p->person_tracker->update(persons, width, height);

  const auto t_post = std::chrono::steady_clock::now();

  // M3 (class-agnostic): регионы-кандидаты из FrameDiffDetector. Имя YOLO-класса
  // не присваивается (class_id = -1) — объект определяется поведением, не классом.
  std::vector<integra::Detection> objects;
  if (p->diff_detector) {
    auto changed_regions = p->diff_detector->process_frame(
        bgr_mat, p->diff_pixel_threshold, p->diff_gradient_threshold,
        p->diff_min_region_area_px);
    objects.reserve(changed_regions.size());
    for (const auto& region : changed_regions) {
      // Подавляем регионы, перекрытые людьми: движущийся человек сам порождает
      // передний план, но это не объект сцены (его ведёт контур M2).
      // Условие: заметный IoU ИЛИ регион в основном лежит внутри слегка
      // расширенного бокса человека (motion-блобы у краёв силуэта).
      bool on_person = false;
      for (const auto& per : persons) {
        const float pw = per.bbox.x2 - per.bbox.x1;
        const float ph = per.bbox.y2 - per.bbox.y1;
        const float mx = 0.15f * pw;
        const float my = 0.15f * ph;
        const float ix1 = std::max(region.bbox.x1, per.bbox.x1 - mx);
        const float iy1 = std::max(region.bbox.y1, per.bbox.y1 - my);
        const float ix2 = std::min(region.bbox.x2, per.bbox.x2 + mx);
        const float iy2 = std::min(region.bbox.y2, per.bbox.y2 + my);
        const float inter =
            std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
        const float rarea =
            std::max(1.f, (region.bbox.x2 - region.bbox.x1) *
                              (region.bbox.y2 - region.bbox.y1));
        if (integra::iou_xyxy(per.bbox, region.bbox) > 0.20f ||
            inter / rarea > 0.35f) {
          on_person = true;
          break;
        }
      }
      if (on_person) {
        continue;
      }
      integra::Detection det;
      det.bbox = region.bbox;
      det.confidence = std::min(1.f, region.combined_score / 255.f);
      det.class_id = -1;  // class-agnostic
      det.cls_name = "object";
      objects.push_back(det);
    }
    p->object_tracker.update(objects, width, height);
  }
  const auto t_track = std::chrono::steady_clock::now();

  // M4: поведенческая FSM. persons и objects подаются раздельно (общего трекинга нет).
  const double ts = now_wall_sec();
  const double pts_sec = static_cast<double>(pts_ms);
  const auto t_anal0 = std::chrono::steady_clock::now();
  std::vector<integra::AlarmEvent> events = p->analyzer->ingest(
      ts, pts_sec, p->camera_id, objects, persons, width, height, bgr_mat);
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
  if (p->person_tracker) p->person_tracker->reset();
  p->object_tracker.reset();
  if (p->diff_detector) p->diff_detector->reset();
  if (p->analyzer) p->analyzer->reset();
  p->frame_seq = 0;
}

INTEGRA_API const char* integra_ffi_version(void) {
  return "integra-ffi 0.1.0";
}

}  // extern "C"
