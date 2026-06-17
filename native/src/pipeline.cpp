#include "integra/pipeline.hpp"

#include "integra/alarm_sink.hpp"
#include "integra/byte_track.hpp"
#include "integra/frame_diff_detector.hpp"
#include "integra/geom.hpp"
#include "integra/gpu_preprocess.hpp"
#include "integra/inference_engine.hpp"
#include "integra/iou_tracker.hpp"
#include "integra/scene_analyzer.hpp"
#include "integra/video_source.hpp"
#include "integra/yolo_letterbox.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace integra {

namespace {

static void parse_alarm_addr(const std::string& spec, std::string& host, int& port) {
  auto pos = spec.rfind(':');
  if (pos == std::string::npos) {
    host = "127.0.0.1";
    port = 0;
    return;
  }
  host = spec.substr(0, pos);
  port = std::atoi(spec.substr(pos + 1).c_str());
}

}  // namespace

int run_pipeline(const PipelineConfig& cfg) {
  VideoFileSource src;
  if (!src.open(cfg.video_path)) {
    std::cerr << "integra: failed to open video: " << cfg.video_path << "\n";
    return 2;
  }

  GpuLetterboxPrep prep;
  AlarmJsonlSink alarms;
  std::string ah;
  int ap = 0;
  parse_alarm_addr(cfg.alarm_sink, ah, ap);
  if (ap > 0) {
    alarms.configure(ah, ap);
  }

  auto engine = make_inference_engine(cfg.engine_kind);
  if (!engine) {
    std::cerr << "integra: unknown --engine " << cfg.engine_kind << "\n";
    return 3;
  }
  InferenceEngineConfig ecfg;
  ecfg.model_path = cfg.model_path;
  ecfg.input_size = cfg.inference_input_size;
  ecfg.postprocess = cfg.postprocess;
  if (!engine->init(ecfg)) {
    std::cerr << "integra: inference engine init failed\n";
    return 3;
  }

  // M2 (class-based): люди — ByteTrack. M3 (class-agnostic): объекты — IoU + центроид.
  ByteTracker person_tracker;
  IouTracker object_tracker(0.35f, 10, true);
  SceneAnalyzer analyzer(cfg.analyzer);
  std::unique_ptr<FrameDiffDetector> diff_detector;
  if (cfg.analyzer.use_frame_diff_detector) {
    diff_detector =
        std::make_unique<FrameDiffDetector>(cfg.analyzer.frame_diff_buffer_size);
  }
  const int person_class_id = cfg.analyzer.person_class_id;

#if INTEGRA_HAS_CUDA
  int pw = 0, ph = 0;
#else
  std::vector<float> cpu_blob;
#endif

  using clock = std::chrono::steady_clock;
  auto next_tick = clock::now();
  auto last_stat = clock::now();
  std::uint64_t frames_at_stat = 0;
  double infer_ms_acc = 0.0;

  std::uint64_t frames = 0;
  cv::Mat bgr;
  cv::Mat feed;

  while (true) {
    FrameMeta meta{};
    if (!src.read_next(bgr, meta)) {
      break;
    }

    LetterboxMeta letter{};
    if (cfg.inference_input_size > 0) {
      const int s = cfg.inference_input_size;
      if (!yolo_letterbox_bgr(bgr, s, feed, &letter)) {
        std::cerr << "integra: yolo_letterbox_bgr failed\n";
        break;
      }
    } else {
      feed = bgr;
      letter.r = 1.f;
      letter.pad_left = letter.pad_top = 0;
    }

#if INTEGRA_HAS_CUDA
    if (!prep.upload_and_preprocess(feed, pw, ph)) {
      std::cerr << "integra: preprocess failed\n";
      break;
    }
#else
    if (!prep.upload_and_preprocess(feed, cpu_blob)) {
      std::cerr << "integra: preprocess failed\n";
      break;
    }
#endif

    DetectionBatch dets;
    InferenceInput vin;
#if INTEGRA_HAS_CUDA
    vin.nchw = prep.device_nchw();
    vin.width = pw;
    vin.height = ph;
    vin.on_device = true;
    vin.cuda_stream = prep.cuda_stream();
#else
    vin.nchw = cpu_blob.data();
    vin.width = feed.cols;
    vin.height = feed.rows;
    vin.on_device = false;
#endif
    if (!engine->infer(vin, dets)) {
      std::cerr << "integra: inference failed\n";
#if INTEGRA_HAS_CUDA
      prep.sync();
#endif
      break;
    }
    infer_ms_acc += static_cast<double>(dets.inference_ms);

    if (cfg.inference_input_size > 0) {
      yolo_unletterbox_dets(dets.items, letter, bgr.cols, bgr.rows);
    }

    // M2 (class-based): люди cls=0 → ByteTrack (устойчивые ID).
    std::vector<Detection> persons;
    for (const auto& d : dets.items) {
      if (d.class_id == person_class_id) {
        persons.push_back(d);
      }
    }
    person_tracker.update(persons, bgr.cols, bgr.rows);

    // M3 (class-agnostic): регионы-кандидаты из FrameDiffDetector (имя класса не присваивается).
    std::vector<Detection> objects;
    if (diff_detector) {
      auto regions = diff_detector->process_frame(
          bgr, cfg.analyzer.frame_diff_pixel_threshold,
          cfg.analyzer.frame_diff_gradient_threshold,
          cfg.analyzer.frame_diff_min_region_area_px);
      for (const auto& region : regions) {
        bool on_person = false;
        for (const auto& per : persons) {
          if (iou_xyxy(per.bbox, region.bbox) > 0.20f) {
            on_person = true;
            break;
          }
        }
        if (on_person) {
          continue;
        }
        Detection det;
        det.bbox = region.bbox;
        det.confidence = std::min(1.f, region.combined_score / 255.f);
        det.class_id = -1;  // class-agnostic
        det.cls_name = "object";
        objects.push_back(det);
      }
      object_tracker.update(objects, bgr.cols, bgr.rows);
    }

    const double ts =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<AlarmEvent> alarm_evts =
        analyzer.ingest(ts, meta.pos_ms, cfg.camera_id, objects, persons, bgr.cols, bgr.rows, bgr);
    if (alarms.is_configured()) {
      for (const auto& ev : alarm_evts) {
        alarms.send(ev);
      }
    }

    ++frames;

    if (cfg.stats) {
      const auto now = clock::now();
      const double dt = std::chrono::duration<double>(now - last_stat).count();
      if (dt >= 2.0 && frames > frames_at_stat) {
        const double df = static_cast<double>(frames - frames_at_stat);
        const double fps = df / std::max(1e-6, dt);
        const double avg_inf = infer_ms_acc / std::max(1.0, df);
        std::cerr << "integra: stats fps=" << fps << " infer_avg_ms=" << avg_inf << " frames=" << frames
                  << "\n";
        last_stat = now;
        frames_at_stat = frames;
        infer_ms_acc = 0.0;
      }
    }

    if (cfg.demo_alarm && alarms.is_configured() && frames >= 300u && (frames % 300u) == 0u) {
      AlarmEvent ev;
      ev.type = "demo_ping";
      ev.camera_id = cfg.camera_id;
      ev.track_id = -1;
      ev.cls_id = -1;
      ev.cls_name = "native_pipeline";
      ev.confidence = 1.0;
      ev.ts_wall_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
      ev.video_pos_ms = meta.pos_ms;
      alarms.send(ev);
    }

    if (cfg.target_fps > 0) {
      const double tf = static_cast<double>(cfg.target_fps);
      next_tick += std::chrono::duration_cast<clock::duration>(
          std::chrono::duration<double>{1.0 / tf});
      std::this_thread::sleep_until(next_tick);
    }

    if (cfg.max_frames > 0 && frames >= cfg.max_frames) {
      break;
    }
  }

  alarms.close();
  src.close();
  std::cout << "integra: done, frames=" << frames << "\n";
  return 0;
}

}  // namespace integra
