#include "integra/pipeline.hpp"

#include "integra/alarm_sink.hpp"
#include "integra/geom.hpp"
#include "integra/gpu_preprocess.hpp"
#include "integra/inference_engine.hpp"
#include "integra/iou_tracker.hpp"
#include "integra/scene_analyzer.hpp"
#include "integra/video_source.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

/// Статический «бутылка» в координатах feed (640×640), если модель ничего не вернула.
static void apply_synth_bbox(int feed_w, int feed_h, DetectionBatch& dets) {
  if (!dets.items.empty()) {
    return;
  }
  Detection d;
  d.class_id = 39;
  d.cls_name = "bottle";
  d.confidence = 0.95f;
  const float side = static_cast<float>(std::min(feed_w, feed_h));
  const float m = 0.2f * side;
  const float cx = feed_w * 0.5f;
  const float cy = feed_h * 0.55f;
  d.bbox = {cx - m, cy - m, cx + m, cy + m};
  dets.items.push_back(d);
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

  IouTracker tracker;
  SceneAnalyzer analyzer(cfg.analyzer);

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

    if (cfg.inference_input_size > 0) {
      const int s = cfg.inference_input_size;
      cv::resize(bgr, feed, cv::Size(s, s), 0, 0, cv::INTER_LINEAR);
    } else {
      feed = bgr;
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

    if (cfg.synth_detect) {
      apply_synth_bbox(feed.cols, feed.rows, dets);
    }

    const float sx = static_cast<float>(bgr.cols) / static_cast<float>(std::max(1, feed.cols));
    const float sy = static_cast<float>(bgr.rows) / static_cast<float>(std::max(1, feed.rows));
    for (auto& d : dets.items) {
      d.bbox = scale_bbox_xyxy(d.bbox, sx, sy);
    }

    tracker.update(dets.items);

    std::vector<Detection> persons;
    std::vector<Detection> objects;
    persons.reserve(dets.items.size());
    objects.reserve(dets.items.size());
    for (const auto& d : dets.items) {
      if (d.class_id == cfg.analyzer.person_class_id) {
        persons.push_back(d);
      } else {
        objects.push_back(d);
      }
    }

    const double ts =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<AlarmEvent> alarm_evts =
        analyzer.ingest(ts, meta.pos_ms, cfg.camera_id, objects, persons);
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
