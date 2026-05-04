#include "integra/pipeline.hpp"
#include "integra/yolo_postprocess.hpp"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>

static void usage() {
  std::cerr
      << "integra-pipeline — нативный контур (OpenCV + CUDA optional)\n"
      << "  --video PATH           входной файл\n"
      << "  --alarm-sink H:P       TCP JSON-lines → integra-alarmd (напр. 127.0.0.1:9090)\n"
      << "  --camera-id ID         идентификатор камеры в событии\n"
      << "  --target-fps N         ограничение скорости чтения\n"
      << "  --demo-alarm           каждые ~300 кадров слать demo_ping (проверка цепочки)\n"
      << "  --engine KIND          stub|onnx|tensorrt (onnx: INTEGRA_WITH_ONNXRUNTIME; tensorrt: "
         "INTEGRA_WITH_TENSORRT + .engine)\n"
      << "  --model PATH           .onnx (onnx) или .engine (tensorrt)\n"
      << "  --imgsz N              квадрат входа (0 = без resize, по умолчанию 640)\n"
      << "  --synth-detect         если модель пустая — подставить bbox «bottle» (тест FSM)\n"
      << "  --person-class ID      COCO person = 0\n"
      << "  --conf F               порог уверенности YOLO (по умолчанию 0.25)\n"
      << "  --nms-iou F            IoU для NMS (по умолчанию 0.45)\n"
      << "  --num-classes N        число классов выхода (по умолчанию 80)\n"
      << "  --num-anchors N        якоря в выходе (640²→обычно 8400)\n"
      << "  --selftest-yolo        самопроверка decode+NMS и выход\n"
      << "  --stats                раз в ~2 с: FPS и средний infer (мс) в stderr\n"
      << "  --max-frames N         обработать не более N кадров (0 = весь файл)\n";
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--selftest-yolo")) {
      return integra::selftest_yolo_postprocess() ? 0 : 2;
    }
  }

  integra::PipelineConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--video") && i + 1 < argc) {
      cfg.video_path = argv[++i];
    } else if (!std::strcmp(argv[i], "--alarm-sink") && i + 1 < argc) {
      cfg.alarm_sink = argv[++i];
    } else if (!std::strcmp(argv[i], "--camera-id") && i + 1 < argc) {
      cfg.camera_id = argv[++i];
    } else if (!std::strcmp(argv[i], "--target-fps") && i + 1 < argc) {
      cfg.target_fps = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--demo-alarm")) {
      cfg.demo_alarm = true;
    } else if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) {
      cfg.engine_kind = argv[++i];
    } else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) {
      cfg.model_path = argv[++i];
    } else if (!std::strcmp(argv[i], "--imgsz") && i + 1 < argc) {
      cfg.inference_input_size = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--synth-detect")) {
      cfg.synth_detect = true;
    } else if (!std::strcmp(argv[i], "--person-class") && i + 1 < argc) {
      cfg.analyzer.person_class_id = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--conf") && i + 1 < argc) {
      cfg.postprocess.conf_threshold = static_cast<float>(std::atof(argv[++i]));
    } else if (!std::strcmp(argv[i], "--nms-iou") && i + 1 < argc) {
      cfg.postprocess.nms_iou_threshold = static_cast<float>(std::atof(argv[++i]));
    } else if (!std::strcmp(argv[i], "--num-classes") && i + 1 < argc) {
      cfg.postprocess.num_classes = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--num-anchors") && i + 1 < argc) {
      cfg.postprocess.num_anchors = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--stats")) {
      cfg.stats = true;
    } else if (!std::strcmp(argv[i], "--max-frames") && i + 1 < argc) {
      cfg.max_frames = static_cast<std::uint64_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      usage();
      return 0;
    }
  }
  if (cfg.video_path.empty()) {
    usage();
    return 1;
  }
  if (const char* e = std::getenv("INTEGRA_DEMO_ALARM")) {
    if (e[0] != '0' && e[0] != '\0') {
      cfg.demo_alarm = true;
    }
  }
  return integra::run_pipeline(cfg);
}
