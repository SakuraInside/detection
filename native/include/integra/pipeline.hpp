#pragma once

#include "integra/scene_analyzer.hpp"
#include "integra/yolo_postprocess.hpp"

#include <cstdint>
#include <string>

namespace integra {

struct PipelineConfig {
  std::string video_path;
  std::string camera_id = "cam0";
  /// integra-alarmd или шлюз: "host:port"
  std::string alarm_sink;
  int target_fps = 0;  // 0 = полная скорость файла
  /// INTEGRA_DEMO_ALARM: периодически слать тестовую тревогу (проверка интеграции).
  bool demo_alarm = false;

  /// Движок: stub | tensorrt | onnx (tensorrt — пока заглушка; onnx — см. INTEGRA_WITH_ONNXRUNTIME).
  std::string engine_kind = "stub";
  /// Путь к .engine / .onnx; для stub может быть пустым.
  std::string model_path;
  /// Квадрат входа нейросети (640/896/…); 0 = без resize (полный кадр, тяжелее).
  int inference_input_size = 640;

  /// Подставить статический объект без модели (проверка трекера+FSM).
  bool synth_detect = false;

  /// Раз в ~2 с в stderr: FPS и среднее время инференса (мс).
  bool stats = false;

  /// Ограничение числа кадров (0 = до конца файла; удобно для бенчмарка).
  std::uint64_t max_frames = 0;

  AnalyzerParams analyzer;

  /// Пороги YOLO-постобработки (для будущего ONNX/TensorRT; stub игнорирует).
  PostprocessParams postprocess;
};

/// Главный цикл: декод → GPU/CPU NCHW → (заглушка детектора) → политика тревог → JSONL.
int run_pipeline(const PipelineConfig& cfg);

}  // namespace integra
