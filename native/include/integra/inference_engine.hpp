#pragma once

#include "integra/types.hpp"
#include "integra/yolo_postprocess.hpp"

#include <memory>
#include <string>

namespace integra {

// ---------------------------------------------------------------------------
// Legacy API (используется integra-pipeline / integra-analyticsd / parity).
// Один engine = один поток инференса. Сохраняется без изменений для обратной
// совместимости; новые места должны использовать ISharedEngine + IStreamContext.
// ---------------------------------------------------------------------------

/// Вход в движок: NCHW float [1,3,H,W] (как после `integra_cuda_bgr_to_nchw_float`).
struct InferenceInput {
  const float* nchw = nullptr;  // host или device — см. on_device
  int width = 0;
  int height = 0;
  bool on_device = false;       // true: CUDA pointer (TensorRT); false: CPU
  void* cuda_stream = nullptr;  // cudaStream_t при on_device; иначе nullptr
};

struct InferenceEngineConfig {
  std::string model_path;
  int input_size = 640;
  PostprocessParams postprocess;
};

/// Плагин к TensorRT / ONNX Runtime / OpenCV DNN. Реализация по умолчанию — stub.
class IInferenceEngine {
 public:
  virtual ~IInferenceEngine() = default;

  /// model_path: .engine / .onnx / пусто для stub. input_size: квадрат 640, 960, …
  virtual bool init(const InferenceEngineConfig& cfg) = 0;
  virtual bool infer(const InferenceInput& in, DetectionBatch& out) = 0;
  /// Признак заглушки (stub backend). Используется для fail-fast в real-native режиме.
  virtual bool is_stub() const { return false; }
};

/// Фабрика legacy движков: kind ∈ {"stub", "opencv", "onnx", "tensorrt"}.
std::unique_ptr<IInferenceEngine> make_inference_engine(const std::string& kind);

// ---------------------------------------------------------------------------
// Shared engine + per-stream context API (используется integra_ffi + Rust).
// Один тяжёлый ресурс (TRT engine + weights в VRAM) — общий на N потоков;
// у каждого потока свой лёгкий ExecutionContext + cudaStream + I/O буферы.
//
// Реализации:
//   • SharedTRTEngine (tensorrt_engine.cpp)  — настоящий пул контекстов;
//   • LegacySharedEngine (shared_engine.cpp) — обёртка над IInferenceEngine
//     для opencv/onnx/stub (один движок под mutex, без true parallel).
// ---------------------------------------------------------------------------

/// Кадр на вход. На текущем этапе принимаем BGR uint8 host или device.
/// row_stride_bytes = 0 трактуется как плотный (width * 3).
struct StreamFrameInput {
  const void* bgr = nullptr;
  int width = 0;
  int height = 0;
  int row_stride_bytes = 0;
  bool on_device = false;       // device pointer (CUDA) — для in-process FFI пути
  void* cuda_stream = nullptr;  // если задан и on_device=true, источник копий — этот stream
};

/// Контекст одного видеопотока. Не shared между потоками — у каждого свой.
class IStreamContext {
 public:
  virtual ~IStreamContext() = default;

  /// Полный путь: resize → letterbox → BGR→NCHW → inference → decode + NMS.
  /// Синхронный по своему CUDA stream (внутри cudaStreamSynchronize).
  virtual bool infer(const StreamFrameInput& in, DetectionBatch& out) = 0;

  /// Запасной путь: вход уже подготовлен как NCHW (как у IInferenceEngine).
  /// Полезно когда препроцесс делается снаружи (например, в Rust pipeline).
  virtual bool infer_nchw(const InferenceInput& in, DetectionBatch& out) = 0;
};

/// Тяжёлый ресурс (движок + веса). Создаётся один на (kind, model_path).
class ISharedEngine {
 public:
  virtual ~ISharedEngine() = default;

  /// Создаёт лёгкий контекст для одного потока. nullptr при ошибке.
  virtual std::unique_ptr<IStreamContext> create_stream_context() = 0;

  /// Реальный движок (false для stub).
  virtual bool is_real() const = 0;

  /// Размер входа (640 / 960 / …); 0 = без resize, как есть.
  virtual int input_size() const = 0;

  /// Параметры постобработки (conf / iou / num_classes / num_anchors).
  virtual const PostprocessParams& postprocess() const = 0;

  /// Тип движка, как был запрошен: "tensorrt" | "opencv" | "onnx" | "stub".
  virtual const std::string& kind() const = 0;
};

/// Фабрика shared engine. Возвращает nullptr, если backend недоступен на сборке.
/// Для TensorRT кэш — по `cfg.model_path`: повторный вызов отдаёт тот же engine.
std::shared_ptr<ISharedEngine> make_shared_engine(const std::string& kind,
                                                   const InferenceEngineConfig& cfg);

}  // namespace integra
