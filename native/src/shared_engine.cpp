// Универсальная фабрика ISharedEngine.
//
// Для kind="tensorrt" делегирует в make_shared_trt_engine (tensorrt_engine.cpp),
// который реализует настоящий пул контекстов поверх одного TRT engine.
//
// Для kind in {"opencv","onnx","stub"} — оборачивает legacy IInferenceEngine
// в LegacySharedEngine: один движок, защищённый мьютексом, lightweight
// "контексты" сериализуют вызовы. Это нужно чтобы Rust FFI (шаг 4) мог
// работать через единый ISharedEngine API независимо от backend'а.
//
// Препроцесс для shared opencv/onnx/stub пути выполняется на стороне
// LegacyStreamContext: каждый контекст держит свой GpuLetterboxPrep,
// чтобы не делить device-буферы между потоками.

#include "integra/gpu_preprocess.hpp"
#include "integra/inference_engine.hpp"
#include "integra/yolo_postprocess.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace integra {

#if INTEGRA_HAS_TENSORRT
// Объявлено в tensorrt_engine.cpp
std::shared_ptr<ISharedEngine> make_shared_trt_engine(const InferenceEngineConfig& cfg);
#endif

namespace {

// ---------------------------------------------------------------------------
// LegacySharedEngine — обёртка над одним IInferenceEngine.
// Не предоставляет настоящего multi-stream параллелизма; вызовы infer()
// сериализуются мьютексом. Для opencv/onnx/stub это нормально — они либо
// CPU-bound, либо OpenCV DNN c CUDA backend (внутри сам управляет stream'ом).
// ---------------------------------------------------------------------------
class LegacySharedEngine final : public ISharedEngine {
 public:
  LegacySharedEngine(std::string kind, InferenceEngineConfig cfg,
                     std::unique_ptr<IInferenceEngine> engine)
      : kind_(std::move(kind)), cfg_(std::move(cfg)), engine_(std::move(engine)) {}

  std::unique_ptr<IStreamContext> create_stream_context() override;
  bool is_real() const override { return engine_ && !engine_->is_stub(); }
  int input_size() const override { return cfg_.input_size; }
  const PostprocessParams& postprocess() const override { return cfg_.postprocess; }
  const std::string& kind() const override { return kind_; }

  // Сериализованный вызов в нижележащий движок.
  bool locked_infer(const InferenceInput& in, DetectionBatch& out) {
    std::lock_guard<std::mutex> lk(mu_);
    return engine_->infer(in, out);
  }

 private:
  std::string kind_;
  InferenceEngineConfig cfg_;
  std::unique_ptr<IInferenceEngine> engine_;
  std::mutex mu_;
};

// ---------------------------------------------------------------------------
// LegacyStreamContext — лёгкий handle поверх LegacySharedEngine.
// Содержит свой препроцессор (GpuLetterboxPrep) и resize-буфер; вызов
// engine->infer уходит под мьютекс shared engine'а.
// ---------------------------------------------------------------------------
class LegacyStreamContext final : public IStreamContext {
 public:
  explicit LegacyStreamContext(std::shared_ptr<LegacySharedEngine> eng)
      : engine_(std::move(eng)) {}

  bool infer(const StreamFrameInput& in, DetectionBatch& out) override {
    out.items.clear();
    out.inference_ms = 0.f;
    if (!engine_ || in.bgr == nullptr || in.width <= 0 || in.height <= 0) {
      return false;
    }
    if (in.on_device) {
      std::cerr << "integra: LegacyStreamContext::infer(on_device=true) не поддержан\n";
      return false;
    }

    const int trg = engine_->input_size() > 0 ? engine_->input_size() : in.width;
    const int row_stride =
        in.row_stride_bytes > 0 ? in.row_stride_bytes : in.width * 3;

    cv::Mat src_view(in.height, in.width, CV_8UC3, const_cast<void*>(in.bgr),
                     static_cast<std::size_t>(row_stride));
    if (resize_buf_.cols != trg || resize_buf_.rows != trg) {
      resize_buf_ = cv::Mat(trg, trg, CV_8UC3);
    }
    if (in.width == trg && in.height == trg) {
      src_view.copyTo(resize_buf_);
    } else {
      cv::resize(src_view, resize_buf_, cv::Size(trg, trg), 0, 0, cv::INTER_LINEAR);
    }

    InferenceInput vin;
#if INTEGRA_HAS_CUDA
    int pw = 0, ph = 0;
    if (!prep_.upload_and_preprocess(resize_buf_, pw, ph)) {
      std::cerr << "integra: GPU preprocess (shared legacy) failed\n";
      return false;
    }
    vin.nchw = prep_.device_nchw();
    vin.width = pw;
    vin.height = ph;
    vin.on_device = true;
    vin.cuda_stream = prep_.cuda_stream();
#else
    if (!prep_.upload_and_preprocess(resize_buf_, cpu_blob_)) {
      std::cerr << "integra: CPU preprocess (shared legacy) failed\n";
      return false;
    }
    vin.nchw = cpu_blob_.data();
    vin.width = resize_buf_.cols;
    vin.height = resize_buf_.rows;
    vin.on_device = false;
    vin.cuda_stream = nullptr;
#endif

    if (!engine_->locked_infer(vin, out)) {
      return false;
    }

    // Маппинг bbox обратно в координаты исходного кадра.
    const float sx = static_cast<float>(in.width) / static_cast<float>(trg);
    const float sy = static_cast<float>(in.height) / static_cast<float>(trg);
    for (auto& d : out.items) {
      d.bbox.x1 *= sx;
      d.bbox.x2 *= sx;
      d.bbox.y1 *= sy;
      d.bbox.y2 *= sy;
    }
    return true;
  }

  bool infer_nchw(const InferenceInput& in, DetectionBatch& out) override {
    if (!engine_) return false;
    return engine_->locked_infer(in, out);
  }

 private:
  std::shared_ptr<LegacySharedEngine> engine_;
  GpuLetterboxPrep prep_;
  cv::Mat resize_buf_;
#if !INTEGRA_HAS_CUDA
  std::vector<float> cpu_blob_;
#endif
};

std::unique_ptr<IStreamContext> LegacySharedEngine::create_stream_context() {
  // shared_from_this недоступен из-за приватного наследования shared_ptr API:
  // вместо этого захватываем shared_ptr через явное make_shared в фабрике.
  // Здесь восстанавливаем shared_ptr на себя через aliasing-конструктор:
  // не получится без enable_shared_from_this — поэтому фабрика хранит
  // owning shared_ptr и передаёт его сюда явно.
  //
  // Решение: фабрика создаёт shared_ptr<LegacySharedEngine>, а сам класс
  // не наследует enable_shared_from_this (см. make_shared_engine ниже —
  // там используется обёртка с явным shared_ptr).
  std::cerr << "integra: LegacySharedEngine::create_stream_context() called without owner\n";
  return nullptr;
}

// Перегружаемая «настоящая» фабрика контекста с явно переданным owner.
std::unique_ptr<IStreamContext> create_legacy_stream_context(
    std::shared_ptr<LegacySharedEngine> eng) {
  return std::make_unique<LegacyStreamContext>(std::move(eng));
}

// ---------------------------------------------------------------------------
// Обёртка: делегирует ISharedEngine, но в create_stream_context передаёт
// shared_ptr<LegacySharedEngine> через захваченный self.
// ---------------------------------------------------------------------------
class LegacySharedEngineHolder final : public ISharedEngine {
 public:
  explicit LegacySharedEngineHolder(std::shared_ptr<LegacySharedEngine> inner)
      : inner_(std::move(inner)) {}

  std::unique_ptr<IStreamContext> create_stream_context() override {
    return create_legacy_stream_context(inner_);
  }
  bool is_real() const override { return inner_->is_real(); }
  int input_size() const override { return inner_->input_size(); }
  const PostprocessParams& postprocess() const override { return inner_->postprocess(); }
  const std::string& kind() const override { return inner_->kind(); }

 private:
  std::shared_ptr<LegacySharedEngine> inner_;
};

}  // namespace

std::shared_ptr<ISharedEngine> make_shared_engine(const std::string& kind,
                                                   const InferenceEngineConfig& cfg) {
  if (kind == "tensorrt") {
#if INTEGRA_HAS_TENSORRT
    return make_shared_trt_engine(cfg);
#else
    std::cerr << "integra: SharedEngine kind='tensorrt' недоступен (соберите с "
                 "-DINTEGRA_WITH_TENSORRT=ON)\n";
    return nullptr;
#endif
  }
  auto legacy = make_inference_engine(kind);
  if (!legacy) {
    std::cerr << "integra: SharedEngine: неизвестный kind='" << kind << "'\n";
    return nullptr;
  }
  if (!legacy->init(cfg)) {
    std::cerr << "integra: SharedEngine: init('" << kind << "') failed\n";
    return nullptr;
  }
  auto inner = std::make_shared<LegacySharedEngine>(kind, cfg, std::move(legacy));
  return std::make_shared<LegacySharedEngineHolder>(inner);
}

}  // namespace integra
