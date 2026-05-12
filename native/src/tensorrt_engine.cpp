// TensorRT backend.
//
// Архитектура:
//   SharedTRTEngine        — один на (model_path): IRuntime + ICudaEngine,
//                            кэш через weak_ptr (не утечёт, но переиспользуется).
//   TRTStreamContext       — на каждый видеопоток: свой cudaStream_t,
//                            свой IExecutionContext, свои cudaMalloc-буферы
//                            (input float NCHW + output float),
//                            pinned host output (cudaMallocHost).
//   TRTLegacyAdapter       — IInferenceEngine для обратной совместимости:
//                            создаёт один stream context из shared engine.
//
// Файл компилируется только при INTEGRA_HAS_TENSORRT=1.

#include "integra/inference_engine.hpp"
#include "integra/yolo_postprocess.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <NvInfer.h>
#include <NvInferVersion.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" cudaError_t integra_cuda_bgr_to_nchw_float(const unsigned char* d_bgr_hwc,
                                                     float* d_nchw,
                                                     int width,
                                                     int height,
                                                     cudaStream_t stream);

namespace integra {

namespace {

// ---------------------------------------------------------------------------
// Логгер TRT — печатает только warning+ (info/verbose молчит).
// ---------------------------------------------------------------------------
class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "tensorrt: " << msg << "\n";
    }
  }
};

static TrtLogger g_trt_logger;

static std::size_t volume_dims(const nvinfer1::Dims& d) {
  std::size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    const int di = d.d[i];
    if (di > 0) {
      v *= static_cast<std::size_t>(di);
    }
  }
  return v;
}

static void trt_destroy_engine(nvinfer1::ICudaEngine* e) {
  if (!e) return;
#if NV_TENSORRT_MAJOR >= 10
  delete e;
#else
  e->destroy();
#endif
}

static void trt_destroy_runtime(nvinfer1::IRuntime* r) {
  if (!r) return;
#if NV_TENSORRT_MAJOR >= 10
  delete r;
#else
  r->destroy();
#endif
}

static void trt_destroy_context(nvinfer1::IExecutionContext* c) {
  if (!c) return;
#if NV_TENSORRT_MAJOR >= 10
  delete c;
#else
  c->destroy();
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// SharedTRTEngine: тяжёлый ресурс. Создаётся один на model_path,
// деcериализует .engine и держит ICudaEngine. Контексты создаются дёшево.
// ---------------------------------------------------------------------------
class SharedTRTEngine final : public ISharedEngine,
                              public std::enable_shared_from_this<SharedTRTEngine> {
 public:
  SharedTRTEngine() = default;
  ~SharedTRTEngine() override {
    trt_destroy_engine(engine_);
    trt_destroy_runtime(runtime_);
  }

  SharedTRTEngine(const SharedTRTEngine&) = delete;
  SharedTRTEngine& operator=(const SharedTRTEngine&) = delete;

  bool init(const InferenceEngineConfig& cfg) {
    cfg_ = cfg;
    kind_ = "tensorrt";

    if (cfg.model_path.empty()) {
      std::cerr << "integra: --engine tensorrt требует непустой --model (путь к .engine)\n";
      return false;
    }

    std::ifstream f(cfg.model_path, std::ios::binary);
    if (!f) {
      std::cerr << "integra: не удалось открыть .engine: " << cfg.model_path << "\n";
      return false;
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty()) {
      std::cerr << "integra: пустой файл движка\n";
      return false;
    }

    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) {
      std::cerr << "integra: createInferRuntime failed\n";
      return false;
    }

    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) {
      std::cerr << "integra: deserializeCudaEngine failed\n";
      return false;
    }

    const int32_t nio = engine_->getNbIOTensors();
    if (nio < 2) {
      std::cerr << "integra: ожидается хотя бы вход и выход (I/O tensors >= 2)\n";
      return false;
    }

    for (int32_t i = 0; i < nio; ++i) {
      const char* nm = engine_->getIOTensorName(i);
      const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(nm);
      if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
        input_name_ = nm;
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_name_.empty()) {
        output_name_ = nm;
      }
    }

    if (input_name_.empty() || output_name_.empty()) {
      std::cerr << "integra: не найдены имена входного/выходного тензора\n";
      return false;
    }

    const nvinfer1::DataType itype = engine_->getTensorDataType(input_name_.c_str());
    const nvinfer1::DataType otype = engine_->getTensorDataType(output_name_.c_str());
    if (itype != nvinfer1::DataType::kFLOAT || otype != nvinfer1::DataType::kFLOAT) {
      std::cerr << "integra: пока поддерживается только FP32 I/O\n";
      return false;
    }

    return true;
  }

  // --- ISharedEngine -------------------------------------------------------
  std::unique_ptr<IStreamContext> create_stream_context() override;
  bool is_real() const override { return engine_ != nullptr; }
  int input_size() const override { return cfg_.input_size; }
  const PostprocessParams& postprocess() const override { return cfg_.postprocess; }
  const std::string& kind() const override { return kind_; }

  // --- доступ для TRTStreamContext (через shared_from_this) ----------------
  nvinfer1::ICudaEngine* engine() const { return engine_; }
  const std::string& input_name() const { return input_name_; }
  const std::string& output_name() const { return output_name_; }
  const InferenceEngineConfig& cfg() const { return cfg_; }

 private:
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  std::string input_name_;
  std::string output_name_;
  InferenceEngineConfig cfg_;
  std::string kind_;
};

// ---------------------------------------------------------------------------
// TRTStreamContext: per-thread state. Создаётся каждым видеопотоком.
// Содержит свой stream + IExecutionContext + cudaMalloc буферы +
// pinned host output buffer (для async D2H).
// ---------------------------------------------------------------------------
class TRTStreamContext final : public IStreamContext {
 public:
  explicit TRTStreamContext(std::shared_ptr<SharedTRTEngine> eng) : engine_(std::move(eng)) {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    ctx_ = engine_->engine()->createExecutionContext();
  }

  ~TRTStreamContext() override {
    trt_destroy_context(ctx_);
    if (d_input_) {
      cudaFree(d_input_);
    }
    if (d_output_) {
      cudaFree(d_output_);
    }
    if (d_bgr_) {
      cudaFree(d_bgr_);
    }
    if (h_output_pinned_) {
      cudaFreeHost(h_output_pinned_);
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  TRTStreamContext(const TRTStreamContext&) = delete;
  TRTStreamContext& operator=(const TRTStreamContext&) = delete;

  // -------------------------------------------------------------------------
  // Полный путь: BGR → (resize до input_size) → BGR→NCHW float (CUDA) →
  // TRT enqueue → async D2H в pinned → sync на своём stream → decode + NMS.
  // -------------------------------------------------------------------------
  bool infer(const StreamFrameInput& in, DetectionBatch& out) override {
    out.items.clear();
    out.inference_ms = 0.f;

    if (!ctx_ || !engine_ || in.bgr == nullptr || in.width <= 0 || in.height <= 0) {
      return false;
    }

    const int trg = engine_->input_size() > 0 ? engine_->input_size() : in.width;

    // 1. На вход TRT нужен квадратный BGR размера trg×trg. Resize делаем через
    //    OpenCV (CPU, но дешёвый по сравнению с TRT). Для high-FPS пути в будущем
    //    можно заменить на CUDA NPP resize, но это вне scope шага 1.
    //
    //    Чтобы не плодить лишних копий, если уже trg×trg — пропускаем resize.
    const int row_stride =
        in.row_stride_bytes > 0 ? in.row_stride_bytes : in.width * 3;

    cv::Mat src_view;
    if (in.on_device) {
      // device-вход для этого пути пока не поддерживаем (требуется resize на GPU);
      // обратная совместимость живёт через infer_nchw().
      std::cerr << "integra: TRTStreamContext::infer(on_device=true) пока не реализован\n";
      return false;
    } else {
      src_view = cv::Mat(in.height, in.width, CV_8UC3,
                         const_cast<void*>(in.bgr), static_cast<std::size_t>(row_stride));
    }

    const std::size_t need_bgr = static_cast<std::size_t>(trg) * trg * 3;
    if (resize_buf_.cols != trg || resize_buf_.rows != trg) {
      resize_buf_ = cv::Mat(trg, trg, CV_8UC3);
    }
    if (in.width == trg && in.height == trg) {
      src_view.copyTo(resize_buf_);
    } else {
      cv::resize(src_view, resize_buf_, cv::Size(trg, trg), 0, 0, cv::INTER_LINEAR);
    }

    // 2. cudaMalloc один раз — далее переиспользуем.
    if (cap_bgr_ < need_bgr) {
      if (d_bgr_) cudaFree(d_bgr_);
      if (cudaMalloc(&d_bgr_, need_bgr) != cudaSuccess) {
        std::cerr << "integra: cudaMalloc d_bgr failed\n";
        return false;
      }
      cap_bgr_ = need_bgr;
    }

    const std::size_t need_in = static_cast<std::size_t>(3) * trg * trg * sizeof(float);
    if (cap_input_ < need_in) {
      if (d_input_) cudaFree(d_input_);
      if (cudaMalloc(&d_input_, need_in) != cudaSuccess) {
        std::cerr << "integra: cudaMalloc d_input failed\n";
        return false;
      }
      cap_input_ = need_in;
    }

    // 3. H2D BGR в собственный stream.
    if (cudaMemcpyAsync(d_bgr_, resize_buf_.ptr<unsigned char>(),
                        need_bgr, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      std::cerr << "integra: cudaMemcpyAsync BGR H2D failed\n";
      return false;
    }

    // 4. Препроцесс на GPU: BGR HWC → NCHW float / 255.
    if (integra_cuda_bgr_to_nchw_float(static_cast<unsigned char*>(d_bgr_),
                                       static_cast<float*>(d_input_), trg, trg,
                                       stream_) != cudaSuccess) {
      std::cerr << "integra: BGR→NCHW kernel failed\n";
      return false;
    }

    // 5. Привязка тензоров к контексту + setInputShape (динамический батч-1).
    nvinfer1::Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = 1;
    in_dims.d[1] = 3;
    in_dims.d[2] = trg;
    in_dims.d[3] = trg;
    if (!ctx_->setInputShape(engine_->input_name().c_str(), in_dims)) {
      std::cerr << "integra: TRT setInputShape failed (проверьте профиль .engine)\n";
      return false;
    }

    const nvinfer1::Dims od = ctx_->getTensorShape(engine_->output_name().c_str());
    for (int i = 0; i < od.nbDims; ++i) {
      if (od.d[i] <= 0) {
        std::cerr << "integra: выходной тензор имеет неполную размерность после setInputShape\n";
        return false;
      }
    }
    if (od.nbDims != 3 || od.d[0] != 1) {
      std::cerr << "integra: ожидается выход rank 3 с batch 1\n";
      return false;
    }
    const std::size_t out_elems = volume_dims(od);
    const std::size_t need_out = out_elems * sizeof(float);
    if (cap_output_ < need_out) {
      if (d_output_) cudaFree(d_output_);
      if (cudaMalloc(&d_output_, need_out) != cudaSuccess) {
        std::cerr << "integra: cudaMalloc d_output failed\n";
        return false;
      }
      cap_output_ = need_out;
    }
    if (host_pinned_cap_ < need_out) {
      if (h_output_pinned_) cudaFreeHost(h_output_pinned_);
      if (cudaMallocHost(&h_output_pinned_, need_out) != cudaSuccess) {
        std::cerr << "integra: cudaMallocHost output failed\n";
        return false;
      }
      host_pinned_cap_ = need_out;
    }

    ctx_->setTensorAddress(engine_->input_name().c_str(), d_input_);
    ctx_->setTensorAddress(engine_->output_name().c_str(), d_output_);

    // 6. Async inference.
    auto t0 = std::chrono::steady_clock::now();
    if (!ctx_->enqueueV3(stream_)) {
      std::cerr << "integra: TRT enqueueV3 failed\n";
      return false;
    }

    // 7. Async D2H в pinned буфер.
    if (cudaMemcpyAsync(h_output_pinned_, d_output_, need_out,
                        cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
      std::cerr << "integra: cudaMemcpyAsync output D2H failed\n";
      return false;
    }

    // 8. Барьер только по СВОЕМУ stream — параллельные контексты не блокируются.
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      std::cerr << "integra: cudaStreamSynchronize failed\n";
      return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    out.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 9. Decode + NMS на CPU.
    const std::int64_t d1 = od.d[1];
    const std::int64_t d2 = od.d[2];
    const bool chw = yolo_output_channel_first(d1, d2);
    PostprocessParams pp = engine_->postprocess();
    std::vector<Detection> decoded;
    decode_yolov8_flat(static_cast<const float*>(h_output_pinned_),
                       static_cast<int>(d1), static_cast<int>(d2), chw, pp,
                       static_cast<float>(trg), static_cast<float>(trg),
                       transpose_scratch_, decoded);
    nms_greedy_xyxy(decoded, pp.nms_iou_threshold);

    // 10. Маппинг bbox обратно из координат TRT-входа в координаты исходного кадра.
    const float sx = static_cast<float>(in.width) / static_cast<float>(trg);
    const float sy = static_cast<float>(in.height) / static_cast<float>(trg);
    for (auto& d : decoded) {
      d.bbox.x1 *= sx;
      d.bbox.x2 *= sx;
      d.bbox.y1 *= sy;
      d.bbox.y2 *= sy;
    }
    out.items = std::move(decoded);
    return true;
  }

  // -------------------------------------------------------------------------
  // Запасной путь: NCHW уже готов снаружи (host или device).
  // Сохраняет совместимость со старым IInferenceEngine API.
  // -------------------------------------------------------------------------
  bool infer_nchw(const InferenceInput& in, DetectionBatch& out) override {
    out.items.clear();
    out.inference_ms = 0.f;
    if (!ctx_ || !engine_ || in.nchw == nullptr || in.width <= 0 || in.height <= 0) {
      return false;
    }

    const int W = in.width;
    const int H = in.height;
    const std::size_t n_in = static_cast<std::size_t>(3) * W * H;

    nvinfer1::Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = 1;
    in_dims.d[1] = 3;
    in_dims.d[2] = H;
    in_dims.d[3] = W;
    if (!ctx_->setInputShape(engine_->input_name().c_str(), in_dims)) {
      std::cerr << "integra: TRT setInputShape failed\n";
      return false;
    }

    const float* d_in = nullptr;
    if (in.on_device) {
      d_in = in.nchw;
    } else {
      const std::size_t need = n_in * sizeof(float);
      if (cap_input_ < need) {
        if (d_input_) cudaFree(d_input_);
        if (cudaMalloc(&d_input_, need) != cudaSuccess) {
          std::cerr << "integra: cudaMalloc input (nchw) failed\n";
          return false;
        }
        cap_input_ = need;
      }
      // Если caller не указал stream — используем свой, иначе соблюдаем его.
      cudaStream_t src_stream =
          in.cuda_stream ? static_cast<cudaStream_t>(in.cuda_stream) : stream_;
      if (cudaMemcpyAsync(d_input_, in.nchw, need, cudaMemcpyHostToDevice,
                          src_stream) != cudaSuccess) {
        std::cerr << "integra: cudaMemcpyAsync H2D nchw failed\n";
        return false;
      }
      d_in = static_cast<float*>(d_input_);
    }

    const nvinfer1::Dims od = ctx_->getTensorShape(engine_->output_name().c_str());
    for (int i = 0; i < od.nbDims; ++i) {
      if (od.d[i] <= 0) {
        std::cerr << "integra: выходной тензор имеет неполную размерность\n";
        return false;
      }
    }
    if (od.nbDims != 3 || od.d[0] != 1) {
      std::cerr << "integra: ожидается выход rank 3 с batch 1\n";
      return false;
    }
    const std::size_t out_elems = volume_dims(od);
    const std::size_t need_out = out_elems * sizeof(float);
    if (cap_output_ < need_out) {
      if (d_output_) cudaFree(d_output_);
      if (cudaMalloc(&d_output_, need_out) != cudaSuccess) {
        std::cerr << "integra: cudaMalloc output failed\n";
        return false;
      }
      cap_output_ = need_out;
    }
    if (host_pinned_cap_ < need_out) {
      if (h_output_pinned_) cudaFreeHost(h_output_pinned_);
      if (cudaMallocHost(&h_output_pinned_, need_out) != cudaSuccess) {
        std::cerr << "integra: cudaMallocHost output failed\n";
        return false;
      }
      host_pinned_cap_ = need_out;
    }

    ctx_->setTensorAddress(engine_->input_name().c_str(),
                           const_cast<void*>(static_cast<const void*>(d_in)));
    ctx_->setTensorAddress(engine_->output_name().c_str(), d_output_);

    // Если препроцесс шёл на другом stream — синхронизируем его до enqueue.
    cudaStream_t exec_stream = stream_;
    if (in.on_device && in.cuda_stream != nullptr &&
        in.cuda_stream != static_cast<void*>(stream_)) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(in.cuda_stream));
    }

    auto t0 = std::chrono::steady_clock::now();
    if (!ctx_->enqueueV3(exec_stream)) {
      std::cerr << "integra: TRT enqueueV3 (nchw) failed\n";
      return false;
    }
    if (cudaMemcpyAsync(h_output_pinned_, d_output_, need_out, cudaMemcpyDeviceToHost,
                        exec_stream) != cudaSuccess) {
      std::cerr << "integra: cudaMemcpyAsync D2H (nchw) failed\n";
      return false;
    }
    if (cudaStreamSynchronize(exec_stream) != cudaSuccess) {
      std::cerr << "integra: cudaStreamSynchronize (nchw) failed\n";
      return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    out.inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const std::int64_t d1 = od.d[1];
    const std::int64_t d2 = od.d[2];
    const bool chw = yolo_output_channel_first(d1, d2);
    PostprocessParams pp = engine_->postprocess();
    std::vector<Detection> decoded;
    decode_yolov8_flat(static_cast<const float*>(h_output_pinned_),
                       static_cast<int>(d1), static_cast<int>(d2), chw, pp,
                       static_cast<float>(W), static_cast<float>(H),
                       transpose_scratch_, decoded);
    nms_greedy_xyxy(decoded, pp.nms_iou_threshold);
    out.items = std::move(decoded);
    return true;
  }

 private:
  std::shared_ptr<SharedTRTEngine> engine_;
  nvinfer1::IExecutionContext* ctx_ = nullptr;
  cudaStream_t stream_{};

  void* d_input_ = nullptr;
  std::size_t cap_input_ = 0;

  void* d_output_ = nullptr;
  std::size_t cap_output_ = 0;

  void* d_bgr_ = nullptr;
  std::size_t cap_bgr_ = 0;

  void* h_output_pinned_ = nullptr;
  std::size_t host_pinned_cap_ = 0;

  cv::Mat resize_buf_;
  std::vector<float> transpose_scratch_;
};

std::unique_ptr<IStreamContext> SharedTRTEngine::create_stream_context() {
  auto ctx = std::make_unique<TRTStreamContext>(shared_from_this());
  return ctx;
}

// ---------------------------------------------------------------------------
// Кэш SharedTRTEngine по model_path (weak_ptr — engine не утечёт).
// Защищён мьютексом — make_shared_trt_engine может вызываться из разных потоков.
// ---------------------------------------------------------------------------
namespace {

std::mutex& trt_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::weak_ptr<SharedTRTEngine>>& trt_cache() {
  static std::unordered_map<std::string, std::weak_ptr<SharedTRTEngine>> c;
  return c;
}

}  // namespace

std::shared_ptr<ISharedEngine> make_shared_trt_engine(const InferenceEngineConfig& cfg) {
  if (cfg.model_path.empty()) {
    std::cerr << "integra: tensorrt SharedEngine: пустой model_path\n";
    return nullptr;
  }
  std::lock_guard<std::mutex> lk(trt_cache_mutex());
  auto it = trt_cache().find(cfg.model_path);
  if (it != trt_cache().end()) {
    if (auto sp = it->second.lock()) {
      return sp;
    }
    trt_cache().erase(it);
  }
  auto eng = std::make_shared<SharedTRTEngine>();
  if (!eng->init(cfg)) {
    return nullptr;
  }
  trt_cache()[cfg.model_path] = eng;
  return eng;
}

// ---------------------------------------------------------------------------
// Legacy adapter: IInferenceEngine поверх Shared+Stream.
// Сохраняет совместимость с integra-analyticsd и parity_native_vs_python.py.
// ---------------------------------------------------------------------------
namespace {

class TRTLegacyAdapter final : public IInferenceEngine {
 public:
  bool init(const InferenceEngineConfig& cfg) override {
    shared_ = make_shared_trt_engine(cfg);
    if (!shared_) {
      return false;
    }
    ctx_ = shared_->create_stream_context();
    return ctx_ != nullptr;
  }

  bool infer(const InferenceInput& in, DetectionBatch& out) override {
    if (!ctx_) return false;
    return ctx_->infer_nchw(in, out);
  }

  bool is_stub() const override { return false; }

 private:
  std::shared_ptr<ISharedEngine> shared_;
  std::unique_ptr<IStreamContext> ctx_;
};

}  // namespace

std::unique_ptr<IInferenceEngine> make_tensorrt_engine() {
  return std::make_unique<TRTLegacyAdapter>();
}

}  // namespace integra
