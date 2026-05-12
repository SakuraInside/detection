// integra_trt_bake: офлайн-сборка TensorRT .plan/.engine из ONNX для production.
// Линкуется с nvinfer + nvonnxparser. Использование см. native/README.md и
// native/scripts/trt_bake_cached.sh (кеш по sha256 + compute capability).

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace {

class TrtBakeLogger final : public nvinfer1::ILogger {
 public:
  explicit TrtBakeLogger(bool verbose) : verbose_(verbose) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (!verbose_ && severity > Severity::kWARNING) {
      return;
    }
    std::cerr << "[integra_trt_bake] " << msg << "\n";
  }

 private:
  bool verbose_;
};

static void usage() {
  std::cerr
      << "integra_trt_bake — сериализация TensorRT engine из ONNX.\n"
      << "\n"
      << "Usage:\n"
      << "  integra_trt_bake --onnx PATH --out PATH [--fp16] [--workspace-mb N] [--verbose]\n"
      << "\n"
      << "  --fp16           Включить kFP16, если platformHasFastFp16 (рекомендуется sm_75+).\n"
      << "  --workspace-mb   Лимит workspace pool (по умолчанию 4096).\n"
      << "  --verbose        Печатать INFO от TensorRT парсера/билдера.\n"
      << "\n"
      << "Кеш с детерминированным именем: native/scripts/trt_bake_cached.sh\n";
}

static bool write_file(const std::string& path, const void* data, std::size_t size) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "integra_trt_bake: cannot write: " << path << "\n";
    return false;
  }
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  return static_cast<bool>(f);
}

static void destroy_host_plan(nvinfer1::IHostMemory* plan) {
  if (!plan) return;
#if NV_TENSORRT_MAJOR >= 10
  delete plan;
#else
  plan->destroy();
#endif
}

static void destroy_config(nvinfer1::IBuilderConfig* cfg) {
  if (!cfg) return;
#if NV_TENSORRT_MAJOR >= 10
  delete cfg;
#else
  cfg->destroy();
#endif
}

static void destroy_network(nvinfer1::INetworkDefinition* net) {
  if (!net) return;
#if NV_TENSORRT_MAJOR >= 10
  delete net;
#else
  net->destroy();
#endif
}

static void destroy_builder(nvinfer1::IBuilder* b) {
  if (!b) return;
#if NV_TENSORRT_MAJOR >= 10
  delete b;
#else
  b->destroy();
#endif
}

static void destroy_parser(nvonnxparser::IParser* p) {
  if (!p) return;
#if NV_TENSORRT_MAJOR >= 10
  delete p;
#else
  p->destroy();
#endif
}

}  // namespace

int main(int argc, char** argv) {
  std::string onnx_path;
  std::string out_path;
  bool fp16 = false;
  bool verbose = false;
  std::uint32_t workspace_mb = 4096;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!std::strcmp(a, "--onnx") && i + 1 < argc) {
      onnx_path = argv[++i];
    } else if (!std::strcmp(a, "--out") && i + 1 < argc) {
      out_path = argv[++i];
    } else if (!std::strcmp(a, "--fp16")) {
      fp16 = true;
    } else if (!std::strcmp(a, "--verbose")) {
      verbose = true;
    } else if (!std::strcmp(a, "--workspace-mb") && i + 1 < argc) {
      workspace_mb = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
      usage();
      return 0;
    } else {
      std::cerr << "integra_trt_bake: unknown arg: " << a << "\n";
      usage();
      return 2;
    }
  }

  if (onnx_path.empty() || out_path.empty()) {
    usage();
    return 2;
  }

  TrtBakeLogger logger(verbose);
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  if (!builder) {
    std::cerr << "integra_trt_bake: createInferBuilder failed\n";
    return 1;
  }

  const auto explicit_batch =
      1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
  if (!network) {
    std::cerr << "integra_trt_bake: createNetworkV2 failed\n";
    destroy_builder(builder);
    return 1;
  }

  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  if (!parser) {
    std::cerr << "integra_trt_bake: createParser failed\n";
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }

  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(verbose ? nvinfer1::ILogger::Severity::kINFO
                                   : nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "integra_trt_bake: не удалось распарсить ONNX: " << onnx_path << "\n";
    destroy_parser(parser);
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }

  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (!config) {
    std::cerr << "integra_trt_bake: createBuilderConfig failed\n";
    destroy_parser(parser);
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }

  const std::size_t workspace_bytes = static_cast<std::size_t>(workspace_mb) * 1024 * 1024;
#if NV_TENSORRT_MAJOR >= 10
  if (!config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes)) {
    std::cerr << "integra_trt_bake: setMemoryPoolLimit(WORKSPACE) failed\n";
    destroy_config(config);
    destroy_parser(parser);
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }
#else
  if (!config->setMaxWorkspaceSize(workspace_bytes)) {
    std::cerr << "integra_trt_bake: setMaxWorkspaceSize failed\n";
    destroy_config(config);
    destroy_parser(parser);
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }
#endif

  if (fp16) {
    if (builder->platformHasFastFp16()) {
      if (!config->setFlag(nvinfer1::BuilderFlag::kFP16)) {
        std::cerr << "integra_trt_bake: setFlag(kFP16) failed\n";
      } else {
        std::cerr << "integra_trt_bake: FP16 включён\n";
      }
    } else {
      std::cerr << "integra_trt_bake: предупреждение: FP16 недоступен на этой платформе, остаёмся в FP32\n";
    }
  }

  std::cerr << "integra_trt_bake: buildSerializedNetwork (может занять несколько минут)...\n";

  nvinfer1::IHostMemory* plan = nullptr;
#if NV_TENSORRT_MAJOR >= 8
  plan = builder->buildSerializedNetwork(*network, *config);
#else
  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  if (engine) {
    plan = engine->serialize();
#if NV_TENSORRT_MAJOR >= 10
    delete engine;
#else
    engine->destroy();
#endif
  }
#endif

  if (!plan || plan->size() == 0) {
    std::cerr << "integra_trt_bake: сборка не удалась (пустой plan)\n";
    destroy_host_plan(plan);
    destroy_config(config);
    destroy_parser(parser);
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }

  if (!write_file(out_path, plan->data(), plan->size())) {
    destroy_host_plan(plan);
    destroy_config(config);
    destroy_parser(parser);
    destroy_network(network);
    destroy_builder(builder);
    return 1;
  }

  std::cerr << "integra_trt_bake: записано " << out_path << " (" << plan->size() << " байт)\n";
  std::cout << out_path << "\n";

  destroy_host_plan(plan);
  destroy_config(config);
  destroy_parser(parser);
  destroy_network(network);
  destroy_builder(builder);

  return 0;
}
