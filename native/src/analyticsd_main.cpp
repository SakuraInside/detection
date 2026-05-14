// integra-analyticsd: TCP server for Python → C++ analytics (TensorRT/ONNX + tracker + FSM).
//
// Protocol: docs/native_analytics_protocol.md (JSON line + u32 length + BGR bytes).
//
// Build: this executable links integra_core. For real inference you must build
// with -DINTEGRA_WITH_TENSORRT=ON and provide --engine tensorrt + --model .engine.

#include "integra/frame_filter.hpp"
#include "integra/gpu_preprocess.hpp"
#include "integra/inference_engine.hpp"
#include "integra/iou_tracker.hpp"
#include "integra/pipeline.hpp"
#include "integra/scene_analyzer.hpp"
#include "integra/yolo_letterbox.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
using socket_t = SOCKET;
static void closesock(socket_t s) { closesocket(s); }
static bool sock_init() {
  WSADATA wsa;
  return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
}
static void sock_shutdown() { WSACleanup(); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static void closesock(socket_t s) { ::close(s); }
static bool sock_init() { return true; }
static void sock_shutdown() {}
#endif

namespace {

static std::string json_esc(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char uc : s) {
    char c = static_cast<char>(uc);
    if (c == '"' || c == '\\') {
      o.push_back('\\');
    }
    o.push_back(c);
  }
  return o;
}

static bool read_line(socket_t s, std::string& out) {
  out.clear();
  char ch;
  while (true) {
#ifdef _WIN32
    int n = recv(s, &ch, 1, 0);
#else
    ssize_t n = ::recv(s, &ch, 1, 0);
#endif
    if (n <= 0) return false;
    if (ch == '\n') break;
    out.push_back(ch);
    if (out.size() > 1024 * 1024) return false;
  }
  return true;
}

static bool recv_exact(socket_t s, void* dst, std::size_t n) {
  std::size_t pos = 0;
  while (pos < n) {
#ifdef _WIN32
    int got = recv(s, reinterpret_cast<char*>(dst) + pos, static_cast<int>(n - pos), 0);
#else
    ssize_t got = ::recv(s, reinterpret_cast<char*>(dst) + pos, n - pos, 0);
#endif
    if (got <= 0) return false;
    pos += static_cast<std::size_t>(got);
  }
  return true;
}

static bool send_all(socket_t s, const std::string& data) {
  std::size_t pos = 0;
  while (pos < data.size()) {
#ifdef _WIN32
    int n = send(s, data.data() + pos, static_cast<int>(data.size() - pos), 0);
#else
    ssize_t n = ::send(s, data.data() + pos, data.size() - pos, 0);
#endif
    if (n <= 0) return false;
    pos += static_cast<std::size_t>(n);
  }
  return true;
}

static bool has_type(const std::string& line, const char* type) {
  std::string needle = std::string("\"type\":\"") + type + "\"";
  return line.find(needle) != std::string::npos;
}

static bool parse_u64_field(const std::string& line, const char* key, std::uint64_t& out) {
  std::string needle = std::string("\"") + key + "\":";
  auto p = line.find(needle);
  if (p == std::string::npos) return false;
  p += needle.size();
  char* end = nullptr;
  const char* s = line.c_str() + p;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) return false;
  out = static_cast<std::uint64_t>(v);
  return true;
}

static bool parse_i32_field(const std::string& line, const char* key, int& out) {
  std::string needle = std::string("\"") + key + "\":";
  auto p = line.find(needle);
  if (p == std::string::npos) return false;
  p += needle.size();
  char* end = nullptr;
  const char* s = line.c_str() + p;
  long v = std::strtol(s, &end, 10);
  if (end == s) return false;
  out = static_cast<int>(v);
  return true;
}

static bool parse_f64_field(const std::string& line, const char* key, double& out) {
  std::string needle = std::string("\"") + key + "\":";
  auto p = line.find(needle);
  if (p == std::string::npos) return false;
  p += needle.size();
  char* end = nullptr;
  const char* s = line.c_str() + p;
  double v = std::strtod(s, &end);
  if (end == s) return false;
  out = v;
  return true;
}

static bool parse_i32_array_field(const std::string& line, const char* key, std::vector<int>& out) {
  std::string needle = std::string("\"") + key + "\":[";
  auto p = line.find(needle);
  if (p == std::string::npos) return false;
  p += needle.size();
  auto e = line.find(']', p);
  if (e == std::string::npos || e <= p) return false;
  out.clear();
  std::string body = line.substr(p, e - p);
  std::size_t cur = 0;
  while (cur < body.size()) {
    while (cur < body.size() && (body[cur] == ' ' || body[cur] == ',')) ++cur;
    if (cur >= body.size()) break;
    char* end = nullptr;
    const char* s = body.c_str() + cur;
    long v = std::strtol(s, &end, 10);
    if (end == s) {
      ++cur;
      continue;
    }
    out.push_back(static_cast<int>(v));
    cur = static_cast<std::size_t>(end - body.c_str());
  }
  return true;
}

struct ServerConfig {
  std::string listen = "127.0.0.1:9909";
  std::string engine_kind = "tensorrt";
  std::string model_path;
  int input_size = 640;
  integra::PostprocessParams pp;
  integra::AnalyzerParams ap;
};

// FrameFilterConfig + apply_frame_filter — общая реализация (см. include/integra/frame_filter.hpp).
using integra::FrameFilterConfig;
using integra::apply_frame_filter;

static bool parse_host_port(const std::string& spec, std::string& host, int& port) {
  auto pos = spec.rfind(':');
  if (pos == std::string::npos) return false;
  host = spec.substr(0, pos);
  port = std::atoi(spec.substr(pos + 1).c_str());
  return port > 0;
}

static double now_wall_sec() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::string hello_ok(const std::string& engine_kind, bool real_engine) {
  std::ostringstream os;
  os << "{\"type\":\"hello_ok\",\"v\":1,\"ok\":true,\"message\":\""
     << (real_engine ? "ready" : "stub backend")
     << "\",\"engine\":\"" << json_esc(engine_kind) << "\",\"real_engine\":"
     << (real_engine ? "true" : "false") << "}\n";
  return os.str();
}

static std::string reset_ok() {
  return std::string("{\"type\":\"reset_ok\",\"v\":1,\"ok\":true}\n");
}

static std::string error_msg(const std::string& msg) {
  std::ostringstream os;
  os << "{\"type\":\"error\",\"v\":1,\"message\":\"" << json_esc(msg) << "\"}\n";
  return os.str();
}

// is_relevant_class — общая реализация (см. include/integra/frame_filter.hpp).
using integra::is_relevant_class;

#if 0
// Старая локальная копия удалена — оставлено для архива:
static bool is_relevant_class_legacy(int cls_id, int person_class_id, const std::vector<int>& object_classes) {
  if (cls_id == person_class_id) {
    return true;
  }
  if (!object_classes.empty()) {
    return std::find(object_classes.begin(), object_classes.end(), cls_id) != object_classes.end();
  }
  switch (cls_id) {
    case 24:
    case 26:
    case 28:
    case 39:
    case 40:
    case 41:
    case 45:
      return true;
    default:
      return false;
  }
}
#endif

static void write_result(socket_t c,
                         std::uint64_t frame_id,
                         double pos_ms,
                         const std::vector<integra::Detection>& dets,
                         const std::vector<integra::Detection>& persons,
                         const std::vector<integra::TrackSnapshot>& tracks,
                         const std::vector<integra::AlarmEvent>& events,
                         double preprocess_ms,
                         double infer_ms,
                         double postprocess_ms,
                         double tracker_ms,
                         double analyzer_ms) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "{\"type\":\"result\",\"v\":1,\"frame_id\":" << frame_id << ",\"pos_ms\":" << pos_ms
     << ",\"metrics\":{\"preprocess_ms\":" << preprocess_ms << ",\"infer_ms\":" << infer_ms
     << ",\"postprocess_ms\":" << postprocess_ms << ",\"tracker_ms\":" << tracker_ms
     << ",\"analyzer_ms\":" << analyzer_ms << "},\"detections\":[";
  for (std::size_t i = 0; i < dets.size(); ++i) {
    const auto& d = dets[i];
    if (i) os << ",";
    os << "{\"track_id\":" << d.track_id << ",\"cls_id\":" << d.class_id << ",\"cls_name\":\""
       << json_esc(d.cls_name) << "\",\"confidence\":" << d.confidence << ",\"bbox\":["
       << d.bbox.x1 << "," << d.bbox.y1 << "," << d.bbox.x2 << "," << d.bbox.y2 << "]}";
  }
  os << "],\"persons\":[";
  for (std::size_t i = 0; i < persons.size(); ++i) {
    const auto& d = persons[i];
    if (i) os << ",";
    os << "{\"track_id\":" << d.track_id << ",\"cls_id\":" << d.class_id << ",\"cls_name\":\""
       << json_esc(d.cls_name) << "\",\"confidence\":" << d.confidence << ",\"bbox\":["
       << d.bbox.x1 << "," << d.bbox.y1 << "," << d.bbox.x2 << "," << d.bbox.y2 << "]}";
  }
  os << "],\"tracks\":[";
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const auto& t = tracks[i];
    if (i) os << ",";
    os << "{\"id\":" << t.id << ",\"cls\":\"" << json_esc(t.cls) << "\",\"state\":\""
       << json_esc(t.state) << "\",\"bbox\":[" << t.bbox[0] << "," << t.bbox[1] << "," << t.bbox[2]
       << "," << t.bbox[3] << "],\"conf\":" << t.conf << ",\"static_for_sec\":" << t.static_for_sec
       << ",\"unattended_for_sec\":" << t.unattended_for_sec << ",\"alarm\":"
       << (t.alarm ? "true" : "false") << "}";
  }
  os << "],\"events\":[";
  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto& e = events[i];
    if (i) os << ",";
    os << "{\"type\":\"" << json_esc(e.type) << "\",\"track_id\":" << e.track_id << ",\"cls_id\":"
       << e.cls_id << ",\"cls_name\":\"" << json_esc(e.cls_name) << "\",\"confidence\":"
       << e.confidence << ",\"bbox\":[" << e.bbox[0] << "," << e.bbox[1] << "," << e.bbox[2] << ","
       << e.bbox[3] << "]";
    if (!e.note.empty()) {
      os << ",\"note\":\"" << json_esc(e.note) << "\"";
    }
    os << "}";
  }
  os << "]}\n";
  (void)send_all(c, os.str());
}

static void handle_client(socket_t c, const ServerConfig& cfg) {
  // Expect hello
  std::string line;
  if (!read_line(c, line) || !has_type(line, "hello")) {
    send_all(c, error_msg("expected hello"));
    return;
  }
  ServerConfig session_cfg = cfg;
  FrameFilterConfig ff;
  float trk_iou = 0.35f;
  int trk_miss = 10;
  bool trk_soft = true;
  // Применяем runtime-настройки из hello (протокол Python bridge):
  // postprocess/analyzer и whitelist классов для строгого anti-noise режима.
  {
    double v = 0.0;
    int iv = 0;
    if (parse_f64_field(line, "conf", v)) session_cfg.pp.conf_threshold = static_cast<float>(v);
    if (parse_f64_field(line, "iou", v)) session_cfg.pp.nms_iou_threshold = static_cast<float>(v);
    if (parse_f64_field(line, "nms_iou", v)) session_cfg.pp.nms_iou_threshold = static_cast<float>(v);
    if (parse_f64_field(line, "tracker_iou", v)) trk_iou = static_cast<float>(v);
    if (parse_i32_field(line, "tracker_max_missed", iv)) trk_miss = std::max(1, iv);
    if (parse_i32_field(line, "tracker_soft_centroid", iv)) trk_soft = (iv != 0);
    if (parse_i32_field(line, "person_class_id", iv)) session_cfg.ap.person_class_id = iv;
    if (parse_i32_field(line, "min_box_size_px", iv)) ff.min_box_px = std::max(8, iv);
    ff.tune_from_postprocess(session_cfg.pp.conf_threshold);
  }

  auto engine = integra::make_inference_engine(cfg.engine_kind);
  if (!engine) {
    send_all(c, error_msg("unknown engine"));
    return;
  }
  integra::InferenceEngineConfig ec;
  ec.model_path = cfg.model_path;
  ec.input_size = cfg.input_size;
  ec.postprocess = session_cfg.pp;
  if (!engine->init(ec)) {
    send_all(c, error_msg("engine init failed"));
    return;
  }
  const bool real_engine = !engine->is_stub();
  if (!send_all(c, hello_ok(cfg.engine_kind, real_engine))) {
    return;
  }

  integra::GpuLetterboxPrep prep;
  integra::IouTracker tracker(trk_iou, trk_miss, trk_soft);
  integra::SceneAnalyzer analyzer(session_cfg.ap);
  std::vector<int> object_classes;
  parse_i32_array_field(line, "object_classes", object_classes);
  object_classes.erase(
      std::remove(object_classes.begin(), object_classes.end(), session_cfg.ap.person_class_id),
      object_classes.end());

#if INTEGRA_HAS_CUDA
  int pw = 0, ph = 0;
#else
  std::vector<float> cpu_blob;
#endif

  std::vector<unsigned char> payload;

  while (true) {
    if (!read_line(c, line)) break;
    if (has_type(line, "reset")) {
      tracker.reset();
      analyzer.reset();
      if (!send_all(c, reset_ok())) {
        break;
      }
      continue;
    }
    if (!has_type(line, "frame")) {
      send_all(c, error_msg("expected frame"));
      break;
    }

    std::uint64_t frame_id = 0;
    int w = 0, h = 0;
    double pos_ms = 0.0;
    if (!parse_u64_field(line, "frame_id", frame_id) ||
        !parse_i32_field(line, "width", w) ||
        !parse_i32_field(line, "height", h) ||
        !parse_f64_field(line, "pos_ms", pos_ms)) {
      send_all(c, error_msg("bad frame meta"));
      break;
    }
    std::uint32_t ln_le = 0;
    if (!recv_exact(c, &ln_le, 4)) break;
    const std::uint32_t ln = ln_le;
    const std::uint32_t expected = static_cast<std::uint32_t>(w) * static_cast<std::uint32_t>(h) * 3u;
    if (w <= 0 || h <= 0 || ln != expected) {
      send_all(c, error_msg("bad payload length"));
      break;
    }
    payload.resize(ln);
    if (!recv_exact(c, payload.data(), ln)) break;

    cv::Mat bgr(h, w, CV_8UC3, payload.data());
    integra::LetterboxMeta lb{};
    cv::Mat feed;
    if (cfg.input_size > 0) {
      if (!integra::yolo_letterbox_bgr(bgr, cfg.input_size, feed, &lb)) {
        send_all(c, error_msg("letterbox failed"));
        break;
      }
    } else {
      feed = bgr;
      lb.r = 1.f;
      lb.pad_left = lb.pad_top = 0;
    }

    auto t0 = std::chrono::steady_clock::now();
#if INTEGRA_HAS_CUDA
    if (!prep.upload_and_preprocess(feed, pw, ph)) {
      send_all(c, error_msg("preprocess failed"));
      break;
    }
#else
    if (!prep.upload_and_preprocess(feed, cpu_blob)) {
      send_all(c, error_msg("preprocess failed"));
      break;
    }
#endif
    auto t1 = std::chrono::steady_clock::now();

    integra::DetectionBatch batch;
    integra::InferenceInput vin;
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
    vin.cuda_stream = nullptr;
#endif
    if (!engine->infer(vin, batch)) {
      send_all(c, error_msg("infer failed"));
      break;
    }
    auto t2 = std::chrono::steady_clock::now();

    if (cfg.input_size > 0) {
      integra::yolo_unletterbox_dets(batch.items, lb, bgr.cols, bgr.rows);
    }
    // Отсекаем нерелевантные классы до трекера/FSM:
    // это резко снижает ложные треки и RAM на сцене без деградации целевого кейса.
    std::vector<integra::Detection> filtered;
    filtered.reserve(batch.items.size());
    for (const auto& d : batch.items) {
      if (is_relevant_class(d.class_id, session_cfg.ap.person_class_id, object_classes)) {
        filtered.push_back(d);
      }
    }
    batch.items = apply_frame_filter(filtered, bgr.cols, bgr.rows, session_cfg.ap.person_class_id, ff);
    integra::merge_same_class_objects_only(batch.items, session_cfg.ap.person_class_id, 0.48f);
    integra::merge_luggage_cross_class_by_iou(batch.items, session_cfg.ap.person_class_id, 0.38f);
    integra::suppress_duplicate_bottles_by_iou(batch.items, session_cfg.ap.person_class_id, 0.40f);

    auto t3 = std::chrono::steady_clock::now();
    tracker.update(batch.items, bgr.cols, bgr.rows);
    auto t4 = std::chrono::steady_clock::now();

    std::vector<integra::Detection> persons;
    std::vector<integra::Detection> objects;
    persons.reserve(batch.items.size());
    objects.reserve(batch.items.size());
    for (const auto& d : batch.items) {
      if (d.class_id == session_cfg.ap.person_class_id) persons.push_back(d);
      else objects.push_back(d);
    }

    const double ts = now_wall_sec();
    auto t5 = std::chrono::steady_clock::now();
    std::vector<integra::AlarmEvent> evs =
        analyzer.ingest(ts, pos_ms, "main", objects, persons, bgr.cols, bgr.rows);
    auto t6 = std::chrono::steady_clock::now();
    std::vector<integra::TrackSnapshot> tracks = analyzer.tracks_snapshot(ts);

    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double postprocess_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double tracker_ms =
        std::chrono::duration<double, std::milli>(t4 - t3).count();
    const double analyzer_ms =
        std::chrono::duration<double, std::milli>(t6 - t5).count();

    write_result(c, frame_id, pos_ms, objects, persons, tracks, evs,
                 preprocess_ms, infer_ms, postprocess_ms, tracker_ms, analyzer_ms);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (!sock_init()) {
    std::cerr << "integra-analyticsd: socket init failed\n";
    return 2;
  }

  ServerConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--listen") && i + 1 < argc) cfg.listen = argv[++i];
    else if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) cfg.engine_kind = argv[++i];
    else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) cfg.model_path = argv[++i];
    else if (!std::strcmp(argv[i], "--imgsz") && i + 1 < argc) cfg.input_size = std::atoi(argv[++i]);
  }

  std::string host;
  int port = 0;
  if (!parse_host_port(cfg.listen, host, port)) {
    std::cerr << "integra-analyticsd: invalid --listen, expected host:port\n";
    sock_shutdown();
    return 2;
  }

  // Sensible defaults matching Python-ish config.
  cfg.pp.conf_threshold = 0.22f;
  cfg.pp.nms_iou_threshold = 0.45f;
  cfg.pp.num_classes = 80;
  cfg.pp.num_anchors = 8400;
  cfg.ap.person_class_id = 0;
  cfg.ap.static_displacement_px = 7.0;
  cfg.ap.static_window_sec = 3.0;
  cfg.ap.abandon_time_sec = 15.0;
  cfg.ap.owner_proximity_px = 180.0;
  cfg.ap.owner_left_sec = 5.0;
  cfg.ap.disappear_grace_sec = 4.0;
  cfg.ap.min_object_area_px = 100.0;
  cfg.ap.centroid_history_maxlen = 72;
  cfg.ap.max_active_tracks = 256;

  socket_t srv = static_cast<socket_t>(socket(AF_INET, SOCK_STREAM, 0));
  if (
#ifdef _WIN32
      srv == INVALID_SOCKET
#else
      srv < 0
#endif
  ) {
    std::cerr << "integra-analyticsd: socket() failed\n";
    sock_shutdown();
    return 2;
  }
  int one = 1;
#ifdef _WIN32
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
#else
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = inet_addr(host.c_str());
  if (addr.sin_addr.s_addr == INADDR_NONE) {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  }
  if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    std::cerr << "integra-analyticsd: bind failed\n";
    closesock(srv);
    sock_shutdown();
    return 2;
  }
  if (listen(srv, 4) != 0) {
    std::cerr << "integra-analyticsd: listen failed\n";
    closesock(srv);
    sock_shutdown();
    return 2;
  }

  std::cerr << "integra-analyticsd listening on " << host << ":" << port << "\n";
  while (true) {
    socket_t c =
#ifdef _WIN32
        accept(srv, nullptr, nullptr);
    if (c == INVALID_SOCKET) continue;
#else
        accept(srv, nullptr, nullptr);
    if (c < 0) continue;
#endif
    // Single-thread per connection.
    std::thread([c, cfg]() {
      handle_client(c, cfg);
      closesock(c);
    }).detach();
  }

  // unreachable
  closesock(srv);
  sock_shutdown();
  return 0;
}

