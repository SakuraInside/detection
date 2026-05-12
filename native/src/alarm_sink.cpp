#include "integra/alarm_sink.hpp"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <sstream>
#include <string_view>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace integra {

namespace {

std::string json_esc(std::string_view s) {
  std::string o;
  for (unsigned char uc : s) {
    char c = static_cast<char>(uc);
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += c;
  }
  return o;
}

#ifdef _WIN32
using socket_len_t = int;
static bool ensure_winsock() {
  static bool inited = false;
  static bool ok = false;
  if (!inited) {
    WSADATA wsa;
    ok = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    inited = true;
  }
  return ok;
}
#endif

}  // namespace

void AlarmJsonlSink::configure(const std::string& host, int port) {
  close();
  host_ = host;
  port_ = port;
}

void AlarmJsonlSink::close() {
  if (fd_ != static_cast<alarm_socket_t>(-1)) {
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(fd_));
#else
    ::close(static_cast<int>(fd_));
#endif
    fd_ = static_cast<alarm_socket_t>(-1);
  }
  port_ = 0;
}

void AlarmJsonlSink::send(const AlarmEvent& ev) {
  if (port_ <= 0) {
    return;
  }
#ifdef _WIN32
  if (!ensure_winsock()) {
    return;
  }
#endif
  if (fd_ == static_cast<alarm_socket_t>(-1)) {
    char port_buf[16];
    std::snprintf(port_buf, sizeof(port_buf), "%d", port_);
    struct addrinfo hints {};
    struct addrinfo* res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    if (getaddrinfo(host_.c_str(), port_buf, &hints, &res) != 0 || res == nullptr) {
      return;
    }
    int s = -1;
    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next) {
      s = static_cast<int>(socket(p->ai_family, p->ai_socktype, p->ai_protocol));
      if (s < 0) {
        continue;
      }
      if (connect(s, p->ai_addr, p->ai_addrlen) == 0) {
        break;
      }
#ifdef _WIN32
      closesocket(s);
#else
      ::close(s);
#endif
      s = -1;
    }
    freeaddrinfo(res);
    if (s < 0) {
      return;
    }
    fd_ = static_cast<alarm_socket_t>(s);
  }

  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "{\"type\":\"" << json_esc(ev.type) << "\",\"camera_id\":\"" << json_esc(ev.camera_id)
     << "\",\"track_id\":" << ev.track_id << ",\"cls_id\":" << ev.cls_id << ",\"cls_name\":\""
     << json_esc(ev.cls_name) << "\",\"confidence\":" << ev.confidence << ",\"ts_ms\":"
     << ev.ts_wall_ms << ",\"video_pos_ms\":" << ev.video_pos_ms << ",\"bbox\":[" << ev.bbox[0] << ","
     << ev.bbox[1] << "," << ev.bbox[2] << "," << ev.bbox[3] << "]";
  if (!ev.note.empty()) {
    os << ",\"note\":\"" << json_esc(ev.note) << "\"";
  }
  os << "}\n";
  const std::string line = os.str();
#ifdef _WIN32
  int n = static_cast<int>(
      ::send(static_cast<SOCKET>(fd_), line.data(), static_cast<int>(line.size()), MSG_NOSIGNAL));
#else
  int n = static_cast<int>(
      ::send(static_cast<int>(fd_), line.data(), static_cast<int>(line.size()), MSG_NOSIGNAL));
#endif
  if (n < 0 || static_cast<std::size_t>(n) != line.size()) {
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(fd_));
#else
    ::close(static_cast<int>(fd_));
#endif
    fd_ = static_cast<alarm_socket_t>(-1);
  }
}

}  // namespace integra
