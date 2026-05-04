#pragma once

#include "integra/types.hpp"

#include <string>

namespace integra {

/// Отправка JSON-lines на TCP (integra-alarmd или внешняя интеграция).
class AlarmJsonlSink {
 public:
  void configure(const std::string& host, int port);
  void send(const AlarmEvent& ev);
  void close();
  bool is_configured() const { return port_ > 0; }

 private:
  std::string host_ = "127.0.0.1";
  int port_ = 0;
  int fd_ = -1;
};

}  // namespace integra
