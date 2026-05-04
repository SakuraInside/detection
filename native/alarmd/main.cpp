// Отдельный сервис приёма тревог: TCP JSON-lines, запись в файл + stdout.
// Интеграция с SIEM / Kafka — добавить позже поверх этого процесса.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

static std::mutex g_log_mutex;

static void handle_client(int cfd, const std::string& log_path) {
  char buf[8192];
  std::string carry;
  while (true) {
    ssize_t n = recv(cfd, buf, sizeof(buf), 0);
    if (n <= 0) {
      break;
    }
    carry.append(buf, static_cast<std::size_t>(n));
    while (true) {
      auto pos = carry.find('\n');
      if (pos == std::string::npos) {
        break;
      }
      std::string line = carry.substr(0, pos);
      carry.erase(0, pos + 1);
      if (!line.empty()) {
        std::cout << line << "\n" << std::flush;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        std::lock_guard<std::mutex> lock(g_log_mutex);
        std::ofstream log(log_path, std::ios::app);
        if (log) {
          log << ms << "\t" << line << "\n";
        }
      }
    }
  }
  ::close(cfd);
}

int main(int argc, char** argv) {
  const char* listen_arg = "0.0.0.0:9090";
  std::string log_path = "logs/native_alerts.jsonl";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--listen") && i + 1 < argc) {
      listen_arg = argv[++i];
    } else if (!strcmp(argv[i], "--log") && i + 1 < argc) {
      log_path = argv[++i];
    }
  }

  std::filesystem::create_directories(std::filesystem::path(log_path).parent_path());

  std::string host = "0.0.0.0";
  int port = 9090;
  {
    std::string s(listen_arg);
    auto p = s.rfind(':');
    if (p != std::string::npos) {
      host = s.substr(0, p);
      port = std::atoi(s.substr(p + 1).c_str());
    }
  }

  int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
  if (fd < 0) {
    perror("socket");
    return 1;
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = (host == "0.0.0.0") ? INADDR_ANY : inet_addr(host.c_str());
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    perror("bind");
    return 1;
  }
  if (listen(fd, 8) < 0) {
    perror("listen");
    return 1;
  }

  std::cerr << "integra-alarmd listening on " << host << ":" << port << " log=" << log_path << "\n";

  while (true) {
    int c = accept(fd, nullptr, nullptr);
    if (c < 0) {
      perror("accept");
      continue;
    }
    std::thread(handle_client, c, log_path).detach();
  }
}
