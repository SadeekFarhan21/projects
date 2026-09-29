// mk-broker: runs a single broker until SIGINT/SIGTERM.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#include "minikafka/broker.h"

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

void usage() {
  std::fprintf(stderr,
               "usage: mk-broker [--data-dir DIR] [--host H] [--port P] [--segment-bytes N]\n"
               "                 [--index-interval N] [--retention-bytes N] [--flush none|fsync|full]\n");
}
}  // namespace

int main(int argc, char** argv) {
  mk::BrokerConfig cfg;
  cfg.port = 9092;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--data-dir") cfg.data_dir = next();
    else if (a == "--host") cfg.host = next();
    else if (a == "--port") cfg.port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--segment-bytes") cfg.log.segment_bytes = std::stoull(next());
    else if (a == "--index-interval") cfg.log.index_interval_bytes = static_cast<uint32_t>(std::stoul(next()));
    else if (a == "--retention-bytes") cfg.log.retention_bytes = std::stoll(next());
    else if (a == "--flush") {
      std::string f = next();
      cfg.log.flush = f == "fsync" ? mk::FlushPolicy::Fsync
                      : f == "full" ? mk::FlushPolicy::FullFsync
                                    : mk::FlushPolicy::None;
    } else {
      usage();
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  mk::Broker broker(cfg);
  broker.start();
  std::printf("mk-broker listening on %s:%u, data dir %s\n", cfg.host.c_str(), broker.port(),
              cfg.data_dir.c_str());
  std::fflush(stdout);
  while (!g_stop) pause();
  std::printf("shutting down\n");
  broker.stop();
  return 0;
}
