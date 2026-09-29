// mk-cli: small command-line client.
//
//   mk-cli [--broker host:port] create-topic <topic> <partitions>
//   mk-cli [--broker host:port] produce <topic> [key]        (one message per stdin line)
//   mk-cli [--broker host:port] consume <topic> <group> [max_messages]
//   mk-cli [--broker host:port] offsets <topic> <partition>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <string>

#include "minikafka/client.h"
#include "minikafka/consumer.h"
#include "minikafka/producer.h"

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  std::string host = "127.0.0.1";
  uint16_t port = 9092;
  int i = 1;
  if (i + 1 < argc && std::string(argv[i]) == "--broker") {
    std::string hp = argv[i + 1];
    auto c = hp.rfind(':');
    host = hp.substr(0, c);
    port = static_cast<uint16_t>(std::stoi(hp.substr(c + 1)));
    i += 2;
  }
  if (i >= argc) {
    std::fprintf(stderr, "usage: mk-cli [--broker host:port] create-topic|produce|consume|offsets ...\n");
    return 2;
  }
  const std::string cmd = argv[i++];
  auto arg = [&](int k) -> std::string { return i + k < argc ? argv[i + k] : ""; };

  try {
    mk::Client client(host, port);
    if (cmd == "create-topic") {
      auto e = client.create_topic(arg(0), static_cast<uint32_t>(std::stoul(arg(1))));
      std::printf("%s\n", mk::error_name(e));
      return e == mk::ErrorCode::None ? 0 : 1;
    }
    if (cmd == "produce") {
      mk::Producer producer(client);
      std::string line, key = arg(1);
      uint64_t n = 0;
      while (std::getline(std::cin, line)) {
        producer.send(arg(0), key, line);
        ++n;
      }
      producer.flush();
      std::fprintf(stderr, "produced %llu messages\n", static_cast<unsigned long long>(n));
      return 0;
    }
    if (cmd == "consume") {
      mk::Consumer consumer(client, {.group = arg(1)});
      consumer.subscribe({arg(0)});
      const uint64_t max = arg(2).empty() ? UINT64_MAX : std::stoull(arg(2));
      uint64_t n = 0;
      while (n < max) {
        auto recs = consumer.poll(1000);
        for (const auto& r : recs) {
          std::printf("offset=%llu key=%s value=%s\n", static_cast<unsigned long long>(r.offset),
                      r.key.c_str(), r.value.c_str());
          if (++n >= max) break;
        }
        consumer.commit_sync();
        if (recs.empty() && !arg(2).empty()) break;  // bounded mode stops when caught up
      }
      return 0;
    }
    if (cmd == "offsets") {
      auto [start, end] = client.list_offsets(arg(0), static_cast<uint32_t>(std::stoul(arg(1))));
      std::printf("log_start=%llu log_end=%llu\n", static_cast<unsigned long long>(start),
                  static_cast<unsigned long long>(end));
      return 0;
    }
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
