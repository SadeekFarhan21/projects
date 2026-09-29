// mk-bench: throughput and latency experiments against an in-process broker
// reached over loopback TCP. Every mode prints CSV to stdout.
//
//   mk-bench produce  [--reps N]   producer throughput vs batch size x message size
//   mk-bench consume  [--reps N]   consumer throughput vs fetch size x message size
//   mk-bench latency               end-to-end latency percentiles (+ raw samples to --raw FILE)
//   mk-bench flush    [--reps N]   producer throughput vs flush policy x batch size
//   mk-bench scaling  [--reps N]   aggregate producer throughput vs producer count
//
// The broker's data directory is a fresh temp dir per run (removed afterwards).
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "minikafka/broker.h"
#include "minikafka/client.h"
#include "minikafka/producer.h"
#include "minikafka/record.h"

using namespace mk;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace {

// 1-minute load average, recorded next to every result because this machine
// was shared with other heavy jobs while the benchmarks ran.
double load1() {
  double l[3] = {0, 0, 0};
  getloadavg(l, 3);
  return l[0];
}

double secs(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Nearest-rank percentile of a sorted vector.
double pct(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return NAN;
  size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * sorted.size()));
  idx = std::clamp<size_t>(idx, 1, sorted.size()) - 1;
  return sorted[idx];
}

std::string random_payload(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::string s(n, '\0');
  for (auto& c : s) c = static_cast<char>('a' + rng() % 26);
  return s;
}

struct BenchBroker {
  fs::path dir;
  std::unique_ptr<Broker> broker;
  explicit BenchBroker(FlushPolicy flush = FlushPolicy::None) {
    dir = fs::temp_directory_path() / ("minikafka-bench-" + std::to_string(::getpid()));
    fs::remove_all(dir);
    BrokerConfig cfg;
    cfg.data_dir = dir;
    cfg.log.segment_bytes = 256ull << 20;
    cfg.log.flush = flush;
    broker = std::make_unique<Broker>(cfg);
    broker->start();
  }
  ~BenchBroker() {
    broker->stop();
    broker.reset();
    fs::remove_all(dir);
  }
  uint16_t port() const { return broker->port(); }
};

struct ThroughputResult {
  uint64_t msgs;
  double seconds;
};

// Sends up to `max_msgs` messages of `msg_size` bytes to topic/partition in
// batches of `batch`, stopping early after `time_cap` seconds.
ThroughputResult produce_run(uint16_t port, const std::string& topic, uint32_t partition,
                             size_t msg_size, uint32_t batch, uint64_t max_msgs, double time_cap) {
  Client c("127.0.0.1", port);
  const std::string payload = random_payload(msg_size, 1234);
  // Pre-encode one full batch and resend it: this measures the broker and the
  // transport, not the client's record encoding (which is timed separately in
  // the consume benchmark via parse_records).
  std::vector<uint8_t> buf;
  for (uint32_t i = 0; i < batch; ++i) append_record(buf, 0, "", payload);
  uint64_t sent = 0;
  const auto t0 = Clock::now();
  while (sent < max_msgs) {
    c.produce(topic, partition, buf, batch);
    sent += batch;
    if ((sent / batch) % 16 == 0 && secs(Clock::now() - t0) > time_cap) break;
  }
  return {sent, secs(Clock::now() - t0)};
}

int arg_int(int argc, char** argv, const char* name, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  return def;
}
std::string arg_str(int argc, char** argv, const char* name, const std::string& def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  return def;
}

const char* flush_name(FlushPolicy f) {
  switch (f) {
    case FlushPolicy::None: return "none";
    case FlushPolicy::Fsync: return "fsync";
    case FlushPolicy::FullFsync: return "full_fsync";
  }
  return "?";
}

void bench_produce(int reps) {
  std::printf("msg_size,batch,reps,msgs,msgs_per_s_median,mb_per_s_median,mb_per_s_min,mb_per_s_max,load1\n");
  BenchBroker bb;
  Client admin("127.0.0.1", bb.port());
  int topic_id = 0;
  for (size_t msg : {100, 1000, 10000}) {
    for (uint32_t batch : {1u, 10u, 100u, 1000u}) {
      std::vector<double> mbps, mps;
      uint64_t msgs = 0;
      for (int r = 0; r < reps; ++r) {
        const std::string topic = "p" + std::to_string(topic_id++);
        admin.create_topic(topic, 1);
        const uint64_t cap = std::max<uint64_t>(batch, (256ull << 20) / msg);
        auto res = produce_run(bb.port(), topic, 0, msg, batch, std::min<uint64_t>(cap, 1000000), 3.0);
        msgs = res.msgs;
        mps.push_back(res.msgs / res.seconds);
        mbps.push_back(res.msgs * msg / res.seconds / 1e6);
      }
      std::printf("%zu,%u,%d,%llu,%.0f,%.1f,%.1f,%.1f,%.0f\n", msg, batch, reps,
                  static_cast<unsigned long long>(msgs), median(mps), median(mbps),
                  *std::min_element(mbps.begin(), mbps.end()), *std::max_element(mbps.begin(), mbps.end()), load1());
      std::fflush(stdout);
    }
  }
}

void bench_consume(int reps) {
  std::printf("msg_size,fetch_bytes,reps,msgs,msgs_per_s_median,mb_per_s_median,mb_per_s_min,mb_per_s_max,load1\n");
  BenchBroker bb;
  Client admin("127.0.0.1", bb.port());
  for (size_t msg : {100, 1000, 10000}) {
    const std::string topic = "c" + std::to_string(msg);
    admin.create_topic(topic, 1);
    const uint64_t total = std::min<uint64_t>((256ull << 20) / msg, 1000000);
    const uint32_t batch = std::min<uint64_t>(1000, (1u << 20) / msg);
    produce_run(bb.port(), topic, 0, msg, batch, total, 1e9);
    const uint64_t end = admin.list_offsets(topic, 0).second;
    for (uint32_t fetch_bytes : {4u << 10, 16u << 10, 64u << 10, 256u << 10, 1u << 20, 4u << 20}) {
      std::vector<double> mbps, mps;
      uint64_t msgs = 0;
      for (int r = 0; r < reps; ++r) {
        Client c("127.0.0.1", bb.port());
        uint64_t pos = 0, bytes = 0;
        msgs = 0;
        const auto t0 = Clock::now();
        while (pos < end) {
          auto resp = c.fetch({{topic, 0, pos, fetch_bytes}}, 0);
          // parse_records verifies every CRC, as a real consumer would.
          auto recs = parse_records(resp[0].records.data(), resp[0].records.size());
          if (recs.empty()) break;
          for (const auto& rv : recs) bytes += rv.value.size();
          msgs += recs.size();
          pos = recs.back().offset + 1;
        }
        const double s = secs(Clock::now() - t0);
        mps.push_back(msgs / s);
        mbps.push_back(bytes / s / 1e6);
      }
      std::printf("%zu,%u,%d,%llu,%.0f,%.1f,%.1f,%.1f,%.0f\n", msg, fetch_bytes, reps,
                  static_cast<unsigned long long>(msgs), median(mps), median(mbps),
                  *std::min_element(mbps.begin(), mbps.end()), *std::max_element(mbps.begin(), mbps.end()), load1());
      std::fflush(stdout);
    }
  }
}

// One latency configuration: a producer sends `count` messages at `rate`/s in
// batches of `batch`, each value carrying its send time; a consumer long-polls
// and records receive time minus send time.
std::vector<double> latency_run(FlushPolicy flush, uint32_t rate, uint32_t batch, uint32_t count,
                                size_t msg_size, std::vector<double>* ack_us) {
  BenchBroker bb(flush);
  Client admin("127.0.0.1", bb.port());
  admin.create_topic("lat", 1);
  std::vector<double> lat_us;
  lat_us.reserve(count);
  std::atomic<bool> done{false};

  std::thread consumer([&] {
    Client c("127.0.0.1", bb.port());
    uint64_t pos = 0;
    while (lat_us.size() < count) {
      auto resp = c.fetch({{"lat", 0, pos, 1 << 20}}, 100);
      const auto now = Clock::now().time_since_epoch().count();
      for (const auto& rv : parse_records(resp[0].records.data(), resp[0].records.size())) {
        int64_t sent;
        std::memcpy(&sent, rv.value.data(), 8);
        lat_us.push_back((now - sent) / 1000.0);
        pos = rv.offset + 1;
      }
      if (done && resp[0].records.empty()) break;
    }
  });

  Client c("127.0.0.1", bb.port());
  std::string payload = random_payload(std::max<size_t>(msg_size, 8), 99);
  const auto interval = std::chrono::nanoseconds(1'000'000'000ull * batch / rate);
  auto next = Clock::now();
  std::vector<uint8_t> buf;
  for (uint32_t sent = 0; sent < count; sent += batch) {
    // Pace against a fixed schedule; sleep for most of the gap, spin the rest.
    while (Clock::now() < next) {
      if (next - Clock::now() > std::chrono::microseconds(200)) std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    next += interval;
    buf.clear();
    for (uint32_t i = 0; i < batch; ++i) {
      const int64_t t = Clock::now().time_since_epoch().count();
      std::memcpy(payload.data(), &t, 8);
      append_record(buf, 0, "", payload);
    }
    const auto a0 = Clock::now();
    c.produce("lat", 0, buf, batch);
    if (ack_us) ack_us->push_back(secs(Clock::now() - a0) * 1e6);
  }
  done = true;
  consumer.join();
  std::sort(lat_us.begin(), lat_us.end());
  return lat_us;
}

void bench_latency(const std::string& raw_path) {
  std::printf("flush,rate_msgs_per_s,batch,msg_size,count,p50_us,p90_us,p99_us,p999_us,max_us,ack_p50_us,ack_p99_us,load1\n");
  std::ofstream raw;
  if (!raw_path.empty()) {
    raw.open(raw_path);
    raw << "flush,rate_msgs_per_s,batch,latency_us\n";
  }
  struct Cfg {
    FlushPolicy flush;
    uint32_t rate, batch, count;
  };
  const std::vector<Cfg> cfgs = {
      {FlushPolicy::None, 1000, 1, 10000},
      {FlushPolicy::None, 10000, 1, 50000},
      {FlushPolicy::None, 50000, 10, 200000},
      {FlushPolicy::Fsync, 1000, 1, 10000},
      {FlushPolicy::FullFsync, 200, 1, 2000},
  };
  for (const auto& cfg : cfgs) {
    std::vector<double> ack;
    auto lat = latency_run(cfg.flush, cfg.rate, cfg.batch, cfg.count, 100, &ack);
    std::sort(ack.begin(), ack.end());
    std::printf("%s,%u,%u,100,%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f\n", flush_name(cfg.flush), cfg.rate,
                cfg.batch, lat.size(), pct(lat, 50), pct(lat, 90), pct(lat, 99), pct(lat, 99.9),
                lat.empty() ? NAN : lat.back(), pct(ack, 50), pct(ack, 99), load1());
    std::fflush(stdout);
    if (raw.is_open())
      for (double v : lat) raw << flush_name(cfg.flush) << "," << cfg.rate << "," << cfg.batch << "," << v << "\n";
  }
}

void bench_flush(int reps) {
  std::printf("flush,msg_size,batch,reps,msgs,msgs_per_s_median,mb_per_s_median,load1\n");
  for (FlushPolicy f : {FlushPolicy::None, FlushPolicy::Fsync, FlushPolicy::FullFsync}) {
    BenchBroker bb(f);
    Client admin("127.0.0.1", bb.port());
    int topic_id = 0;
    for (uint32_t batch : {1u, 10u, 100u, 1000u}) {
      std::vector<double> mps, mbps;
      uint64_t msgs = 0;
      for (int r = 0; r < reps; ++r) {
        const std::string topic = "f" + std::to_string(topic_id++);
        admin.create_topic(topic, 1);
        auto res = produce_run(bb.port(), topic, 0, 1000, batch, 200000, 2.0);
        msgs = res.msgs;
        mps.push_back(res.msgs / res.seconds);
        mbps.push_back(res.msgs * 1000.0 / res.seconds / 1e6);
      }
      std::printf("%s,1000,%u,%d,%llu,%.0f,%.1f,%.0f\n", flush_name(f), batch, reps,
                  static_cast<unsigned long long>(msgs), median(mps), median(mbps), load1());
      std::fflush(stdout);
    }
  }
}

void bench_scaling(int reps) {
  std::printf("producers,msg_size,batch,reps,msgs_per_s_median,mb_per_s_median,load1\n");
  BenchBroker bb;
  Client admin("127.0.0.1", bb.port());
  int topic_id = 0;
  for (int producers : {1, 2, 4, 8}) {
    std::vector<double> mps, mbps;
    for (int r = 0; r < reps; ++r) {
      const std::string topic = "s" + std::to_string(topic_id++);
      admin.create_topic(topic, static_cast<uint32_t>(producers));
      std::vector<ThroughputResult> res(producers);
      std::vector<std::thread> ts;
      const auto t0 = Clock::now();
      for (int p = 0; p < producers; ++p)
        ts.emplace_back([&, p] { res[p] = produce_run(bb.port(), topic, p, 1000, 100, 400000, 2.0); });
      for (auto& t : ts) t.join();
      const double s = secs(Clock::now() - t0);
      uint64_t total = 0;
      for (auto& x : res) total += x.msgs;
      mps.push_back(total / s);
      mbps.push_back(total * 1000.0 / s / 1e6);
    }
    std::printf("%d,1000,100,%d,%.0f,%.1f,%.0f\n", producers, reps, median(mps), median(mbps), load1());
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mk-bench produce|consume|latency|flush|scaling [--reps N] [--raw FILE]\n");
    return 2;
  }
  const std::string mode = argv[1];
  const int reps = arg_int(argc, argv, "--reps", 3);
  if (mode == "produce") bench_produce(reps);
  else if (mode == "consume") bench_consume(reps);
  else if (mode == "latency") bench_latency(arg_str(argc, argv, "--raw", ""));
  else if (mode == "flush") bench_flush(reps);
  else if (mode == "scaling") bench_scaling(reps);
  else {
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  }
  return 0;
}
