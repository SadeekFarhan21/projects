// mdp_decode: decode an IEX HIST capture (TOPS or DEEP) into normalized tables.
//
//   mdp_decode INPUT --out DIR [--no-book] [--until-ns NS] [--max-packets N] [--quiet]
//
// INPUT may be .pcap, .pcapng, gzip of either, or "-" for stdin. Tables are
// written as packed binary rows (DIR/<table>.bin) plus DIR/manifest.json;
// python/mdq/to_parquet.py turns them into Parquet.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "mdp/feed.hpp"
#include "mdp/normalizer.hpp"
#include "mdp/pcap.hpp"

using namespace mdp;

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: mdp_decode INPUT --out DIR [--no-book] [--until-ns NS] [--max-packets N] [--quiet]\n");
  std::exit(2);
}

// Forwards segment callbacks to the normalizer.
struct SegForward {
  Normalizer* n;
  void on_segment(const SegmentHeader& h, int64_t cts, uint16_t skip) { n->on_segment(h, cts, skip); }
};

std::string feed_name(const DecodeStats& s, const SeqTracker& seq) {
  (void)s;
  for (const auto& [k, v] : seq.sessions()) {
    switch (v.protocol_id) {
      case kProtoTops16: return "TOPS1.6";
      case kProtoTops15: return "TOPS1.5";
      case kProtoDeep10: return "DEEP1.0";
      case kProtoDeepPlus: return "DEEP+";
      default: return "unknown";
    }
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, out;
  bool book = true, quiet = false;
  int64_t until_ns = INT64_MAX;
  uint64_t max_packets = UINT64_MAX;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--out") out = val();
    else if (a == "--no-book") book = false;
    else if (a == "--until-ns") until_ns = std::stoll(val());
    else if (a == "--max-packets") max_packets = std::stoull(val());
    else if (a == "--quiet") quiet = true;
    else if (a == "-h" || a == "--help") usage();
    else if (input.empty()) input = a;
    else usage();
  }
  if (input.empty() || out.empty()) usage();
  std::filesystem::create_directories(out);

  try {
    CaptureReader reader(open_source(input));
    Normalizer norm(out, book);
    SegForward fwd{&norm};
    FeedDecoder<Normalizer, SegForward> dec(norm, &fwd);
    auto t0 = std::chrono::steady_clock::now();
    Packet pkt;
    uint64_t n = 0;
    while (n < max_packets && reader.next(pkt)) {
      if (pkt.ts_ns > until_ns) break;
      dec.on_packet(pkt);
      ++n;
      if (!quiet && (n & ((1u << 24) - 1)) == 0) {
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[mdp_decode] %llu packets, %llu msgs, %.1f GB in, %.0fs\n",
                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(dec.stats().messages),
                     reader.bytes_consumed() / 1e9, s);
      }
    }
    norm.close();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string fmt = reader.format() == CaptureFormat::PcapNg ? "pcapng" : "pcap";
    norm.write_manifest(out + "/manifest.json", dec, feed_name(dec.stats(), dec.seq()), input, secs,
                        reader.bytes_consumed(), reader.truncated(), fmt);
    if (!quiet) {
      std::fprintf(stderr,
                   "[mdp_decode] done: %llu packets, %llu segments, %llu msgs, %.2f GB capture, %.1fs, "
                   "%.2f M msgs/s, seq anomalies %llu, truncated=%d\n",
                   static_cast<unsigned long long>(dec.stats().packets),
                   static_cast<unsigned long long>(dec.stats().segments),
                   static_cast<unsigned long long>(dec.stats().messages), reader.bytes_consumed() / 1e9, secs,
                   dec.stats().messages / secs / 1e6, static_cast<unsigned long long>(dec.seq().anomaly_count()),
                   reader.truncated() ? 1 : 0);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mdp_decode: %s\n", e.what());
    return 1;
  }
  return 0;
}
