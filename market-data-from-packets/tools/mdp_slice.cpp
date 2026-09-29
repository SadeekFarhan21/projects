// mdp_slice: copy packets from a capture into a classic nanosecond pcap.
//
//   mdp_slice INPUT OUTPUT [--from-ns NS] [--until-ns NS] [--max-packets N] [--max-bytes N]
//
// Used to cut the checked-in test fixture and the benchmark input out of
// full-day HIST files. Packets are copied byte for byte.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "mdp/pcap.hpp"
#include "mdp/pcap_writer.hpp"

using namespace mdp;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: mdp_slice INPUT OUTPUT [--from-ns NS] [--until-ns NS] [--max-packets N] [--max-bytes N]\n");
    return 2;
  }
  std::string in = argv[1], out = argv[2];
  int64_t from = INT64_MIN, until = INT64_MAX;
  uint64_t max_packets = UINT64_MAX, max_bytes = UINT64_MAX;
  for (int i = 3; i + 1 < argc; i += 2) {
    std::string a = argv[i];
    if (a == "--from-ns") from = std::stoll(argv[i + 1]);
    else if (a == "--until-ns") until = std::stoll(argv[i + 1]);
    else if (a == "--max-packets") max_packets = std::stoull(argv[i + 1]);
    else if (a == "--max-bytes") max_bytes = std::stoull(argv[i + 1]);
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  try {
    CaptureReader r(open_source(in));
    Packet p;
    std::unique_ptr<PcapWriter> w;
    uint64_t bytes = 24;
    while (r.next(p)) {
      if (p.ts_ns < from) continue;
      if (p.ts_ns > until) break;
      if (!w) w = std::make_unique<PcapWriter>(out, p.link_type, true);
      if (w->packets() >= max_packets || bytes + 16 + p.caplen > max_bytes) break;
      w->write(p.ts_ns, p.data, p.caplen, p.origlen);
      bytes += 16 + p.caplen;
    }
    std::fprintf(stderr, "mdp_slice: wrote %llu packets, %llu bytes\n",
                 static_cast<unsigned long long>(w ? w->packets() : 0), static_cast<unsigned long long>(bytes));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mdp_slice: %s\n", e.what());
    return 1;
  }
  return 0;
}
