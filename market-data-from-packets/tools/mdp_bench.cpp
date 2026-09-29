// mdp_bench: single-thread decode throughput on an in-memory capture.
//
//   mdp_bench INPUT [--reps N]
//
// INPUT is read fully into memory first (gzip is decompressed up front), so
// the timed loops measure parsing only, not disk or zlib. Three loops:
//   decode      pcap/pcapng framing + Ethernet/IPv4/UDP + IEX-TP + typed
//               message decode, folded into a checksum so nothing is elided
//   decode_book the same plus the per-symbol DEEP book and BBO emission
//   normalize   decode + book + building every output row (writers disabled)
// Prints one JSON object on stdout.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "mdp/book.hpp"
#include "mdp/feed.hpp"
#include "mdp/normalizer.hpp"
#include "mdp/pcap.hpp"

using namespace mdp;

namespace {

struct ChecksumHandler : NullHandler {
  uint64_t acc = 0;
  void on_quote(const MsgContext& c, const QuoteUpdate& q) {
    acc += static_cast<uint64_t>(q.ts ^ q.bid_price ^ q.ask_price) + q.bid_size + q.ask_size + c.seq;
  }
  void on_trade(const MsgContext& c, const Trade& t) {
    acc += static_cast<uint64_t>(t.ts ^ t.price ^ t.trade_id) + t.size + c.seq;
  }
  void on_price_level(const MsgContext& c, const PriceLevelUpdate& u) {
    acc += static_cast<uint64_t>(u.ts ^ u.price) + u.size + u.flags + c.seq;
  }
  void on_system_event(const MsgContext&, const SystemEvent& e) { acc += e.event; }
  void on_security_directory(const MsgContext&, const SecurityDirectory& d) { acc += d.round_lot; }
  void on_trading_status(const MsgContext&, const TradingStatus& s) { acc += s.status; }
  void on_security_status(const MsgContext&, const SecurityStatus& s) { acc += s.status; }
  void on_official_price(const MsgContext&, const OfficialPrice& p) { acc += static_cast<uint64_t>(p.price); }
  void on_auction(const MsgContext&, const AuctionInfo& a) { acc += a.paired_shares; }
};

struct BookHandler : ChecksumHandler {
  BookManager books;
  uint64_t bbo_changes = 0;
  void on_price_level(const MsgContext& c, const PriceLevelUpdate& u) {
    ChecksumHandler::on_price_level(c, u);
    if (books.book(u.symbol).apply(u)) ++bbo_changes;
  }
};

std::vector<uint8_t> slurp(const std::string& path) {
  auto src = open_source(path);
  std::vector<uint8_t> out;
  const size_t chunk = 64 << 20;
  for (;;) {
    size_t have = 0;
    const uint8_t* p = nullptr;
    // Grab as much as is available up to chunk.
    size_t want = chunk;
    while (want > 0 && !(p = src->ensure(want))) want /= 2;
    if (!p) {
      if ((p = src->ensure(1))) want = 1;
      else break;
    }
    have = want;
    out.insert(out.end(), p, p + have);
    src->consume(have);
  }
  return out;
}

template <class H>
double run(const std::vector<uint8_t>& buf, H& h, uint64_t& msgs, uint64_t& packets) {
  auto t0 = std::chrono::steady_clock::now();
  CaptureReader r(std::make_unique<MemorySource>(buf.data(), buf.size()));
  FeedDecoder<H> dec(h);
  Packet p;
  while (r.next(p)) dec.on_packet(p);
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  msgs = dec.stats().messages;
  packets = dec.stats().packets;
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: mdp_bench INPUT [--reps N]\n");
    return 2;
  }
  std::string in = argv[1];
  int reps = 3;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::string(argv[i]) == "--reps") reps = std::stoi(argv[i + 1]);
  }
  auto t0 = std::chrono::steady_clock::now();
  std::vector<uint8_t> buf = slurp(in);
  double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  uint64_t msgs = 0, packets = 0, sink = 0;
  double best_decode = 1e30, best_book = 1e30, best_norm = 1e30;
  uint64_t bbo = 0;
  for (int r = 0; r < reps; ++r) {
    ChecksumHandler h;
    best_decode = std::min(best_decode, run(buf, h, msgs, packets));
    sink ^= h.acc;
  }
  for (int r = 0; r < reps; ++r) {
    BookHandler h;
    best_book = std::min(best_book, run(buf, h, msgs, packets));
    sink ^= h.acc;
    bbo = h.bbo_changes;
  }
  for (int r = 0; r < reps; ++r) {
    Normalizer n("", true);
    best_norm = std::min(best_norm, run(buf, n, msgs, packets));
    sink ^= n.bbo_rows();
  }
  const double mb = buf.size() / 1e6;
  std::printf(
      "{\"input\": \"%s\", \"bytes\": %zu, \"packets\": %llu, \"messages\": %llu, \"reps\": %d, "
      "\"load_seconds\": %.3f, "
      "\"decode_seconds\": %.4f, \"decode_msgs_per_s\": %.0f, \"decode_mb_per_s\": %.1f, "
      "\"decode_book_seconds\": %.4f, \"decode_book_msgs_per_s\": %.0f, \"decode_book_mb_per_s\": %.1f, "
      "\"normalize_seconds\": %.4f, \"normalize_msgs_per_s\": %.0f, \"normalize_mb_per_s\": %.1f, "
      "\"bbo_changes\": %llu, \"checksum\": %llu}\n",
      in.c_str(), buf.size(), static_cast<unsigned long long>(packets), static_cast<unsigned long long>(msgs), reps,
      load_s, best_decode, msgs / best_decode, mb / best_decode, best_book, msgs / best_book, mb / best_book,
      best_norm, msgs / best_norm, mb / best_norm, static_cast<unsigned long long>(bbo),
      static_cast<unsigned long long>(sink));
  return 0;
}
