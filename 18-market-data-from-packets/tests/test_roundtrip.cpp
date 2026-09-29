// Randomized round trips: random messages -> encoder -> IEX-TP segments ->
// UDP/IPv4/Ethernet -> pcap or pcapng bytes -> CaptureReader -> FeedDecoder ->
// decoded structs -> encoder again. The re-encoded bytes must equal the
// originals, message for message, with the right sequence numbers.
#include <gtest/gtest.h>

#include <cstring>
#include <random>

#include "mdp/feed.hpp"
#include "mdp/normalizer.hpp"
#include "support/encoder.hpp"

using namespace mdp;
using namespace mdp::test;

namespace {

struct Recorder : NullHandler {
  std::vector<std::pair<int64_t, Bytes>> out;
  void on_system_event(const MsgContext& c, const SystemEvent& e) { out.emplace_back(c.seq, encode(e)); }
  void on_security_directory(const MsgContext& c, const SecurityDirectory& d) { out.emplace_back(c.seq, encode(d)); }
  void on_trading_status(const MsgContext& c, const TradingStatus& s) { out.emplace_back(c.seq, encode(s)); }
  void on_security_status(const MsgContext& c, const SecurityStatus& s) { out.emplace_back(c.seq, encode(s)); }
  void on_quote(const MsgContext& c, const QuoteUpdate& q) { out.emplace_back(c.seq, encode(q)); }
  void on_trade(const MsgContext& c, const Trade& t) { out.emplace_back(c.seq, encode(t)); }
  void on_official_price(const MsgContext& c, const OfficialPrice& p) { out.emplace_back(c.seq, encode(p)); }
  void on_price_level(const MsgContext& c, const PriceLevelUpdate& u) { out.emplace_back(c.seq, encode(u)); }
  void on_auction(const MsgContext& c, const AuctionInfo& a) { out.emplace_back(c.seq, encode(a)); }
};

class Gen {
 public:
  explicit Gen(uint64_t seed) : rng_(seed) {}
  uint64_t u() { return rng_(); }
  uint8_t pick(const char* s) { return static_cast<uint8_t>(s[u() % std::strlen(s)]); }
  int64_t ts() { return 1789729540000000000 + static_cast<int64_t>(u() % 30'000'000'000'000ull); }
  int64_t px() { return static_cast<int64_t>(u() % 50'000'000); }
  Symbol sym() {
    static const char* syms[] = {"AAPL", "MSFT", "SPY", "ZIEXT", "BRK.A", "QQQ", "ABCDEFGH", "T"};
    return make_symbol(syms[u() % 8]);
  }
  Bytes message() {
    switch (u() % 13) {
      case 0: return encode(SystemEvent{pick("OSRMEC"), ts()});
      case 1:
        return encode(SecurityDirectory{static_cast<uint8_t>(u()), ts(), sym(), static_cast<uint32_t>(u()), px(),
                                        static_cast<uint8_t>(u() % 3)});
      case 2: return encode(TradingStatus{pick("HOPT"), ts(), sym(), static_cast<uint32_t>(u())});
      case 3: return encode(SecurityStatus{'I', pick("ABC "), ts(), sym(), 0});
      case 4: return encode(SecurityStatus{'O', pick("ON"), ts(), sym(), 0});
      case 5: return encode(SecurityStatus{'P', static_cast<uint8_t>(u() % 2), ts(), sym(), pick(" ACDN")});
      case 6: return encode(SecurityStatus{'E', pick("OC"), ts(), sym(), 0});
      case 7:
        return encode(QuoteUpdate{static_cast<uint8_t>(u() & 0xc0), ts(), sym(), static_cast<uint32_t>(u()), px(),
                                  px(), static_cast<uint32_t>(u())});
      case 8:
        return encode(Trade{pick("TB"), static_cast<uint8_t>(u() & 0xf8), ts(), sym(),
                            static_cast<uint32_t>(u()), px(), static_cast<int64_t>(u() >> 1)});
      case 9: return encode(OfficialPrice{pick("QM"), ts(), sym(), px()});
      case 10:
        return encode(AuctionInfo{pick("OCIHV"), ts(), sym(), static_cast<uint32_t>(u()), px(), px(),
                                  static_cast<uint32_t>(u()), pick("BSN"), static_cast<uint8_t>(u()),
                                  static_cast<uint32_t>(u()), px(), px(), px(), px()});
      default:
        return encode(PriceLevelUpdate{static_cast<uint8_t>(u() % 2 ? '8' : '5'), static_cast<uint8_t>(u() % 2),
                                       ts(), sym(), static_cast<uint32_t>(u()), px()});
    }
  }

 private:
  std::mt19937_64 rng_;
};

Bytes make_capture(const std::vector<TimedFrame>& frames, uint64_t variant) {
  switch (variant % 4) {
    case 0: return write_pcap(frames, true, false);
    case 1: return write_pcap(frames, false, true);
    case 2: {
      PcapNgOptions o;
      o.tsresol = 9;
      o.sections = 2;
      return write_pcapng(frames, o);
    }
    default: {
      PcapNgOptions o;
      o.swap = true;
      return write_pcapng(frames, o);
    }
  }
}

}  // namespace

class RoundTrip : public ::testing::TestWithParam<uint64_t> {};

TEST_P(RoundTrip, RandomMessagesSurviveEveryLayer) {
  const uint64_t seed = GetParam();
  Gen g(seed);
  std::vector<Bytes> msgs;
  std::vector<TimedFrame> frames;
  SegmentSpec spec;
  spec.protocol_id = seed % 2 ? kProtoTops16 : kProtoDeep10;
  spec.first_seq = 1 + static_cast<int64_t>(g.u() % 1000);
  const int64_t seq0 = spec.first_seq;
  int n_segments = 300;
  for (int s = 0; s < n_segments; ++s) {
    std::vector<Bytes> batch;
    int k = static_cast<int>(g.u() % 12);  // 0 means heartbeat
    uint16_t payload = 0;
    for (int i = 0; i < k; ++i) {
      batch.push_back(g.message());
      payload = static_cast<uint16_t>(payload + 2 + batch.back().size());
    }
    spec.send_time = 1789729540000000000 + s * 1000;
    FrameSpec f;
    f.vlan_tags = static_cast<int>(g.u() % 2);
    frames.push_back({spec.send_time + 50, frame(build_segment(spec, batch), f)});
    for (auto& m : batch) msgs.push_back(std::move(m));
    spec.first_seq += k;
    spec.stream_offset += payload;
  }
  Bytes cap = make_capture(frames, seed);
  Recorder rec;
  FeedDecoder<Recorder> dec(rec);
  CaptureReader r(std::make_unique<MemorySource>(cap.data(), cap.size()));
  Packet p;
  while (r.next(p)) dec.on_packet(p);
  ASSERT_EQ(rec.out.size(), msgs.size());
  for (size_t i = 0; i < msgs.size(); ++i) {
    ASSERT_EQ(rec.out[i].first, seq0 + static_cast<int64_t>(i)) << i;
    ASSERT_EQ(rec.out[i].second, msgs[i]) << "message " << i << " type " << msgs[i][0];
  }
  const auto& st = dec.seq().sessions().begin()->second;
  EXPECT_EQ(st.gaps + st.duplicates + st.overlaps + st.offset_mismatches, 0u);
  EXPECT_EQ(dec.stats().framing_errors, 0u);
  EXPECT_EQ(dec.stats().packets, static_cast<uint64_t>(n_segments));
}

INSTANTIATE_TEST_SUITE_P(Seeds, RoundTrip, ::testing::Range<uint64_t>(1, 13));

// Replay the same stream with random drops and duplicated packets. Every
// dropped message must be counted as a gap, every repeat dropped, and the
// messages that do arrive must come out exactly once and in order.
TEST(RoundTripFaults, DropsAndDuplicatesAreAccounted) {
  Gen g(99);
  SegmentSpec spec;
  std::vector<std::pair<std::vector<Bytes>, Bytes>> segments;  // messages, segment bytes
  for (int s = 0; s < 500; ++s) {
    std::vector<Bytes> batch;
    int k = 1 + static_cast<int>(g.u() % 6);
    uint16_t payload = 0;
    for (int i = 0; i < k; ++i) {
      batch.push_back(g.message());
      payload = static_cast<uint16_t>(payload + 2 + batch.back().size());
    }
    segments.emplace_back(batch, build_segment(spec, batch));
    spec.first_seq += k;
    spec.stream_offset += payload;
  }
  std::vector<TimedFrame> frames;
  std::vector<Bytes> expected;
  uint64_t dropped_msgs = 0, dup_msgs = 0;
  bool last_dropped = false;
  for (size_t i = 0; i < segments.size(); ++i) {
    auto roll = g.u() % 10;
    if (roll == 0 && i + 1 < segments.size() && i > 0) {
      dropped_msgs += segments[i].first.size();
      last_dropped = true;
      continue;
    }
    frames.push_back({static_cast<int64_t>(i), frame(segments[i].second)});
    for (const auto& m : segments[i].first) expected.push_back(m);
    if (roll == 1 && !last_dropped) {
      frames.push_back({static_cast<int64_t>(i), frame(segments[i].second)});
      dup_msgs += segments[i].first.size();
    }
    last_dropped = false;
  }
  Bytes cap = write_pcap(frames, true, false);
  Recorder rec;
  FeedDecoder<Recorder> dec(rec);
  CaptureReader r(std::make_unique<MemorySource>(cap.data(), cap.size()));
  Packet p;
  while (r.next(p)) dec.on_packet(p);
  const auto& st = dec.seq().sessions().begin()->second;
  EXPECT_EQ(st.gap_messages, dropped_msgs);
  EXPECT_EQ(st.duplicate_messages, dup_msgs);
  ASSERT_EQ(rec.out.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) ASSERT_EQ(rec.out[i].second, expected[i]);
}

// The Normalizer (the handler used by mdp_decode) must emit one row per
// message into the right table and build the book from DEEP updates.
TEST(Normalizer, CountsRowsPerTableAndBuildsBbo) {
  Normalizer n("", true);
  FeedDecoder<Normalizer> dec(n);
  SegmentSpec spec;
  const Symbol s = make_symbol("SPY");
  std::vector<Bytes> batch = {
      encode(PriceLevelUpdate{'8', 1, 10, s, 100, 5000000}),
      encode(PriceLevelUpdate{'5', 0, 11, s, 200, 5000100}),
      encode(PriceLevelUpdate{'5', 1, 11, s, 300, 5000200}),
      encode(Trade{'T', 0, 12, s, 50, 5000100, 77}),
      encode(QuoteUpdate{0, 13, s, 100, 5000000, 5000100, 200}),
  };
  Bytes seg = build_segment(spec, batch);
  dec.on_udp(seg.data(), static_cast<uint32_t>(seg.size()), 0);
  EXPECT_EQ(n.bbo_rows(), 2u);  // after seq 1 (bid only) and after seq 3 (two sided)
  const auto& b = n.books().book_by_id(0).bbo();
  EXPECT_EQ(b, (Bbo{5000000, 100, 5000100, 200}));
}
