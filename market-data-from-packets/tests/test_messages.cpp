// Decoding of the byte examples printed in the IEX TOPS v1.66, DEEP v1.08 and
// IEX-TP v1 specifications. Each hex string is copied from the spec example.
#include <gtest/gtest.h>

#include <vector>

#include "mdp/feed.hpp"
#include "mdp/iex_messages.hpp"
#include "mdp/iextp.hpp"
#include "support/encoder.hpp"

using namespace mdp;
using mdp::test::hex;

namespace {
const Symbol kZiext = make_symbol("ZIEXT");
constexpr int64_t kTs20160823_153032 = 1471980632572715948;  // 2016-08-23 15:30:32.572715948 ET
}  // namespace

TEST(Symbol, PackAndTrim) {
  EXPECT_EQ(symbol_str(kZiext), "ZIEXT");
  auto b = hex("5a 49 45 58 54 20 20 20");
  EXPECT_EQ(load_le<uint64_t>(b.data()), kZiext);
  EXPECT_EQ(symbol_str(make_symbol("BRK.A")), "BRK.A");
  EXPECT_EQ(symbol_str(make_symbol("ABCDEFGH")), "ABCDEFGH");
}

TEST(SpecExample, SystemEvent) {
  auto m = hex("53 45 00 a0 99 97 e9 3d b6 14");
  ASSERT_EQ(m.size(), msg::min_length('S'));
  auto e = decode_system_event(m.data());
  EXPECT_EQ(e.event, 'E');  // End of System Hours
  EXPECT_EQ(e.ts, 1492448400000000000);  // 2017-04-17 17:00:00 (the spec prints the raw clock)
}

TEST(SpecExample, SecurityDirectory) {
  auto m = hex("44 80 00 20 89 7b 5a 1f b6 14 5a 49 45 58 54 20 20 20 64 00 00 00 24 1d 0f 00 00 00 00 00 01");
  ASSERT_EQ(m.size(), 31u);
  auto d = decode_security_directory(m.data());
  EXPECT_EQ(d.flags, 0x80);  // test security
  EXPECT_EQ(d.ts, 1492414800000000000);
  EXPECT_EQ(d.symbol, kZiext);
  EXPECT_EQ(d.round_lot, 100u);
  EXPECT_EQ(d.adjusted_poc_price, 990500);  // $99.05
  EXPECT_EQ(d.luld_tier, 1);
}

TEST(SpecExample, TradingStatus) {
  auto m = hex("48 48 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20 54 31 20 20");
  ASSERT_EQ(m.size(), 22u);
  auto s = decode_trading_status(m.data());
  EXPECT_EQ(s.status, 'H');
  EXPECT_EQ(s.ts, kTs20160823_153032);
  EXPECT_EQ(s.symbol, kZiext);
  char reason[5] = {};
  std::memcpy(reason, &s.reason, 4);
  EXPECT_STREQ(reason, "T1  ");
}

TEST(SpecExample, RetailLiquidityIndicator) {
  auto m = hex("49 41 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20");
  auto s = decode_security_status(m.data(), static_cast<uint16_t>(m.size()));
  EXPECT_EQ(s.type, 'I');
  EXPECT_EQ(s.status, 'A');
  EXPECT_EQ(s.ts, kTs20160823_153032);
  EXPECT_EQ(s.symbol, kZiext);
}

TEST(SpecExample, OperationalHalt) {
  auto m = hex("4f 4f ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20");
  auto s = decode_security_status(m.data(), static_cast<uint16_t>(m.size()));
  EXPECT_EQ(s.type, 'O');
  EXPECT_EQ(s.status, 'O');
  EXPECT_EQ(s.detail, 0);
}

TEST(SpecExample, ShortSalePriceTest) {
  auto m = hex("50 01 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20 41");
  ASSERT_EQ(m.size(), 19u);
  auto s = decode_security_status(m.data(), static_cast<uint16_t>(m.size()));
  EXPECT_EQ(s.type, 'P');
  EXPECT_EQ(s.status, 1);
  EXPECT_EQ(s.detail, 'A');
}

TEST(SpecExample, SecurityEvent) {
  auto m = hex("45 4f 00 f0 30 2a 5b 25 b6 14 5a 49 45 58 54 20 20 20");
  auto s = decode_security_status(m.data(), static_cast<uint16_t>(m.size()));
  EXPECT_EQ(s.type, 'E');
  EXPECT_EQ(s.status, 'O');  // Opening Process Complete
  EXPECT_EQ(s.ts, 1492421400000000000);
}

TEST(SpecExample, QuoteUpdateTops) {
  auto m = hex(
      "51 00 ac 63 c0 20 96 86 6d 14 5A 49 45 58 54 20 20 20 e4 25 00 00 24 1d 0f 00 00 00 00 00 "
      "ec 1d 0f 00 00 00 00 00 e8 03 00 00");
  ASSERT_EQ(m.size(), 42u);
  auto q = decode_quote(m.data());
  EXPECT_EQ(q.flags, 0);
  EXPECT_EQ(q.ts, kTs20160823_153032);
  EXPECT_EQ(q.symbol, kZiext);
  EXPECT_EQ(q.bid_size, 9700u);
  EXPECT_EQ(q.bid_price, 990500);
  EXPECT_EQ(q.ask_price, 990700);
  EXPECT_EQ(q.ask_size, 1000u);
}

TEST(SpecExample, PriceLevelUpdateBuy) {
  auto m = hex("38 01 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20 e4 25 00 00 24 1d 0f 00 00 00 00 00");
  ASSERT_EQ(m.size(), 30u);
  auto u = decode_price_level(m.data());
  EXPECT_EQ(u.side, '8');
  EXPECT_EQ(u.flags, kEventProcessingComplete);
  EXPECT_EQ(u.size, 9700u);
  EXPECT_EQ(u.price, 990500);
}

TEST(SpecExample, TradeReport) {
  auto m = hex(
      "54 00 c3 df f7 05 a2 86 6d 14 5a 49 45 58 54 20 20 20 64 00 00 00 24 1d 0f 00 00 00 00 00 "
      "96 8f 06 00 00 00 00 00");
  ASSERT_EQ(m.size(), 38u);
  auto t = decode_trade(m.data());
  EXPECT_EQ(t.type, 'T');
  EXPECT_EQ(t.flags, 0);
  EXPECT_EQ(t.ts, 1471980683662974915);
  EXPECT_EQ(t.size, 100u);
  EXPECT_EQ(t.price, 990500);
  EXPECT_EQ(t.trade_id, 429974);
}

TEST(SpecExample, TradeBreak) {
  auto m = hex(
      "42 00 b2 8f a5 a0 ab 86 6d 14 5a 49 45 58 54 20 20 20 64 00 00 00 24 1d 0f 00 00 00 00 00 "
      "96 8f 06 00 00 00 00 00");
  auto t = decode_trade(m.data());
  EXPECT_EQ(t.type, 'B');
  EXPECT_EQ(t.ts, 1471980724912754610);
  EXPECT_EQ(t.trade_id, 429974);
}

TEST(SpecExample, OfficialPrice) {
  auto m = hex("58 51 00 f0 30 2a 5b 25 b6 14 5a 49 45 58 54 20 20 20 24 1d 0f 00 00 00 00 00");
  ASSERT_EQ(m.size(), 26u);
  auto p = decode_official_price(m.data());
  EXPECT_EQ(p.price_type, 'Q');
  EXPECT_EQ(p.price, 990500);
}

TEST(SpecExample, AuctionInformation) {
  auto m = hex(
      "41 43 dd c7 f0 9a 1a 3a b6 14 5a 49 45 58 54 20 20 20 a0 86 01 00 24 1d 0f 00 00 00 00 00 "
      "18 1f 0f 00 00 00 00 00 10 27 00 00 42 00 80 e6 f4 58 0c 21 0f 00 00 00 00 00 "
      "c0 1c 0f 00 00 00 00 00 a4 99 0d 00 00 00 00 00 dc 9f 10 00 00 00 00 00");
  ASSERT_EQ(m.size(), 80u);
  auto a = decode_auction(m.data());
  EXPECT_EQ(a.auction_type, 'C');
  EXPECT_EQ(a.ts, 1492444212462929885);
  EXPECT_EQ(a.paired_shares, 100000u);
  EXPECT_EQ(a.reference_price, 990500);
  EXPECT_EQ(a.indicative_clearing_price, 991000);
  EXPECT_EQ(a.imbalance_shares, 10000u);
  EXPECT_EQ(a.imbalance_side, 'B');
  EXPECT_EQ(a.extension_number, 0);
  EXPECT_EQ(a.scheduled_auction_time, 1492444800u);
  EXPECT_EQ(a.auction_book_clearing_price, 991500);
  EXPECT_EQ(a.collar_reference_price, 990400);
  EXPECT_EQ(a.lower_auction_collar, 891300);
  EXPECT_EQ(a.upper_auction_collar, 1089500);
}

// The IEX-TP spec's multicast example: a DEEP segment with a trade and a PLU.
TEST(SpecExample, IexTpSegmentWithTwoMessages) {
  auto seg = hex(
      "01 00 04 80 01 00 00 00 00 00 87 42 48 00 02 00 8c a6 21 00 00 00 00 00 "
      "ca c3 00 00 00 00 00 00 ec 45 c2 20 96 86 6d 14 "
      "26 00 54 00 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20 64 00 00 00 "
      "24 1d 0f 00 00 00 00 00 96 8f 06 00 00 00 00 00 "
      "1e 00 38 01 ac 63 c0 20 96 86 6d 14 5a 49 45 58 54 20 20 20 e4 25 00 00 "
      "24 1d 0f 00 00 00 00 00");
  SegmentHeader h;
  const uint8_t* payload = nullptr;
  ASSERT_EQ(parse_segment(seg.data(), static_cast<uint32_t>(seg.size()), h, payload), SegStatus::Ok);
  EXPECT_EQ(h.version, 1);
  EXPECT_EQ(h.protocol_id, kProtoDeep10);
  EXPECT_EQ(h.channel_id, 1u);
  EXPECT_EQ(h.session_id, 0x42870000u);
  EXPECT_EQ(h.payload_length, 72);
  EXPECT_EQ(h.message_count, 2);
  EXPECT_EQ(h.stream_offset, 2205324);
  EXPECT_EQ(h.first_seq, 50122);
  EXPECT_EQ(h.send_time, 1471980632572839404);

  struct Capture : NullHandler {
    std::vector<int64_t> seqs;
    std::vector<Trade> trades;
    std::vector<PriceLevelUpdate> plus;
    void on_trade(const MsgContext& c, const Trade& t) {
      seqs.push_back(c.seq);
      trades.push_back(t);
    }
    void on_price_level(const MsgContext& c, const PriceLevelUpdate& u) {
      seqs.push_back(c.seq);
      plus.push_back(u);
    }
  } cap;
  FeedDecoder<Capture> dec(cap);
  ASSERT_TRUE(dec.on_udp(seg.data(), static_cast<uint32_t>(seg.size()), 0));
  ASSERT_EQ(cap.trades.size(), 1u);
  ASSERT_EQ(cap.plus.size(), 1u);
  EXPECT_EQ(cap.seqs, (std::vector<int64_t>{50122, 50123}));
  EXPECT_EQ(cap.trades[0].trade_id, 429974);
  EXPECT_EQ(cap.plus[0].size, 9700u);
  EXPECT_EQ(dec.stats().framing_errors, 0u);
}

// Gap Fill Test Response from the IEX-TP spec: a heartbeat-shaped TOPS 1.5 segment.
TEST(SpecExample, HeartbeatSegment) {
  auto seg = hex(
      "01 00 02 80 01 00 00 00 00 00 87 42 00 00 00 00 8c a6 21 00 00 00 00 00 "
      "ca c3 00 00 00 00 00 00 ec 45 c2 20 96 86 6d 14");
  SegmentHeader h;
  const uint8_t* payload = nullptr;
  ASSERT_EQ(parse_segment(seg.data(), static_cast<uint32_t>(seg.size()), h, payload), SegStatus::Ok);
  EXPECT_TRUE(h.is_heartbeat());
  EXPECT_EQ(h.protocol_id, kProtoTops15);
  EXPECT_EQ(h.first_seq, 50122);
}

TEST(Segment, RejectsBadInputs) {
  SegmentHeader h;
  const uint8_t* p = nullptr;
  std::vector<uint8_t> short_seg(39, 0);
  EXPECT_EQ(parse_segment(short_seg.data(), 39, h, p), SegStatus::TooShort);
  std::vector<uint8_t> v2(40, 0);
  v2[0] = 2;
  EXPECT_EQ(parse_segment(v2.data(), 40, h, p), SegStatus::BadVersion);
  std::vector<uint8_t> mism(44, 0);
  mism[0] = 1;
  mism[12] = 10;  // claims 10 payload bytes, has 4
  EXPECT_EQ(parse_segment(mism.data(), 44, h, p), SegStatus::LengthMismatch);
}

TEST(Dispatch, UnknownMalformedAndGrownMessages) {
  struct Counts : NullHandler {
    int unknown = 0, malformed = 0, quotes = 0;
    void on_unknown(const MsgContext&, const uint8_t*, uint16_t) { ++unknown; }
    void on_malformed(const MsgContext&, const uint8_t*, uint16_t) { ++malformed; }
    void on_quote(const MsgContext&, const QuoteUpdate& q) {
      ++quotes;
      EXPECT_EQ(q.ask_size, 1000u);
    }
  } h;
  MsgContext ctx{1, 0, kProtoTops16};
  std::vector<uint8_t> unknown = {'z', 1, 2, 3};
  dispatch(ctx, unknown.data(), 4, h);
  auto q = hex(
      "51 00 ac 63 c0 20 96 86 6d 14 5A 49 45 58 54 20 20 20 e4 25 00 00 24 1d 0f 00 00 00 00 00 "
      "ec 1d 0f 00 00 00 00 00 e8 03 00 00");
  dispatch(ctx, q.data(), 41, h);  // one byte short
  q.push_back(0xff);                // a grown message: extra trailing field
  dispatch(ctx, q.data(), static_cast<uint16_t>(q.size()), h);
  dispatch(ctx, q.data(), 0, h);
  EXPECT_EQ(h.unknown, 1);
  EXPECT_EQ(h.malformed, 2);
  EXPECT_EQ(h.quotes, 1);
}
