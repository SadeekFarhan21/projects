// Message handler that turns decoded IEX messages into normalized table rows
// and, for DEEP, maintains per-symbol books and emits the consistent BBO.
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>

#include "mdp/book.hpp"
#include "mdp/feed.hpp"
#include "mdp/iex_messages.hpp"
#include "mdp/records.hpp"

namespace mdp {

class Normalizer : public NullHandler {
 public:
  // out_dir empty: count rows only (no files).
  Normalizer(const std::string& out_dir, bool build_book);

  void on_system_event(const MsgContext& c, const SystemEvent& e) {
    system_events_.push({e.ts, c.seq, e.event});
    track_ts(e.ts);
  }
  void on_security_directory(const MsgContext& c, const SecurityDirectory& d) {
    security_directory_.push({d.ts, c.seq, d.symbol, d.adjusted_poc_price, d.round_lot, d.flags, d.luld_tier});
    track_ts(d.ts);
  }
  void on_trading_status(const MsgContext& c, const TradingStatus& s) {
    trading_status_.push({s.ts, c.seq, s.symbol, s.reason, s.status});
    track_ts(s.ts);
  }
  void on_security_status(const MsgContext& c, const SecurityStatus& s) {
    security_status_.push({s.ts, c.seq, s.symbol, s.type, s.status, s.detail});
    track_ts(s.ts);
  }
  void on_quote(const MsgContext& c, const QuoteUpdate& q) {
    quotes_.push({q.ts, c.seq, q.symbol, q.bid_price, q.ask_price, q.bid_size, q.ask_size, q.flags});
    track_ts(q.ts);
  }
  void on_trade(const MsgContext& c, const Trade& t) {
    trades_.push({t.ts, c.seq, t.symbol, t.price, t.trade_id, t.size, t.flags, t.type});
    track_ts(t.ts);
  }
  void on_official_price(const MsgContext& c, const OfficialPrice& p) {
    official_prices_.push({p.ts, c.seq, p.symbol, p.price, p.price_type});
    track_ts(p.ts);
  }
  void on_auction(const MsgContext& c, const AuctionInfo& a) {
    auctions_.push({a.ts, c.seq, a.symbol, a.reference_price, a.indicative_clearing_price,
                    a.auction_book_clearing_price, a.collar_reference_price, a.lower_auction_collar,
                    a.upper_auction_collar, a.paired_shares, a.imbalance_shares, a.scheduled_auction_time,
                    a.auction_type, a.imbalance_side, a.extension_number});
    track_ts(a.ts);
  }
  void on_price_level(const MsgContext& c, const PriceLevelUpdate& u) {
    const uint8_t side = u.side == msg::PriceLevelBuy ? 'B' : 'S';
    deep_levels_.push({u.ts, c.seq, u.symbol, u.price, u.size, side, u.flags});
    track_ts(u.ts);
    if (!build_book_) return;
    SymbolBook& b = books_.book(u.symbol);
    if (b.apply(u)) {
      const Bbo& x = b.bbo();
      bbo_.push({u.ts, c.seq, u.symbol, x.bid_price, x.ask_price, x.bid_size, x.ask_size,
                 static_cast<uint16_t>(std::min<size_t>(b.bids().depth(), 65535)),
                 static_cast<uint16_t>(std::min<size_t>(b.asks().depth(), 65535))});
    }
  }
  void on_unknown(const MsgContext& c, const uint8_t* m, uint16_t len) {
    unknown_.push({c.send_time, c.seq, len, len ? m[0] : uint8_t{0}});
  }
  void on_malformed(const MsgContext& c, const uint8_t* m, uint16_t len) {
    ++malformed_;
    unknown_.push({c.send_time, c.seq, len, len ? m[0] : uint8_t{0}});
  }
  // Segment observer hook.
  void on_segment(const SegmentHeader& h, int64_t capture_ts, uint16_t skip) {
    segments_.push({capture_ts, h.send_time, h.first_seq, h.stream_offset, h.channel_id, h.session_id,
                    h.protocol_id, h.message_count, h.payload_length, skip});
  }

  void close();
  // Writes manifest.json with row counts, record sizes and decode statistics.
  template <class Dec>
  void write_manifest(const std::string& path, const Dec& dec, const std::string& feed, const std::string& input,
                      double seconds, uint64_t bytes_in, bool truncated, const std::string& capture_format) const;

  const BookManager& books() const { return books_; }
  uint64_t malformed() const { return malformed_; }
  uint64_t bbo_rows() const { return bbo_.rows(); }

 private:
  void track_ts(int64_t ts) {
    if (ts < min_ts_) min_ts_ = ts;
    if (ts > max_ts_) max_ts_ = ts;
  }
  std::string dir_;
  bool build_book_;
  BookManager books_;
  TableWriter<QuoteRec> quotes_;
  TableWriter<LevelRec> deep_levels_;
  TableWriter<BboRec> bbo_;
  TableWriter<TradeRec> trades_;
  TableWriter<SystemEventRec> system_events_;
  TableWriter<SecurityDirectoryRec> security_directory_;
  TableWriter<TradingStatusRec> trading_status_;
  TableWriter<SecurityStatusRec> security_status_;
  TableWriter<OfficialPriceRec> official_prices_;
  TableWriter<AuctionRec> auctions_;
  TableWriter<UnknownRec> unknown_;
  TableWriter<SegmentRec> segments_;
  uint64_t malformed_ = 0;
  int64_t min_ts_ = INT64_MAX;
  int64_t max_ts_ = INT64_MIN;
};

}  // namespace mdp

#include "mdp/normalizer_manifest.inl"
