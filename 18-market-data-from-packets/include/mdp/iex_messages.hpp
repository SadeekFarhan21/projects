// IEX TOPS 1.6 and DEEP 1.0 message layouts and a static dispatcher.
//
// Field offsets follow the IEX TOPS v1.66 and DEEP v1.08 specifications. All
// integers are little endian; prices are int64 with 4 implied decimals, which
// is exactly the normalized price unit (1e-4 dollars), so no conversion is done.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "mdp/bytes.hpp"

namespace mdp {

// 8-byte space-padded ASCII symbol kept as its raw little-endian u64 image.
using Symbol = uint64_t;

inline Symbol make_symbol(const char* s) {
  char buf[8];
  std::memset(buf, ' ', 8);
  for (int i = 0; i < 8 && s[i]; ++i) buf[i] = s[i];
  Symbol v;
  std::memcpy(&v, buf, 8);
  return v;
}

inline std::string symbol_str(Symbol v) {
  char buf[8];
  std::memcpy(buf, &v, 8);
  int n = 8;
  while (n > 0 && (buf[n - 1] == ' ' || buf[n - 1] == '\0')) --n;
  return std::string(buf, n);
}

namespace msg {
enum Type : uint8_t {
  SystemEvent = 'S',
  SecurityDirectory = 'D',
  TradingStatus = 'H',
  RetailLiquidity = 'I',
  OperationalHalt = 'O',
  ShortSalePriceTest = 'P',
  SecurityEvent = 'E',
  QuoteUpdate = 'Q',
  TradeReport = 'T',
  OfficialPrice = 'X',
  TradeBreak = 'B',
  AuctionInfo = 'A',
  PriceLevelBuy = '8',
  PriceLevelSell = '5',
};

// Minimum message data lengths from the specs. Messages may grow, never shrink.
constexpr uint16_t min_length(uint8_t type) {
  switch (type) {
    case SystemEvent: return 10;
    case SecurityDirectory: return 31;
    case TradingStatus: return 22;
    case RetailLiquidity: return 18;
    case OperationalHalt: return 18;
    case ShortSalePriceTest: return 19;
    case SecurityEvent: return 18;
    case QuoteUpdate: return 42;
    case TradeReport: return 38;
    case OfficialPrice: return 26;
    case TradeBreak: return 38;
    case AuctionInfo: return 80;
    case PriceLevelBuy: return 30;
    case PriceLevelSell: return 30;
    default: return 0;
  }
}
}  // namespace msg

// Quote Update flags (TOPS Appendix A).
constexpr uint8_t kQuoteFlagHalted = 0x80;   // symbol halted, paused or unavailable
constexpr uint8_t kQuoteFlagPrePost = 0x40;  // pre or post market session
// Sale condition flags.
constexpr uint8_t kSaleIso = 0x80;
constexpr uint8_t kSaleExtendedHours = 0x40;
constexpr uint8_t kSaleOddLot = 0x20;
constexpr uint8_t kSaleTradeThroughExempt = 0x10;
constexpr uint8_t kSaleSinglePriceCross = 0x08;
// Price level update event flags.
constexpr uint8_t kEventProcessingComplete = 0x01;

struct SystemEvent {
  uint8_t event;  // O S R M E C
  int64_t ts;
};
struct SecurityDirectory {
  uint8_t flags;
  int64_t ts;
  Symbol symbol;
  uint32_t round_lot;
  int64_t adjusted_poc_price;
  uint8_t luld_tier;
};
struct TradingStatus {
  uint8_t status;  // H O P T
  int64_t ts;
  Symbol symbol;
  uint32_t reason;  // 4 raw ASCII bytes
};
// Retail Liquidity Indicator (I), Operational Halt (O), Short Sale Price Test (P)
// and Security Event (E) share a shape: one status byte, timestamp, symbol,
// and for P a trailing detail byte.
struct SecurityStatus {
  uint8_t type;
  uint8_t status;
  int64_t ts;
  Symbol symbol;
  uint8_t detail;  // P only, else 0
};
struct QuoteUpdate {
  uint8_t flags;
  int64_t ts;
  Symbol symbol;
  uint32_t bid_size;
  int64_t bid_price;
  int64_t ask_price;
  uint32_t ask_size;
};
struct Trade {  // Trade Report (T) or Trade Break (B)
  uint8_t type;
  uint8_t flags;
  int64_t ts;
  Symbol symbol;
  uint32_t size;
  int64_t price;
  int64_t trade_id;
};
struct OfficialPrice {
  uint8_t price_type;  // Q open, M close
  int64_t ts;
  Symbol symbol;
  int64_t price;
};
struct PriceLevelUpdate {
  uint8_t side;  // '8' buy, '5' sell
  uint8_t flags;
  int64_t ts;
  Symbol symbol;
  uint32_t size;
  int64_t price;
};
struct AuctionInfo {
  uint8_t auction_type;
  int64_t ts;
  Symbol symbol;
  uint32_t paired_shares;
  int64_t reference_price;
  int64_t indicative_clearing_price;
  uint32_t imbalance_shares;
  uint8_t imbalance_side;
  uint8_t extension_number;
  uint32_t scheduled_auction_time;  // seconds since epoch
  int64_t auction_book_clearing_price;
  int64_t collar_reference_price;
  int64_t lower_auction_collar;
  int64_t upper_auction_collar;
};

// Decoders assume the length was checked against msg::min_length.
inline SystemEvent decode_system_event(const uint8_t* m) { return {m[1], load_le<int64_t>(m + 2)}; }

inline SecurityDirectory decode_security_directory(const uint8_t* m) {
  return {m[1], load_le<int64_t>(m + 2), load_le<uint64_t>(m + 10), load_le<uint32_t>(m + 18),
          load_le<int64_t>(m + 22), m[30]};
}

inline TradingStatus decode_trading_status(const uint8_t* m) {
  return {m[1], load_le<int64_t>(m + 2), load_le<uint64_t>(m + 10), load_le<uint32_t>(m + 18)};
}

inline SecurityStatus decode_security_status(const uint8_t* m, uint16_t len) {
  uint8_t detail = (m[0] == msg::ShortSalePriceTest && len >= 19) ? m[18] : 0;
  return {m[0], m[1], load_le<int64_t>(m + 2), load_le<uint64_t>(m + 10), detail};
}

inline QuoteUpdate decode_quote(const uint8_t* m) {
  return {m[1],
          load_le<int64_t>(m + 2),
          load_le<uint64_t>(m + 10),
          load_le<uint32_t>(m + 18),
          load_le<int64_t>(m + 22),
          load_le<int64_t>(m + 30),
          load_le<uint32_t>(m + 38)};
}

inline Trade decode_trade(const uint8_t* m) {
  return {m[0],
          m[1],
          load_le<int64_t>(m + 2),
          load_le<uint64_t>(m + 10),
          load_le<uint32_t>(m + 18),
          load_le<int64_t>(m + 22),
          load_le<int64_t>(m + 30)};
}

inline OfficialPrice decode_official_price(const uint8_t* m) {
  return {m[1], load_le<int64_t>(m + 2), load_le<uint64_t>(m + 10), load_le<int64_t>(m + 18)};
}

inline PriceLevelUpdate decode_price_level(const uint8_t* m) {
  return {m[0], m[1], load_le<int64_t>(m + 2), load_le<uint64_t>(m + 10), load_le<uint32_t>(m + 18),
          load_le<int64_t>(m + 22)};
}

inline AuctionInfo decode_auction(const uint8_t* m) {
  return {m[1],
          load_le<int64_t>(m + 2),
          load_le<uint64_t>(m + 10),
          load_le<uint32_t>(m + 18),
          load_le<int64_t>(m + 22),
          load_le<int64_t>(m + 30),
          load_le<uint32_t>(m + 38),
          m[42],
          m[43],
          load_le<uint32_t>(m + 44),
          load_le<int64_t>(m + 48),
          load_le<int64_t>(m + 56),
          load_le<int64_t>(m + 64),
          load_le<int64_t>(m + 72)};
}

// Per-message context handed to handlers.
struct MsgContext {
  int64_t seq;        // IEX-TP sequence number of this message
  int64_t send_time;  // segment send time
  uint16_t protocol_id;
};

// Base handler with no-op callbacks. Real handlers derive and hide the ones they
// need; dispatch() is a template, so calls are resolved statically and inlined.
struct NullHandler {
  void on_system_event(const MsgContext&, const SystemEvent&) {}
  void on_security_directory(const MsgContext&, const SecurityDirectory&) {}
  void on_trading_status(const MsgContext&, const TradingStatus&) {}
  void on_security_status(const MsgContext&, const SecurityStatus&) {}
  void on_quote(const MsgContext&, const QuoteUpdate&) {}
  void on_trade(const MsgContext&, const Trade&) {}
  void on_official_price(const MsgContext&, const OfficialPrice&) {}
  void on_price_level(const MsgContext&, const PriceLevelUpdate&) {}
  void on_auction(const MsgContext&, const AuctionInfo&) {}
  void on_unknown(const MsgContext&, const uint8_t*, uint16_t) {}
  void on_malformed(const MsgContext&, const uint8_t*, uint16_t) {}
};

template <class H>
inline void dispatch(const MsgContext& ctx, const uint8_t* m, uint16_t len, H& h) {
  if (len == 0) {
    h.on_malformed(ctx, m, len);
    return;
  }
  const uint8_t type = m[0];
  const uint16_t need = msg::min_length(type);
  if (need == 0) {
    h.on_unknown(ctx, m, len);
    return;
  }
  if (len < need) {
    h.on_malformed(ctx, m, len);
    return;
  }
  switch (type) {
    case msg::PriceLevelBuy:
    case msg::PriceLevelSell: h.on_price_level(ctx, decode_price_level(m)); break;
    case msg::QuoteUpdate: h.on_quote(ctx, decode_quote(m)); break;
    case msg::TradeReport:
    case msg::TradeBreak: h.on_trade(ctx, decode_trade(m)); break;
    case msg::SystemEvent: h.on_system_event(ctx, decode_system_event(m)); break;
    case msg::SecurityDirectory: h.on_security_directory(ctx, decode_security_directory(m)); break;
    case msg::TradingStatus: h.on_trading_status(ctx, decode_trading_status(m)); break;
    case msg::RetailLiquidity:
    case msg::OperationalHalt:
    case msg::ShortSalePriceTest:
    case msg::SecurityEvent: h.on_security_status(ctx, decode_security_status(m, len)); break;
    case msg::OfficialPrice: h.on_official_price(ctx, decode_official_price(m)); break;
    case msg::AuctionInfo: h.on_auction(ctx, decode_auction(m)); break;
    default: h.on_unknown(ctx, m, len); break;
  }
}

}  // namespace mdp
