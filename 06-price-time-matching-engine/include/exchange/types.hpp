// Core value types. Prices and quantities are fixed-point integers:
// a Price is a count of ticks and a Qty is a count of lots. Nothing in the
// matching path ever touches floating point.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace exch {

using Price = std::int64_t;    // ticks
using Qty = std::int64_t;      // lots
using OrderId = std::uint64_t; // client supplied, 0 is reserved as "none"
using TradeId = std::uint64_t;
using SeqNo = std::uint64_t;   // position of an input event in the log

enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

inline constexpr Side opposite(Side s) { return s == Side::Buy ? Side::Sell : Side::Buy; }

// How an incoming order interacts with the book.
//   Limit     rests any unfilled remainder (good till cancel)
//   Market    no price bound, remainder is cancelled
//   IOC       limit price bound, remainder is cancelled
//   FOK       limit price bound, fills completely or not at all
//   PostOnly  limit order that is rejected if it would take liquidity
enum class OrderType : std::uint8_t { Limit = 0, Market, IOC, FOK, PostOnly };

// Validation limits. Keeping quantities below 2^40 means a price level can
// hold millions of maximal orders without its int64 aggregate overflowing.
inline constexpr Price kMaxPrice = (Price{1} << 40);
inline constexpr Qty kMaxQty = (Qty{1} << 40);

// Fixed-point helpers used only at the edges (CLI, logs). `scale` is the
// number of decimal places one tick represents, e.g. scale 2 means 1 tick
// equals 0.01.
std::optional<std::int64_t> parse_fixed(std::string_view text, int scale);
std::string format_fixed(std::int64_t value, int scale);

const char* to_string(Side s);
const char* to_string(OrderType t);

} // namespace exch
