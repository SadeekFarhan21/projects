// Pre-trade risk checks, run synchronously on the trading thread before an
// order is handed to the gateway. Order of checks is cheapest first.
//
//   fat finger qty       qty == 0 or qty > max_order_qty
//   fat finger notional  price * qty > max_notional (price units are 1e-4 USD)
//   no reference         book has no two-sided market to compare against
//   fat finger price     |price - mid| > max_dev_bps of mid
//   position limit       |position + signed qty| > max_position
//   order rate           more than max_orders_per_window accepted orders in
//                        any trailing window of window_ns
//
// Position assumes every accepted IOC order fills in full (the conservative
// assumption with no fill feedback in v0). Time for the rate limiter is the
// exchange timestamp of the triggering event, which makes replay deterministic;
// production would use the local clock. See DESIGN.md.
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "llt/common.hpp"
#include "llt/messages.hpp"
#include "llt/order_book.hpp"

namespace llt {

struct RiskLimits {
    int64_t max_position{1000};
    uint32_t max_order_qty{1000};
    uint64_t max_notional{250'000ull * 10'000ull};  // $250k in price units * shares
    uint32_t max_dev_bps{100};
    uint32_t max_orders_per_window{20};
    uint64_t window_ns{1'000'000};  // 1 ms
};

enum class RiskVerdict : uint8_t {
    Accept = 0,
    FatFingerQty,
    FatFingerNotional,
    NoReference,
    FatFingerPrice,
    PositionLimit,
    RateLimit,
    Count
};

inline const char* to_string(RiskVerdict v) noexcept {
    switch (v) {
        case RiskVerdict::Accept: return "accept";
        case RiskVerdict::FatFingerQty: return "fat_finger_qty";
        case RiskVerdict::FatFingerNotional: return "fat_finger_notional";
        case RiskVerdict::NoReference: return "no_reference";
        case RiskVerdict::FatFingerPrice: return "fat_finger_price";
        case RiskVerdict::PositionLimit: return "position_limit";
        case RiskVerdict::RateLimit: return "rate_limit";
        default: return "?";
    }
}

class RiskEngine {
public:
    explicit RiskEngine(RiskLimits l = {})
        : l_(l), sent_ts_(l.max_orders_per_window ? l.max_orders_per_window : 1, 0) {}

    // Checks and, on Accept, commits the order's effect on position and rate.
    RiskVerdict check(const OrderRequest& o, const Top& t, uint64_t now_ns) noexcept {
        if (o.qty == 0 || o.qty > l_.max_order_qty) return RiskVerdict::FatFingerQty;
        if (static_cast<uint64_t>(o.price) * o.qty > l_.max_notional)
            return RiskVerdict::FatFingerNotional;
        if (t.bid_qty == 0 || t.ask_qty == 0) return RiskVerdict::NoReference;
        // Compare in doubled units to stay in integers: 2*mid = bid + ask.
        const int64_t mid2 = static_cast<int64_t>(t.bid_px) + t.ask_px;
        const int64_t diff2 = std::llabs(2 * static_cast<int64_t>(o.price) - mid2);
        if (diff2 * 10'000 > static_cast<int64_t>(l_.max_dev_bps) * mid2)
            return RiskVerdict::FatFingerPrice;
        if (o.locate >= kMaxSymbols) return RiskVerdict::PositionLimit;
        const int64_t delta = o.side == 'B' ? static_cast<int64_t>(o.qty) : -static_cast<int64_t>(o.qty);
        if (std::llabs(pos_[o.locate] + delta) > l_.max_position) return RiskVerdict::PositionLimit;
        // Sliding window: sent_ts_ is a circular buffer of the last N accept
        // times; when full, the slot we are about to overwrite is the oldest.
        const std::size_t n = sent_ts_.size();
        if (l_.max_orders_per_window == 0) return RiskVerdict::RateLimit;
        if (count_ == n && now_ns - sent_ts_[next_] < l_.window_ns) return RiskVerdict::RateLimit;

        pos_[o.locate] += delta;
        sent_ts_[next_] = now_ns;
        next_ = next_ + 1 == n ? 0 : next_ + 1;
        if (count_ < n) ++count_;
        return RiskVerdict::Accept;
    }

    int64_t position(uint16_t locate) const noexcept { return locate < kMaxSymbols ? pos_[locate] : 0; }

private:
    RiskLimits l_;
    std::array<int64_t, kMaxSymbols> pos_{};
    std::vector<uint64_t> sent_ts_;
    std::size_t next_{0};
    std::size_t count_{0};
};

}  // namespace llt
