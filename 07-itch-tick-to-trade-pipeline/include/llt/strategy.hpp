// Top-of-book imbalance / microprice strategy.
//
// imbalance  I = (Qb - Qa) / (Qb + Qa)                  in [-1, 1]
// microprice M = (Pb * Qa + Pa * Qb) / (Qb + Qa)
//             = mid + (spread / 2) * I
//
// So "microprice sits more than k half-spreads above mid" is the same test as
// "I > k". The strategy is edge-triggered: it trades when the signal moves
// from neutral (or the opposite side) to a strong reading, lifting the ask on a
// strong bid imbalance and hitting the bid on a strong ask imbalance, subject to
// a per-symbol cooldown measured in exchange time.
//
// This is a latency-testing strategy, not an alpha claim.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "llt/common.hpp"
#include "llt/messages.hpp"
#include "llt/order_book.hpp"

namespace llt {

struct StrategyParams {
    double threshold{0.6};        // |I| needed to act
    uint64_t cooldown_ns{50'000}; // min exchange-time gap between orders per symbol
    uint32_t order_qty{100};
};

inline double imbalance(const Top& t) noexcept {
    const double b = t.bid_qty, a = t.ask_qty;
    return (b - a) / (b + a);
}

inline double microprice(const Top& t) noexcept {
    const double b = t.bid_qty, a = t.ask_qty;
    return (static_cast<double>(t.bid_px) * a + static_cast<double>(t.ask_px) * b) / (a + b);
}

class ImbalanceStrategy {
public:
    explicit ImbalanceStrategy(StrategyParams p = {}) : p_(p) {}

    void on_directory(const MdEvent& e) noexcept {
        if (e.locate < kMaxSymbols) std::memcpy(sym_[e.locate].stock, e.stock, 8);
    }

    // Call after the book has applied `e`. Returns true and fills `out` when the
    // strategy wants to send an order.
    bool on_book(const MdEvent& e, const Top& t, OrderRequest& out) noexcept {
        if (e.locate >= kMaxSymbols) return false;
        State& s = sym_[e.locate];
        if (e.type == 'A' && s.stock[0] == 0) std::memcpy(s.stock, e.stock, 8);
        if (t.bid_qty == 0 || t.ask_qty == 0) {
            s.last_signal = 0;
            return false;
        }
        const double imb = imbalance(t);
        const int sig = imb > p_.threshold ? 1 : (imb < -p_.threshold ? -1 : 0);
        const int prev = s.last_signal;
        s.last_signal = sig;
        if (sig == 0 || sig == prev) return false;
        if (s.traded && e.exch_ts - s.last_order_ts < p_.cooldown_ns) return false;
        s.traded = true;
        s.last_order_ts = e.exch_ts;

        out.t_in = e.t_in;
        out.locate = e.locate;
        out.side = sig > 0 ? 'B' : 'S';
        out.price = sig > 0 ? t.ask_px : t.bid_px;
        out.qty = p_.order_qty;
        std::memcpy(out.stock, s.stock, 8);
        return true;
    }

private:
    struct State {
        char stock[8]{};
        int last_signal{0};
        bool traded{false};
        uint64_t last_order_ts{0};
    };
    StrategyParams p_;
    std::array<State, kMaxSymbols> sym_{};
};

}  // namespace llt
