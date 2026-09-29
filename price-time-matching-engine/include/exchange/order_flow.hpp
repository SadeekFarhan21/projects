// Deterministic random order-flow generator shared by the tests, the fuzz
// driver, the benchmark and the `exch gen` tool. It is open loop (it never
// looks at the book), so it also produces cancels of already-filled ids,
// duplicate ids and invalid sizes, which exercise the reject paths.
//
// std::uniform_int_distribution is not specified bit-for-bit across
// standard libraries, so we use our own PRNG and range reduction to keep
// logs reproducible everywhere.
#pragma once

#include "exchange/events.hpp"

#include <cstdint>
#include <vector>

namespace exch {

class Rng {
public:
    explicit Rng(std::uint64_t seed) : s_(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() { // splitmix64
        std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Uniform in [lo, hi]. Multiply-shift reduction; bias is < 2^-32 for our ranges.
    std::int64_t range(std::int64_t lo, std::int64_t hi) {
        const std::uint64_t span = static_cast<std::uint64_t>(hi - lo) + 1;
        return lo + static_cast<std::int64_t>((static_cast<unsigned __int128>(next()) * span) >> 64);
    }
    bool chance(double p) { return static_cast<double>(next() >> 11) * 0x1.0p-53 < p; }

private:
    std::uint64_t s_;
};

struct FlowConfig {
    std::uint64_t seed = 1;
    Price mid = 10'000;      // starting mid price in ticks
    Price half_width = 20;   // limit prices drawn from mid +/- half_width
    Qty max_qty = 100;
    std::size_t max_live = 500; // cap on ids the generator believes are live
    double p_cancel = 0.30;
    double p_modify = 0.10;
    // Order type mix among new orders (the rest are plain limits).
    double p_market = 0.03;
    double p_ioc = 0.07;
    double p_fok = 0.05;
    double p_post = 0.10;
    double p_invalid = 0.005; // bad qty / price / duplicate or unknown id
    double p_drift = 0.02;    // chance per event the mid moves one tick
};

class OrderFlow {
public:
    explicit OrderFlow(FlowConfig c) : cfg_(c), rng_(c.seed), mid_(c.mid) {}

    InputEvent next() {
        if (rng_.chance(cfg_.p_drift)) mid_ += rng_.chance(0.5) ? 1 : -1;
        if (mid_ < cfg_.half_width + 2) mid_ = cfg_.half_width + 2;

        if (rng_.chance(cfg_.p_invalid)) return invalid();

        const bool full = live_.size() >= cfg_.max_live;
        const double r = static_cast<double>(rng_.next() >> 11) * 0x1.0p-53;
        if (!live_.empty() && (full || r < cfg_.p_cancel)) {
            return InputEvent::cancel(take_live());
        }
        if (!live_.empty() && r < cfg_.p_cancel + cfg_.p_modify) {
            const OrderId id = live_[static_cast<std::size_t>(rng_.range(0, static_cast<std::int64_t>(live_.size()) - 1))];
            return InputEvent::modify(id, price(), rng_.range(1, cfg_.max_qty));
        }
        const OrderId id = next_id_++;
        const Side side = rng_.chance(0.5) ? Side::Buy : Side::Sell;
        OrderType t = OrderType::Limit;
        const double u = static_cast<double>(rng_.next() >> 11) * 0x1.0p-53;
        double acc = cfg_.p_market;
        if (u < acc) t = OrderType::Market;
        else if (u < (acc += cfg_.p_ioc)) t = OrderType::IOC;
        else if (u < (acc += cfg_.p_fok)) t = OrderType::FOK;
        else if (u < (acc += cfg_.p_post)) t = OrderType::PostOnly;
        // Aggressive orders get a price that tends to cross.
        Price p = price();
        if (t == OrderType::Market) p = 0;
        const Qty q = rng_.range(1, cfg_.max_qty);
        if (t == OrderType::Limit || t == OrderType::PostOnly) live_.push_back(id);
        return InputEvent::new_order(id, side, t, p, q);
    }

    std::vector<InputEvent> take(std::size_t n) {
        std::vector<InputEvent> v;
        v.reserve(n);
        for (std::size_t i = 0; i < n; ++i) v.push_back(next());
        return v;
    }

private:
    Price price() { return mid_ + rng_.range(-cfg_.half_width, cfg_.half_width); }

    OrderId take_live() {
        const std::size_t i = static_cast<std::size_t>(rng_.range(0, static_cast<std::int64_t>(live_.size()) - 1));
        const OrderId id = live_[i];
        live_[i] = live_.back();
        live_.pop_back();
        return id;
    }

    InputEvent invalid() {
        switch (rng_.range(0, 5)) {
        case 0: return InputEvent::new_order(next_id_++, Side::Buy, OrderType::Limit, price(), 0);
        case 1: return InputEvent::new_order(next_id_++, Side::Sell, OrderType::Limit, -5, 10);
        case 2: // duplicate of something that may still be resting
            if (!live_.empty()) return InputEvent::new_order(live_.back(), Side::Buy, OrderType::Limit, price(), 5);
            return InputEvent::new_order(0, Side::Buy, OrderType::Limit, price(), 5);
        case 3: return InputEvent::cancel(next_id_ + 1'000'000);
        case 4: return InputEvent::modify(live_.empty() ? 0 : live_.back(), price(), 0);
        default: return InputEvent::new_order(0, Side::Sell, OrderType::IOC, price(), 3);
        }
    }

    FlowConfig cfg_;
    Rng rng_;
    Price mid_;
    OrderId next_id_{1};
    std::vector<OrderId> live_;
};

} // namespace exch
