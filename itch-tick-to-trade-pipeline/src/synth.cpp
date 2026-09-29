#include "llt/synth.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <random>
#include <unordered_map>

#include "llt/itch.hpp"

namespace llt {
namespace {

constexpr uint32_t kTick = 100;  // $0.01 in ITCH price units (4 implied decimals)

struct GOrder {
    uint16_t sym;
    char side;
    uint32_t px;
    uint32_t qty;
    std::size_t live_idx;
};

struct GSym {
    char name[8];
    uint32_t fair;
    uint32_t fair0;
    std::map<uint32_t, std::deque<uint64_t>, std::greater<>> bids;  // best first
    std::map<uint32_t, std::deque<uint64_t>> asks;                  // best first
    std::vector<uint64_t> live;
};

const char* kNames[] = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
                        "NFLX", "INTC", "ORCL", "CRM",  "ADBE", "QCOM",  "CSCO", "AVGO"};
const uint32_t kDollars[] = {190, 420, 120, 180, 170, 500, 250, 160,
                             600, 30,  140, 270, 520, 170, 50,  1500};

}  // namespace

std::vector<uint8_t> generate_capture(const SynthParams& p, SynthStats* stats_out) {
    SynthStats st;
    std::vector<uint8_t> out;
    out.reserve(p.messages * 34);
    itch::Writer w(out);
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const uint32_t nsym = std::min<uint32_t>(p.symbols, 16);
    std::vector<GSym> syms(nsym);
    std::vector<double> weights;
    for (uint32_t i = 0; i < nsym; ++i) {
        std::memset(syms[i].name, ' ', 8);  // ITCH stock field is space padded
        std::memcpy(syms[i].name, kNames[i], std::strlen(kNames[i]));
        syms[i].fair = syms[i].fair0 = kDollars[i] * 10'000;
        weights.push_back(1.0 / (1.0 + i));  // Zipf-like activity
    }
    std::discrete_distribution<uint32_t> pick_sym(weights.begin(), weights.end());

    std::unordered_map<uint64_t, GOrder> orders;
    orders.reserve(nsym * p.target_live_per_symbol * 4);
    uint64_t next_ref = 1, match = 1, emitted = 0;
    uint64_t ts = 34'200ull * 1'000'000'000ull;  // 09:30:00 in ns since midnight
    const double slow_mean =
        std::max(1.0, (p.mean_gap_ns - p.burst_prob * p.burst_mean_ns) / (1.0 - p.burst_prob));
    std::exponential_distribution<double> burst_gap(1.0 / p.burst_mean_ns), slow_gap(1.0 / slow_mean);
    auto tick_ts = [&] {
        const double g = U(rng) < p.burst_prob ? burst_gap(rng) : slow_gap(rng);
        ts += std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(g)));
        ++emitted;
    };

    auto remove_live = [&](uint64_t ref) {
        auto it = orders.find(ref);
        auto& L = syms[it->second.sym].live;
        const std::size_t i = it->second.live_idx;
        const uint64_t last = L.back();
        L[i] = last;
        orders[last].live_idx = i;
        L.pop_back();
        orders.erase(it);
    };
    // First live order at the best level of a side (time priority), or 0.
    auto front_live = [&](auto& levels) -> uint64_t {
        while (!levels.empty()) {
            auto it = levels.begin();
            auto& dq = it->second;
            while (!dq.empty() && !orders.count(dq.front())) dq.pop_front();
            if (dq.empty()) {
                levels.erase(it);
                continue;
            }
            return dq.front();
        }
        return 0;
    };
    auto exec = [&](uint64_t ref, uint32_t q) {
        GOrder& o = orders[ref];
        tick_ts();
        w.executed(o.sym, ts, ref, q, match++);
        ++st.execs;
        o.qty -= q;
        if (o.qty == 0) remove_live(ref);
    };

    for (uint32_t i = 0; i < nsym; ++i) {
        tick_ts();
        w.directory(static_cast<uint16_t>(i), ts, syms[i].name);
        ++st.directory;
    }
    st.first_ts = ts;

    std::geometric_distribution<uint32_t> dist_ticks(0.12), lots(0.45);
    while (emitted < p.messages) {
        const uint16_t s = static_cast<uint16_t>(pick_sym(rng));
        GSym& S = syms[s];

        if (U(rng) < 0.04) {
            // Fair price moves one tick; resting orders it walks through trade.
            const int64_t band = static_cast<int64_t>(p.fair_band_ticks) * kTick;
            int64_t step = U(rng) < 0.5 ? kTick : -static_cast<int64_t>(kTick);
            const int64_t nf = static_cast<int64_t>(S.fair) + step - S.fair0;
            if (nf > band || nf < -band) step = -step;
            S.fair = static_cast<uint32_t>(static_cast<int64_t>(S.fair) + step);
            for (uint64_t r; emitted < p.messages && (r = front_live(S.asks)) && orders[r].px <= S.fair;)
                exec(r, orders[r].qty);
            for (uint64_t r; emitted < p.messages && (r = front_live(S.bids)) && orders[r].px >= S.fair;)
                exec(r, orders[r].qty);
            continue;
        }

        const std::size_t live = S.live.size();
        const double p_add =
            live < p.target_live_per_symbol / 2 ? 1.0 : (live > 2 * p.target_live_per_symbol ? 0.0 : 0.45);
        if (live == 0 || U(rng) < p_add) {
            const char side = U(rng) < 0.5 ? 'B' : 'S';
            const uint32_t k = 1 + std::min(dist_ticks(rng), p.max_ticks_from_fair - 1);
            const uint32_t px = side == 'B' ? S.fair - k * kTick : S.fair + k * kTick;
            const uint32_t qty = U(rng) < 0.1 ? 1 + static_cast<uint32_t>(U(rng) * 99)
                                              : 100 * (1 + std::min<uint32_t>(lots(rng), 20));
            const uint64_t ref = next_ref++;
            S.live.push_back(ref);
            orders[ref] = GOrder{s, side, px, qty, S.live.size() - 1};
            if (side == 'B') S.bids[px].push_back(ref);
            else S.asks[px].push_back(ref);
            tick_ts();
            w.add(s, ts, ref, side, qty, S.name, px);
            ++st.adds;
            continue;
        }

        const double r = U(rng);
        if (r < 0.80) {
            const uint64_t ref = S.live[static_cast<std::size_t>(U(rng) * S.live.size()) % S.live.size()];
            GOrder& o = orders[ref];
            if (r < 0.70 || o.qty < 2) {
                tick_ts();
                w.del(s, ts, ref);
                ++st.deletes;
                remove_live(ref);
            } else {
                const uint32_t c = 1 + static_cast<uint32_t>(U(rng) * (o.qty - 1));
                tick_ts();
                w.cancel(s, ts, ref, std::min(c, o.qty - 1));
                ++st.cancels;
                o.qty -= std::min(c, o.qty - 1);
            }
        } else {
            // Marketable order hits the best level on one side.
            const bool hit_bid = U(rng) < 0.5;
            const uint64_t ref = hit_bid ? front_live(S.bids) : front_live(S.asks);
            if (!ref) continue;
            const uint32_t q = orders[ref].qty;
            const uint32_t eq = U(rng) < 0.5 ? q : 1 + static_cast<uint32_t>(U(rng) * q) % q;
            exec(ref, eq);
        }
    }
    st.last_ts = ts;
    st.bytes = out.size();
    if (stats_out) *stats_out = st;
    return out;
}

}  // namespace llt
