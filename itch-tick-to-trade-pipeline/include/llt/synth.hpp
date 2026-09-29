// Synthetic ITCH-like capture generator.
//
// Models, per symbol, a latent "fair" price that random-walks one tick at a
// time, limit orders placed a geometric number of ticks away from it, order
// deletes and partial cancels, marketable flow that executes against the best
// level in time priority, and sweeps of resting orders the fair price walks
// through. Inter-arrival times are a two-state mixture (bursts plus a slower
// background) so the replay has realistic clumping. See DESIGN.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llt {

struct SynthParams {
    uint64_t messages{1'000'000};
    uint32_t symbols{8};
    uint64_t seed{42};
    double mean_gap_ns{1000.0};  // mean inter-arrival across the whole feed
    double burst_prob{0.3};      // fraction of gaps drawn from the burst state
    double burst_mean_ns{60.0};  // mean gap inside a burst
    uint32_t target_live_per_symbol{1500};
    uint32_t max_ticks_from_fair{200};
    uint32_t fair_band_ticks{1500};  // fair price stays within +-this of its start
};

struct SynthStats {
    uint64_t adds{0}, execs{0}, cancels{0}, deletes{0}, directory{0};
    uint64_t bytes{0};
    uint64_t first_ts{0}, last_ts{0};
};

std::vector<uint8_t> generate_capture(const SynthParams& p, SynthStats* stats = nullptr);

}  // namespace llt
