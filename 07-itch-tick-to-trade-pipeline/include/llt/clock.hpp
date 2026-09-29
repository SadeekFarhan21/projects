// Timestamp source for latency measurement.
//
// On arm64 we read the architectural virtual counter CNTVCT_EL0 directly. This
// is what mach_absolute_time() reads on Apple Silicon, minus the function call
// and commpage indirection. It is synchronized across cores, so a timestamp
// taken on one thread can be subtracted from one taken on another. Its
// frequency on M-series machines is 24 MHz, which means one tick is 41.67 ns.
// That quantization is the dominant limit on single-sample precision and is
// discussed in DESIGN.md and DEVLOG.md. No finer user-space counter exists on
// macOS without kernel PMC access.
#pragma once

#include <chrono>
#include <cstdint>

namespace llt::clk {

inline uint64_t now() noexcept {
#if defined(__aarch64__)
    uint64_t v;
    // isb keeps the counter read from being hoisted above earlier instructions.
    asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v)::"memory");
    return v;
#else
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

// Same counter without the barrier. Cheaper, but may be reordered.
inline uint64_t now_unordered() noexcept {
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return now();
#endif
}

inline uint64_t freq_hz() noexcept {
#if defined(__aarch64__)
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
#else
    using P = std::chrono::steady_clock::period;
    return static_cast<uint64_t>(P::den / P::num);
#endif
}

inline double ns_per_tick() noexcept {
    static const double k = 1e9 / static_cast<double>(freq_hz());
    return k;
}

inline double to_ns(uint64_t ticks) noexcept { return static_cast<double>(ticks) * ns_per_tick(); }

}  // namespace llt::clk
