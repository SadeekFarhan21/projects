// Shared constants and small helpers used across the pipeline.
#pragma once

#include <cstddef>
#include <cstdint>

namespace llt {

// Apple M-series cores report hw.cachelinesize = 128 (sysctl), and the L2
// prefetcher pulls pairs of 64B lines on many x86 parts as well. Padding to 128
// avoids false sharing on both.
inline constexpr std::size_t kCacheLine = 128;

// Upper bound on distinct stock_locate values the books track. Locates beyond
// this are counted and ignored (see DESIGN.md, invariants).
inline constexpr std::size_t kMaxSymbols = 16;

// Busy-wait hint. On arm64 `yield` is a hint to SMT siblings (none on M4), so
// this is effectively a no-op that keeps the loop from being optimized away.
inline void cpu_relax() noexcept {
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

}  // namespace llt
