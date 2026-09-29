// Verifies "no allocation on the hot path" by replacing global operator new
// with a counting version and running the pipeline between the on_hot_start and
// on_hot_end hooks, i.e. after every thread is created and every buffer is
// sized, and before anything is torn down.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <atomic>
#include <cstdlib>
#include <new>

#include "doctest.h"
#include "llt/pipeline.hpp"
#include "llt/synth.hpp"

namespace {
std::atomic<bool> g_counting{false};
std::atomic<uint64_t> g_allocs{0};
}  // namespace

void* operator new(std::size_t n) {
    if (g_counting.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    if (g_counting.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
    const std::size_t al = static_cast<std::size_t>(a);
    if (void* p = std::aligned_alloc(al, (n + al - 1) / al * al)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

using namespace llt;

namespace {
PipelineConfig counting_config() {
    PipelineConfig cfg;
    cfg.pace = false;
    cfg.qos = Qos::Default;
    cfg.on_hot_start = [] {
        g_allocs.store(0);
        g_counting.store(true);
    };
    cfg.on_hot_end = [] { g_counting.store(false); };
    return cfg;
}
const Capture& cap() {
    static const Capture c = [] {
        SynthParams p;
        p.messages = 100'000;
        return capture_from_bytes(generate_capture(p));
    }();
    return c;
}
}  // namespace

TEST_CASE("optimized threaded pipeline performs zero heap allocations while running") {
    const RunResult r = run_threaded<ArrayBook<FlatOrderMap>, SpscRing, false>(cap(), counting_config());
    CHECK(r.orders > 0);
    CHECK(g_allocs.load() == 0);
}

TEST_CASE("inline pipeline performs zero heap allocations while running") {
    const RunResult r = run_inline<ArrayBook<FlatOrderMap>>(cap(), counting_config());
    CHECK(r.orders > 0);
    CHECK(g_allocs.load() == 0);
}

TEST_CASE("the counter works: heap-message and map-book variants do allocate") {
    run_threaded<ArrayBook<FlatOrderMap>, SpscRing, true>(cap(), counting_config());
    CHECK(g_allocs.load() >= cap().message_count);  // one new per market-data message
    run_threaded<MapBook, SpscRing, false>(cap(), counting_config());
    CHECK(g_allocs.load() > 1000);  // map nodes for orders and levels
}
