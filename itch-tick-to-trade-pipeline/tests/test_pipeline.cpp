#include <vector>

#include "doctest.h"
#include "llt/pipeline.hpp"
#include "llt/synth.hpp"

using namespace llt;

namespace {
const Capture& test_capture() {
    static const Capture c = [] {
        SynthParams p;
        p.messages = 120'000;
        p.seed = 2024;
        return capture_from_bytes(generate_capture(p));
    }();
    return c;
}

void check_well_formed(const RunResult& r, std::size_t expect_md) {
    CHECK(r.md_messages == expect_md);
    CHECK(r.orders > 50);
    CHECK(r.seq_errors == 0);
    CHECK(r.decode_errors == 0);
    CHECK(r.timings.size() == r.orders);
    CHECK(r.verdicts[static_cast<std::size_t>(RiskVerdict::Accept)] == r.orders);
    CHECK(r.book.unknown_ref == 0);
    CHECK(r.book.out_of_band == 0);
    std::size_t bad_order = 0;
    for (const auto& t : r.timings)
        if (!(t.t_in <= t.t_decide && t.t_decide <= t.t_out && t.t_out <= t.t_rx)) ++bad_order;
    CHECK(bad_order == 0);
}
}  // namespace

TEST_CASE("threaded and inline pipelines send the identical order stream") {
    const Capture& c = test_capture();
    PipelineConfig cfg;
    cfg.pace = false;
    cfg.qos = Qos::Default;
    using Fast = ArrayBook<FlatOrderMap>;
    const RunResult inl = run_inline<Fast>(c, cfg);
    check_well_formed(inl, c.message_count);

    const RunResult thr = run_threaded<Fast, SpscRing, false>(c, cfg);
    check_well_formed(thr, c.message_count);
    REQUIRE(thr.orders == inl.orders);
    CHECK(thr.exch_orders == inl.exch_orders);
    CHECK(thr.verdicts == inl.verdicts);
}

TEST_CASE("every ablation variant makes the same decisions") {
    const Capture& c = test_capture();
    PipelineConfig cfg;
    cfg.pace = false;
    cfg.qos = Qos::Default;
    using Fast = ArrayBook<FlatOrderMap>;
    const RunResult base = run_threaded<Fast, SpscRing, false>(c, cfg);
    const RunResult a = run_threaded<Fast, MutexQueue, false>(c, cfg);
    const RunResult b = run_threaded<Fast, SpscNaive, false>(c, cfg);
    const RunResult d = run_threaded<MapBook, SpscRing, false>(c, cfg);
    const RunResult e = run_threaded<Fast, SpscRing, true>(c, cfg);
    const RunResult f = run_threaded<MapBook, MutexQueue, true>(c, cfg);
    for (const RunResult* r : {&a, &b, &d, &e, &f}) {
        CHECK(r->seq_errors == 0);
        CHECK(r->exch_orders == base.exch_orders);
    }
}

TEST_CASE("paced replay tracks capture time") {
    // 20k messages at ~1 us mean gap is ~20 ms of exchange time; replaying at
    // 1x must take at least that long, and at 4x at least a quarter of it.
    SynthParams p;
    p.messages = 20'000;
    p.seed = 5;
    SynthStats st;
    const Capture c = capture_from_bytes(generate_capture(p, &st));
    const double span_ns = static_cast<double>(st.last_ts - st.first_ts);
    PipelineConfig cfg;
    cfg.qos = Qos::Default;
    cfg.pace = true;
    const RunResult r1 = run_threaded<ArrayBook<FlatOrderMap>, SpscRing, false>(c, cfg);
    CHECK(r1.wall_ns >= span_ns * 0.99);
    cfg.speed = 4.0;
    const RunResult r4 = run_threaded<ArrayBook<FlatOrderMap>, SpscRing, false>(c, cfg);
    CHECK(r4.wall_ns >= span_ns / 4 * 0.99);
    // No upper-bound check: under sanitizers the pipeline is slower than 4x
    // real time, so only "never runs ahead of the capture clock" is portable.
}
