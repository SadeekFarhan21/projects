// Randomized differential testing: the engine and the reference matcher are
// fed the same random stream and must emit identical output events. The
// market-data mirror must also equal the engine's book. The Release fuzz
// driver (`exch fuzz`) runs the same check on millions of events; this test
// keeps the counts small enough for the sanitizer build.
#include "exchange/engine.hpp"
#include "exchange/market_data.hpp"
#include "exchange/order_flow.hpp"
#include "exchange/reference.hpp"

#include <gtest/gtest.h>

using namespace exch;

namespace {

struct Stats {
    std::size_t trades = 0, rejects = 0, cancels = 0, modifies = 0, fok_kills = 0;
};

Stats run(const FlowConfig& cfg, std::size_t n) {
    OrderFlow flow(cfg);
    MatchingEngine eng;
    ReferenceMatcher ref;
    MarketDataFeed md;
    md.keep_tape = false;
    std::vector<OutputEvent> a, b;
    Stats st;
    for (std::size_t i = 0; i < n; ++i) {
        const InputEvent e = flow.next();
        a.clear();
        b.clear();
        eng.process(e, a);
        ref.process(e, b);
        if (a != b) {
            std::string msg = "divergence at event " + std::to_string(i) + ": " + encode(e) + "\nengine:\n";
            for (auto& o : a) msg += "  " + encode(o) + "\n";
            msg += "reference:\n";
            for (auto& o : b) msg += "  " + encode(o) + "\n";
            ADD_FAILURE() << msg;
            return st;
        }
        for (auto& o : a) {
            md.on_event(o);
            st.trades += o.kind == OutputKind::Trade;
            st.rejects += o.kind == OutputKind::Rejected;
            st.cancels += o.kind == OutputKind::Canceled;
            st.modifies += o.kind == OutputKind::Modified;
            st.fok_kills += o.kind == OutputKind::Canceled && o.cancel == CancelReason::FokUnfilled;
        }
        if (i % 257 == 0) {
            eng.check_invariants();
            EXPECT_EQ(eng.orders_in_priority(Side::Buy), ref.orders_in_priority(Side::Buy));
            EXPECT_EQ(eng.orders_in_priority(Side::Sell), ref.orders_in_priority(Side::Sell));
            EXPECT_EQ(md.depth(Side::Buy), eng.depth(Side::Buy));
            EXPECT_EQ(md.depth(Side::Sell), eng.depth(Side::Sell));
            if (::testing::Test::HasFailure()) return st;
        }
    }
    eng.check_invariants();
    EXPECT_EQ(md.depth(Side::Buy), ref.depth(Side::Buy));
    EXPECT_EQ(md.depth(Side::Sell), ref.depth(Side::Sell));
    return st;
}

} // namespace

TEST(Differential, DefaultFlow) {
    FlowConfig c;
    c.seed = 7;
    Stats st = run(c, 200'000);
    // Make sure the flow actually exercised the interesting paths.
    EXPECT_GT(st.trades, 10'000u);
    EXPECT_GT(st.rejects, 100u);
    EXPECT_GT(st.modifies, 1'000u);
    EXPECT_GT(st.fok_kills, 100u);
}

TEST(Differential, NarrowBookHeavyCrossing) {
    FlowConfig c;
    c.seed = 11;
    c.half_width = 2;
    c.max_qty = 5;
    run(c, 100'000);
}

TEST(Differential, WideBookManyLevels) {
    FlowConfig c;
    c.seed = 13;
    c.half_width = 200;
    c.max_live = 2000;
    c.p_cancel = 0.15;
    run(c, 60'000);
}

TEST(Differential, AggressiveMix) {
    FlowConfig c;
    c.seed = 17;
    c.p_market = 0.15;
    c.p_ioc = 0.15;
    c.p_fok = 0.15;
    c.p_post = 0.2;
    c.p_modify = 0.25;
    run(c, 100'000);
}

TEST(Differential, ManySeedsShort) {
    for (std::uint64_t s = 100; s < 150; ++s) {
        FlowConfig c;
        c.seed = s;
        c.half_width = 1 + static_cast<Price>(s % 7) * 5;
        c.max_qty = 1 + static_cast<Qty>(s % 5) * 20;
        run(c, 5'000);
        if (HasFailure()) break;
    }
}
