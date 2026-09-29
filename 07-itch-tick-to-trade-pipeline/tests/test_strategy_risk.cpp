#include <cstring>

#include "doctest.h"
#include "llt/risk.hpp"
#include "llt/strategy.hpp"

using namespace llt;

TEST_CASE("imbalance and microprice") {
    Top t{1'000'000, 300, 1'000'100, 100};
    CHECK(imbalance(t) == doctest::Approx(0.5));
    // microprice leans toward the ask when the bid is heavier: mid + spread/2 * I
    CHECK(microprice(t) == doctest::Approx(1'000'050 + 50 * 0.5));
    Top even{1'000'000, 100, 1'000'100, 100};
    CHECK(microprice(even) == doctest::Approx(1'000'050));
}

TEST_CASE("strategy is edge triggered, respects cooldown, and prices at the far touch") {
    StrategyParams p;
    p.threshold = 0.6;
    p.cooldown_ns = 1'000;
    p.order_qty = 100;
    ImbalanceStrategy s(p);
    MdEvent dir;
    dir.type = 'R';
    std::memcpy(dir.stock, "AAPL    ", 8);
    s.on_directory(dir);

    MdEvent e;
    e.type = 'A';
    e.exch_ts = 10'000;
    e.t_in = 42;
    OrderRequest o;
    const Top buy_side{1'000'000, 900, 1'000'100, 100};   // I = 0.8
    const Top sell_side{1'000'000, 100, 1'000'100, 900};  // I = -0.8
    const Top flat{1'000'000, 100, 1'000'100, 100};

    REQUIRE(s.on_book(e, buy_side, o));
    CHECK(o.side == 'B');
    CHECK(o.price == 1'000'100);  // lift the ask
    CHECK(o.qty == 100);
    CHECK(o.t_in == 42);
    CHECK(std::memcmp(o.stock, "AAPL    ", 8) == 0);

    e.exch_ts += 5'000;
    CHECK_FALSE(s.on_book(e, buy_side, o));  // same signal, no new edge
    CHECK_FALSE(s.on_book(e, flat, o));
    REQUIRE(s.on_book(e, sell_side, o));
    CHECK(o.side == 'S');
    CHECK(o.price == 1'000'000);  // hit the bid

    e.exch_ts += 10;  // inside cooldown
    CHECK_FALSE(s.on_book(e, buy_side, o));
    CHECK_FALSE(s.on_book(e, Top{1'000'000, 900, 0, 0}, o));  // one-sided book
}

namespace {
OrderRequest order(char side, uint32_t px, uint32_t qty) {
    OrderRequest o;
    o.side = side;
    o.price = px;
    o.qty = qty;
    return o;
}
}  // namespace

TEST_CASE("risk: fat finger quantity, notional, and price band") {
    RiskLimits l;
    l.max_order_qty = 500;
    l.max_notional = 100'000ull * 10'000;  // $100k
    l.max_dev_bps = 50;
    RiskEngine r(l);
    const Top t{1'000'000, 100, 1'000'200, 100};  // $100.00 / $100.02, mid $100.01
    CHECK(r.check(order('B', 1'000'200, 0), t, 0) == RiskVerdict::FatFingerQty);
    CHECK(r.check(order('B', 1'000'200, 501), t, 0) == RiskVerdict::FatFingerQty);
    CHECK(r.check(order('B', 30'000'000, 400), t, 0) == RiskVerdict::FatFingerNotional);  // $3000 * 400
    // 50 bps of $100.01 is about $0.50: $100.60 is outside, $100.40 inside.
    CHECK(r.check(order('B', 1'006'000, 100), t, 0) == RiskVerdict::FatFingerPrice);
    CHECK(r.check(order('S', 994'000, 100), t, 0) == RiskVerdict::FatFingerPrice);
    CHECK(r.check(order('B', 1'004'000, 100), t, 0) == RiskVerdict::Accept);
    CHECK(r.check(order('B', 1'000'200, 100), Top{1'000'000, 100, 0, 0}, 0) == RiskVerdict::NoReference);
}

TEST_CASE("risk: position limit counts accepted orders as filled, both directions") {
    RiskLimits l;
    l.max_position = 250;
    l.max_orders_per_window = 1000;
    RiskEngine r(l);
    const Top t{1'000'000, 100, 1'000'100, 100};
    CHECK(r.check(order('B', 1'000'100, 100), t, 0) == RiskVerdict::Accept);
    CHECK(r.check(order('B', 1'000'100, 100), t, 1) == RiskVerdict::Accept);
    CHECK(r.check(order('B', 1'000'100, 100), t, 2) == RiskVerdict::PositionLimit);  // would be 300
    CHECK(r.position(0) == 200);
    CHECK(r.check(order('S', 1'000'000, 100), t, 3) == RiskVerdict::Accept);
    CHECK(r.position(0) == 100);
    for (int i = 0; i < 3; ++i) CHECK(r.check(order('S', 1'000'000, 100), t, 4 + i) == RiskVerdict::Accept);
    CHECK(r.position(0) == -200);
    CHECK(r.check(order('S', 1'000'000, 100), t, 9) == RiskVerdict::PositionLimit);
}

TEST_CASE("risk: sliding-window order rate limit") {
    RiskLimits l;
    l.max_position = 1'000'000;
    l.max_orders_per_window = 3;
    l.window_ns = 1'000;
    RiskEngine r(l);
    const Top t{1'000'000, 100, 1'000'100, 100};
    auto buy = order('B', 1'000'100, 1);
    CHECK(r.check(buy, t, 100) == RiskVerdict::Accept);
    CHECK(r.check(buy, t, 200) == RiskVerdict::Accept);
    CHECK(r.check(buy, t, 300) == RiskVerdict::Accept);
    CHECK(r.check(buy, t, 1'099) == RiskVerdict::RateLimit);  // 100 is still inside [99, 1099]
    CHECK(r.check(buy, t, 1'100) == RiskVerdict::Accept);     // 100 aged out
    CHECK(r.check(buy, t, 1'150) == RiskVerdict::RateLimit);  // 200 still inside
    CHECK(r.check(buy, t, 1'200) == RiskVerdict::Accept);
    CHECK(r.position(0) == 5);  // rejected orders did not move position
}
