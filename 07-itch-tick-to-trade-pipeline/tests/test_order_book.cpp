#include <cstring>
#include <vector>

#include "doctest.h"
#include "llt/capture.hpp"
#include "llt/itch.hpp"
#include "llt/order_book.hpp"
#include "llt/synth.hpp"

using namespace llt;

namespace {
MdEvent add(uint64_t ref, char side, uint32_t px, uint32_t qty, uint16_t loc = 0) {
    MdEvent e;
    e.type = 'A';
    e.order_ref = ref;
    e.side = side;
    e.price = px;
    e.qty = qty;
    e.locate = loc;
    return e;
}
MdEvent ev(char type, uint64_t ref, uint32_t qty = 0) {
    MdEvent e;
    e.type = type;
    e.order_ref = ref;
    e.qty = qty;
    return e;
}
}  // namespace

TEST_CASE_TEMPLATE("book scenario: adds, partial execution, cancel, delete, best level moves", B,
                   ArrayBook<FlatOrderMap>, ArrayBook<StdOrderMap>, MapBook) {
    B b(100, 4096, 1024);
    b.apply(add(1, 'B', 1'000'000, 100));
    b.apply(add(2, 'B', 999'900, 200));
    b.apply(add(3, 'S', 1'000'200, 300));
    b.apply(add(4, 'S', 1'000'100, 50));
    b.apply(add(5, 'B', 1'000'000, 25));
    Top t = b.top(0);
    CHECK(t == Top{1'000'000, 125, 1'000'100, 50});

    b.apply(ev('E', 4, 20));  // partial execution at best ask
    CHECK(b.top(0).ask_qty == 30);
    b.apply(ev('E', 4, 30));  // fills the rest, best ask level empties
    t = b.top(0);
    CHECK(t.ask_px == 1'000'200);
    CHECK(t.ask_qty == 300);

    b.apply(ev('X', 1, 40));  // partial cancel
    CHECK(b.qty_at(0, 'B', 1'000'000) == 85);
    b.apply(ev('D', 1));
    b.apply(ev('D', 5));  // best bid level now empty
    t = b.top(0);
    CHECK(t.bid_px == 999'900);
    CHECK(t.bid_qty == 200);

    b.apply(ev('X', 2, 1'000));  // over-cancel clamps and removes the order
    CHECK(b.top(0).bid_qty == 0);
    CHECK(b.live_orders() == 1);

    b.apply(ev('D', 12345));  // unknown ref is counted, not fatal
    CHECK(b.stats().unknown_ref == 1);
}

TEST_CASE("ArrayBook bitmap scan crosses 64-level word boundaries") {
    ArrayBook<FlatOrderMap> b(1, 1024, 1024);
    b.apply(add(1, 'B', 500, 10));  // window becomes [0, 1024)
    b.apply(add(2, 'B', 3, 10));    // far below, different bitmap word
    b.apply(add(3, 'S', 510, 10));
    b.apply(add(4, 'S', 1000, 10));
    b.apply(ev('D', 1));
    CHECK(b.top(0).bid_px == 3);
    b.apply(ev('D', 3));
    CHECK(b.top(0).ask_px == 1000);
    b.apply(ev('D', 2));
    b.apply(ev('D', 4));
    CHECK(b.top(0) == Top{});
}

TEST_CASE("ArrayBook counts out-of-band adds and ignores their later events") {
    ArrayBook<FlatOrderMap> b(100, 64, 1024);
    b.apply(add(1, 'B', 1'000'000, 10));
    b.apply(add(2, 'B', 2'000'000, 10));  // far outside the 64-tick window
    CHECK(b.stats().out_of_band == 1);
    CHECK(b.top(0).bid_px == 1'000'000);
    b.apply(ev('D', 2));
    CHECK(b.stats().unknown_ref == 0);
    CHECK(b.live_orders() == 1);
}

TEST_CASE("ArrayBook and MapBook agree event by event on a synthetic capture") {
    SynthParams p;
    p.messages = 150'000;
    p.symbols = 6;
    p.seed = 99;
    const Capture c = capture_from_bytes(generate_capture(p));
    ArrayBook<FlatOrderMap> fast;
    ArrayBook<StdOrderMap> mid;
    MapBook slow;
    itch::FrameCursor cur{c.bytes.data(), c.bytes.data() + c.bytes.size()};
    const uint8_t* m;
    uint16_t len;
    std::size_t n = 0, mismatches = 0;
    while (cur.next(m, len)) {
        MdEvent e;
        REQUIRE(itch::decode(m, len, e));
        fast.apply(e);
        mid.apply(e);
        slow.apply(e);
        if (e.type != 'R') {
            const Top a = fast.top(e.locate), b = mid.top(e.locate), s = slow.top(e.locate);
            if (!(a == s) || !(b == s)) ++mismatches;
            // The generator never produces a crossed book.
            if (s.bid_qty && s.ask_qty) CHECK(s.bid_px < s.ask_px);
        }
        ++n;
    }
    CHECK(mismatches == 0);
    CHECK(n == p.messages);
    for (uint16_t loc = 0; loc < p.symbols; ++loc) {
        CHECK(fast.depth(loc, 'B') == slow.depth(loc, 'B'));
        CHECK(fast.depth(loc, 'S') == slow.depth(loc, 'S'));
    }
    CHECK(fast.live_orders() == slow.live_orders());
    CHECK(fast.stats().unknown_ref == 0);
    CHECK(fast.stats().out_of_band == 0);
    CHECK(slow.stats().unknown_ref == 0);
}
