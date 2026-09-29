#include <cstring>
#include <vector>

#include "doctest.h"
#include "llt/capture.hpp"
#include "llt/itch.hpp"
#include "llt/ouch.hpp"
#include "llt/synth.hpp"

using namespace llt;

TEST_CASE("big-endian helpers round trip") {
    uint8_t b[8];
    itch::put16(b, 0xBEEF);
    CHECK(b[0] == 0xBE);
    CHECK(itch::be16(b) == 0xBEEF);
    itch::put32(b, 0x01020304u);
    CHECK(b[0] == 1);
    CHECK(b[3] == 4);
    CHECK(itch::be32(b) == 0x01020304u);
    itch::put48(b, 0x0000A1B2C3D4E5F6ull);
    CHECK(b[0] == 0xA1);
    CHECK(itch::be48(b) == 0xA1B2C3D4E5F6ull);
    itch::put64(b, 0x1122334455667788ull);
    CHECK(itch::be64(b) == 0x1122334455667788ull);
}

TEST_CASE("each message type encodes to its ITCH 5.0 length and decodes back") {
    std::vector<uint8_t> buf;
    itch::Writer w(buf);
    const char stock[8] = {'A', 'A', 'P', 'L', ' ', ' ', ' ', ' '};
    const uint64_t ts = 34'200'000'000'123ull;
    w.directory(3, ts, stock);
    w.add(3, ts + 1, 77, 'S', 300, stock, 1'901'200);
    w.executed(3, ts + 2, 77, 100, 999);
    w.cancel(3, ts + 3, 77, 50);
    w.del(3, ts + 4, 77);

    itch::FrameCursor cur{buf.data(), buf.data() + buf.size()};
    const uint8_t* m;
    uint16_t len;
    std::vector<uint16_t> lens;
    std::vector<MdEvent> evs;
    while (cur.next(m, len)) {
        lens.push_back(len);
        MdEvent e;
        REQUIRE(itch::decode(m, len, e));
        evs.push_back(e);
    }
    REQUIRE(lens == std::vector<uint16_t>{19, 36, 31, 23, 19});
    CHECK(evs[0].type == 'R');
    CHECK(std::memcmp(evs[0].stock, stock, 8) == 0);
    CHECK(evs[1].type == 'A');
    CHECK(evs[1].locate == 3);
    CHECK(evs[1].exch_ts == ts + 1);
    CHECK(evs[1].order_ref == 77);
    CHECK(evs[1].side == 'S');
    CHECK(evs[1].qty == 300);
    CHECK(evs[1].price == 1'901'200);
    CHECK(evs[2].type == 'E');
    CHECK(evs[2].qty == 100);
    CHECK(evs[3].type == 'X');
    CHECK(evs[3].qty == 50);
    CHECK(evs[4].type == 'D');
    CHECK(evs[4].order_ref == 77);
}

TEST_CASE("add order bytes match the ITCH 5.0 field offsets") {
    std::vector<uint8_t> buf;
    itch::Writer w(buf);
    const char stock[8] = {'M', 'S', 'F', 'T', ' ', ' ', ' ', ' '};
    w.add(0x0102, 0x010203040506ull, 0x1112131415161718ull, 'B', 0x21222324u, stock, 0x31323334u);
    REQUIRE(buf.size() == 2 + 36);
    const uint8_t* p = buf.data() + 2;
    CHECK(buf[0] == 0);
    CHECK(buf[1] == 36);
    CHECK(p[0] == 'A');
    CHECK(p[1] == 0x01);
    CHECK(p[2] == 0x02);
    CHECK(p[5] == 0x01);  // timestamp MSB
    CHECK(p[10] == 0x06);
    CHECK(p[11] == 0x11);  // order ref MSB
    CHECK(p[19] == 'B');
    CHECK(p[20] == 0x21);
    CHECK(p[24] == 'M');
    CHECK(p[32] == 0x31);
    CHECK(p[35] == 0x34);
}

TEST_CASE("decoder rejects unknown and truncated messages; cursor stops at a truncated frame") {
    uint8_t unknown[20] = {'Q'};
    MdEvent e;
    CHECK_FALSE(itch::decode(unknown, sizeof(unknown), e));
    std::vector<uint8_t> buf;
    itch::Writer w(buf);
    const char stock[8] = {'X', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
    w.add(0, 1, 1, 'B', 1, stock, 100);
    CHECK_FALSE(itch::decode(buf.data() + 2, 30, e));  // short body
    buf.pop_back();                                    // truncate the frame
    itch::FrameCursor cur{buf.data(), buf.data() + buf.size()};
    const uint8_t* m;
    uint16_t len;
    CHECK_FALSE(cur.next(m, len));
}

TEST_CASE("OUCH-like enter order round trip") {
    OrderRequest o;
    o.client_id = 12345;
    o.side = 'S';
    o.qty = 200;
    o.price = 4'200'100;
    std::memcpy(o.stock, "NVDA    ", 8);
    uint8_t buf[64];
    const auto n = ouch::encode_enter(buf, o);
    CHECK(n == ouch::kLenEnter);
    ouch::EnterOrder eo;
    REQUIRE(ouch::decode_enter(buf, n, eo));
    CHECK(eo.token == 12345);
    CHECK(eo.side == 'S');
    CHECK(eo.shares == 200);
    CHECK(eo.price == 4'200'100);
    CHECK(std::memcmp(eo.stock, "NVDA    ", 8) == 0);
    CHECK_FALSE(ouch::decode_enter(buf, n - 1, eo));
}

TEST_CASE("synthetic generator is deterministic and well formed") {
    SynthParams p;
    p.messages = 20'000;
    p.seed = 7;
    SynthStats s1, s2;
    const auto a = generate_capture(p, &s1);
    const auto b = generate_capture(p, &s2);
    CHECK(a == b);
    const Capture c = capture_from_bytes(a);
    CHECK(c.message_count == p.messages);
    CHECK(s1.adds > 0);
    CHECK(s1.execs > 0);
    CHECK(s1.cancels > 0);
    CHECK(s1.deletes > 0);
    // Timestamps strictly increase and every frame decodes.
    itch::FrameCursor cur{c.bytes.data(), c.bytes.data() + c.bytes.size()};
    const uint8_t* m;
    uint16_t len;
    uint64_t last = 0;
    std::size_t n = 0;
    while (cur.next(m, len)) {
        MdEvent e;
        REQUIRE(itch::decode(m, len, e));
        CHECK(e.exch_ts > last);
        last = e.exch_ts;
        ++n;
    }
    CHECK(n == p.messages);
}
