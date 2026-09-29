// Event sourcing properties: the log format round-trips, and replaying the
// same inputs always yields the same outputs (determinism), including when
// the replay starts from a fresh process-like engine instance.
#include "exchange/engine.hpp"
#include "exchange/market_data.hpp"
#include "exchange/order_flow.hpp"

#include <gtest/gtest.h>

using namespace exch;

namespace {
std::uint64_t digest_of(const std::vector<InputEvent>& in) {
    MatchingEngine eng(16); // tiny initial capacity: forces pool and map growth
    StreamDigest d;
    std::vector<OutputEvent> buf;
    for (auto& e : in) {
        buf.clear();
        eng.process(e, buf);
        for (auto& o : buf) d.add(o);
    }
    return d.value();
}
} // namespace

TEST(Replay, InputEncodingRoundTrips) {
    FlowConfig c;
    c.seed = 3;
    OrderFlow flow(c);
    for (int i = 0; i < 50'000; ++i) {
        const InputEvent e = flow.next();
        InputEvent back;
        ASSERT_TRUE(decode(encode(e), back)) << encode(e);
        // Fields that do not apply to a kind are not encoded; compare what is.
        EXPECT_EQ(encode(back), encode(e));
    }
}

TEST(Replay, DecodeRejectsGarbage) {
    InputEvent e;
    EXPECT_FALSE(decode("", e));
    EXPECT_FALSE(decode("X 1 2", e));
    EXPECT_FALSE(decode("N 1 Q LMT 100 1", e));
    EXPECT_FALSE(decode("N 1 B WAT 100 1", e));
    EXPECT_FALSE(decode("M 1 100", e));
}

TEST(Replay, SameInputsSameOutputs) {
    FlowConfig c;
    c.seed = 5;
    const auto in = OrderFlow(c).take(100'000);
    const auto d1 = digest_of(in);
    const auto d2 = digest_of(in);
    EXPECT_EQ(d1, d2);
    // And the digest is sensitive: dropping one input changes it.
    auto shorter = in;
    shorter.erase(shorter.begin() + 500);
    EXPECT_NE(digest_of(shorter), d1);
}

TEST(Replay, DecodedLogReplaysIdentically) {
    FlowConfig c;
    c.seed = 6;
    const auto in = OrderFlow(c).take(50'000);
    std::vector<InputEvent> decoded;
    for (auto& e : in) {
        InputEvent back;
        ASSERT_TRUE(decode(encode(e), back));
        decoded.push_back(back);
    }
    EXPECT_EQ(digest_of(in), digest_of(decoded));
}
