#include "exchange/id_map.hpp"
#include "exchange/order_flow.hpp"

#include <gtest/gtest.h>

#include <unordered_map>

using namespace exch;

TEST(IdMap, BasicInsertFindErase) {
    IdMap m(4);
    EXPECT_EQ(m.find(5), IdMap::kMissing);
    EXPECT_TRUE(m.insert(5, 50));
    EXPECT_FALSE(m.insert(5, 51));
    EXPECT_EQ(m.find(5), 50u);
    EXPECT_TRUE(m.erase(5));
    EXPECT_FALSE(m.erase(5));
    EXPECT_EQ(m.size(), 0u);
}

// Random operations against std::unordered_map, with a small key space so
// probe chains collide, wrap around the table end and get erased from the
// middle (the backward-shift path).
TEST(IdMap, MatchesUnorderedMapUnderChurn) {
    IdMap m(8);
    std::unordered_map<OrderId, std::uint32_t> ref;
    Rng rng(99);
    for (int i = 0; i < 400'000; ++i) {
        const OrderId k = static_cast<OrderId>(rng.range(1, 3000));
        const int op = static_cast<int>(rng.range(0, 2));
        if (op == 0) {
            const auto v = static_cast<std::uint32_t>(rng.range(0, 1 << 30));
            EXPECT_EQ(m.insert(k, v), ref.emplace(k, v).second);
        } else if (op == 1) {
            EXPECT_EQ(m.erase(k), ref.erase(k) == 1);
        } else {
            auto it = ref.find(k);
            EXPECT_EQ(m.find(k), it == ref.end() ? IdMap::kMissing : it->second);
        }
        ASSERT_EQ(m.size(), ref.size());
    }
    for (auto& kv : ref) ASSERT_EQ(m.find(kv.first), kv.second);
}
