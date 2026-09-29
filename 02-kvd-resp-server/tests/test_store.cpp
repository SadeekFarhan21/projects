#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>

#include "store.h"

using namespace kv;

TEST(Store, SetGetOverwrite) {
    Store s;
    s.set("a", "1", kNoExpiry);
    ASSERT_NE(s.find("a", 0), nullptr);
    EXPECT_EQ(s.find("a", 0)->value, "1");
    s.set("a", "2", kNoExpiry);
    EXPECT_EQ(s.find("a", 0)->value, "2");
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(s.find("missing", 0), nullptr);
}

TEST(Store, LazyExpiryDeletesOnAccess) {
    Store s;
    s.set("k", "v", /*expire_at=*/1000);
    EXPECT_EQ(s.pttl("k", 400), 600);
    EXPECT_NE(s.find("k", 999), nullptr);
    EXPECT_EQ(s.find("k", 1000), nullptr);  // expiry instant is exclusive
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.ttl_count(), 0u);
    EXPECT_TRUE(s.check_invariants());
}

TEST(Store, SetClearsTtlUnlessKeepTtl) {
    Store s;
    s.set("k", "v", 1000);
    s.set("k", "v2", kNoExpiry, /*keep_ttl=*/true);
    EXPECT_EQ(s.pttl("k", 0), 1000);
    s.set("k", "v3", kNoExpiry);
    EXPECT_EQ(s.pttl("k", 0), -1);
    EXPECT_EQ(s.ttl_count(), 0u);
    EXPECT_TRUE(s.check_invariants());
}

TEST(Store, DelReportsOnlyLiveKeys) {
    Store s;
    s.set("live", "v", kNoExpiry);
    s.set("dead", "v", 10);
    EXPECT_TRUE(s.del("live", 100));
    EXPECT_FALSE(s.del("dead", 100));  // already logically expired
    EXPECT_FALSE(s.del("never", 100));
    EXPECT_EQ(s.size(), 0u);
}

TEST(Store, KeysHidesExpired) {
    Store s;
    s.set("a1", "", kNoExpiry);
    s.set("a2", "", 10);
    s.set("b1", "", kNoExpiry);
    std::vector<std::string> out;
    s.keys("a*", 100, out);
    EXPECT_EQ(out, (std::vector<std::string>{"a1"}));
}

TEST(Store, ActiveExpiryReclaimsMostExpiredKeys) {
    Store s;
    // 10k keys expiring at t=100, 10k that never expire, 1k expiring much later.
    for (int i = 0; i < 10000; ++i) s.set("dead" + std::to_string(i), "v", 100);
    for (int i = 0; i < 10000; ++i) s.set("live" + std::to_string(i), "v", kNoExpiry);
    for (int i = 0; i < 1000; ++i) s.set("later" + std::to_string(i), "v", 1'000'000);
    EXPECT_EQ(s.ttl_count(), 11000u);

    // Run cycles with a generous budget until the expired fraction of the
    // TTL population drops below the 25% stop threshold.
    size_t total = 0;
    for (int cycle = 0; cycle < 200; ++cycle) total += s.active_expire_cycle(200, 1'000'000).expired;
    EXPECT_TRUE(s.check_invariants());
    // With 1000 unexpired TTL keys left, the algorithm stops once expired keys
    // are <= ~25% of samples, so at most roughly 1000/3 dead keys can survive.
    size_t dead_left = s.ttl_count() - 1000;
    EXPECT_GT(total, 9000u);
    EXPECT_LT(dead_left, 1000u);
    EXPECT_EQ(s.size(), 10000u + 1000u + dead_left);
}

TEST(Store, ActiveExpiryRespectsNow) {
    Store s;
    for (int i = 0; i < 100; ++i) s.set("k" + std::to_string(i), "v", 5000);
    auto st = s.active_expire_cycle(1000, 1'000'000);
    EXPECT_EQ(st.expired, 0u);
    EXPECT_EQ(st.rounds, 1u);
    EXPECT_EQ(s.size(), 100u);
}

TEST(Store, RandomizedInvariants) {
    // Random mix of operations, checking the map/ttl-vector invariant after each.
    Store s;
    std::mt19937 rng(7);
    int64_t now = 0;
    for (int step = 0; step < 20000; ++step) {
        std::string key = "k" + std::to_string(rng() % 200);
        switch (rng() % 7) {
            case 0: s.set(key, "v", kNoExpiry); break;
            case 1: s.set(key, "v", now + static_cast<int64_t>(rng() % 50)); break;
            case 2: s.del(key, now); break;
            case 3: s.find(key, now); break;
            case 4: s.set_expiry(key, (rng() % 2) ? kNoExpiry : now + 10, now); break;
            case 5: s.active_expire_cycle(now, 1000); break;
            case 6: now += rng() % 5; break;
        }
        std::string why;
        ASSERT_TRUE(s.check_invariants(&why)) << "step " << step << ": " << why;
    }
}

TEST(Store, PurgeExpiredRemovesExactlyTheDeadKeys) {
    Store s;
    for (int i = 0; i < 500; ++i) s.set("dead" + std::to_string(i), "v", 10);
    for (int i = 0; i < 300; ++i) s.set("ttl" + std::to_string(i), "v", 1000);
    for (int i = 0; i < 200; ++i) s.set("plain" + std::to_string(i), "v", kNoExpiry);
    EXPECT_EQ(s.purge_expired(100), 500u);
    EXPECT_EQ(s.size(), 500u);
    EXPECT_EQ(s.ttl_count(), 300u);
    EXPECT_TRUE(s.check_invariants());
}
