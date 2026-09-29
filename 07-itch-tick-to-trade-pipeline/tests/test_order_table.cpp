#include <random>
#include <unordered_map>
#include <vector>

#include "doctest.h"
#include "llt/order_table.hpp"

using namespace llt;

TEST_CASE("FlatOrderMap basic insert/find/erase") {
    FlatOrderMap m(64);
    CHECK(m.find(1) == nullptr);
    CHECK(m.insert(1, OrderInfo{100, 5, 0, 'B', true}));
    CHECK_FALSE(m.insert(1, OrderInfo{}));  // duplicate
    CHECK_FALSE(m.insert(0, OrderInfo{}));  // reserved key
    REQUIRE(m.find(1) != nullptr);
    CHECK(m.find(1)->qty == 5);
    m.find(1)->qty = 3;
    CHECK(m.find(1)->qty == 3);
    CHECK(m.erase(1));
    CHECK_FALSE(m.erase(1));
    CHECK(m.size() == 0);
}

TEST_CASE("FlatOrderMap refuses to fill its last slot") {
    FlatOrderMap m(16);
    int ok = 0;
    for (uint64_t k = 1; k <= 20; ++k) ok += m.insert(k, OrderInfo{}) ? 1 : 0;
    CHECK(ok == 15);
    CHECK(m.find(999) == nullptr);  // lookup of a missing key still terminates
}

TEST_CASE("FlatOrderMap matches std::unordered_map under random churn (backward shift deletion)") {
    // Small table and clustered keys force long probe chains that wrap around
    // the end of the array, which is where backward-shift bugs hide.
    FlatOrderMap m(256);
    std::unordered_map<uint64_t, uint32_t> ref;
    std::mt19937_64 rng(123);
    for (int step = 0; step < 200'000; ++step) {
        const uint64_t key = 1 + rng() % 400;
        const int op = static_cast<int>(rng() % 3);
        if (op == 0 && ref.size() < 180) {
            const uint32_t v = static_cast<uint32_t>(rng());
            const bool a = m.insert(key, OrderInfo{v, v, 0, 'B', true});
            const bool b = ref.emplace(key, v).second;
            REQUIRE(a == b);
        } else if (op == 1) {
            REQUIRE(m.erase(key) == (ref.erase(key) == 1));
        } else {
            OrderInfo* f = m.find(key);
            auto it = ref.find(key);
            REQUIRE((f != nullptr) == (it != ref.end()));
            if (f) REQUIRE(f->price == it->second);
        }
        REQUIRE(m.size() == ref.size());
    }
    for (auto& [k, v] : ref) {
        REQUIRE(m.find(k) != nullptr);
        CHECK(m.find(k)->price == v);
    }
}
