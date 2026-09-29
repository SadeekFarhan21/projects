#include <cstdint>
#include <memory>
#include <thread>

#include "doctest.h"
#include "llt/queues.hpp"

using namespace llt;

TEST_CASE_TEMPLATE("queue single-thread semantics: empty, full, FIFO, wraparound", Q,
                   SpscRing<int, 8>, SpscNaive<int, 8>, MutexQueue<int, 8>) {
    auto q = std::make_unique<Q>();
    int v = -1;
    CHECK_FALSE(q->try_pop(v));
    for (int i = 0; i < 8; ++i) CHECK(q->try_push(i));
    CHECK_FALSE(q->try_push(99));  // full at exactly N
    for (int i = 0; i < 8; ++i) {
        REQUIRE(q->try_pop(v));
        CHECK(v == i);
    }
    CHECK_FALSE(q->try_pop(v));
    // Many laps around the ring.
    for (int i = 0; i < 1000; ++i) {
        REQUIRE(q->try_push(i));
        REQUIRE(q->try_push(i + 1));
        REQUIRE(q->try_pop(v));
        CHECK(v == i);
        REQUIRE(q->try_pop(v));
        CHECK(v == i + 1);
    }
}

TEST_CASE("SpscRing layout keeps producer and consumer indices on separate cache lines") {
    CHECK(alignof(SpscRing<int, 8>) == kCacheLine);
    CHECK(sizeof(SpscRing<int, 8>) >= 3 * kCacheLine);
}

TEST_CASE_TEMPLATE("queue two-thread stress preserves order and loses nothing", Q,
                   SpscRing<uint64_t, 1024>, SpscNaive<uint64_t, 1024>, MutexQueue<uint64_t, 1024>) {
    constexpr uint64_t kN = 500'000;
    auto q = std::make_unique<Q>();
    std::thread prod([&] {
        for (uint64_t i = 1; i <= kN; ++i) push_spin(*q, i);
    });
    uint64_t expect = 1, sum = 0, v = 0;
    bool ordered = true;
    while (expect <= kN) {
        pop_spin(*q, v);
        ordered &= (v == expect);
        sum += v;
        ++expect;
    }
    prod.join();
    CHECK(ordered);
    CHECK(sum == kN * (kN + 1) / 2);
}
