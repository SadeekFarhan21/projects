#include <gtest/gtest.h>

#include <map>
#include <random>

#include "mdp/book.hpp"

using namespace mdp;

namespace {
const Symbol kZiext = make_symbol("ZIEXT");

PriceLevelUpdate plu(char side, int64_t px, uint32_t size, bool complete, int64_t ts = 1) {
  return {static_cast<uint8_t>(side == 'B' ? '8' : '5'), static_cast<uint8_t>(complete ? 1 : 0), ts, kZiext, size,
          px};
}
}  // namespace

// The worked example in the DEEP spec ("Consuming Price Level Update Messages").
TEST(Book, SpecExampleHoldsBboDuringTransition) {
  SymbolBook b;
  b.apply(plu('S', 253000, 100, true));
  b.apply(plu('S', 252000, 100, true));
  b.apply(plu('S', 251000, 100, true));
  b.apply(plu('B', 250000, 100, true));
  b.apply(plu('B', 249000, 100, true));
  EXPECT_EQ(b.bbo(), (Bbo{250000, 100, 251000, 100}));

  // Step 2: sell 25.10 removed, flags OFF. Book is in transition; BBO unchanged.
  EXPECT_FALSE(b.apply(plu('S', 251000, 0, false)));
  EXPECT_TRUE(b.in_transition());
  EXPECT_EQ(b.bbo(), (Bbo{250000, 100, 251000, 100}));
  // The raw levels have moved already.
  EXPECT_EQ(b.asks().best().price, 252000);

  // Step 3: sell 25.20 removed, flags ON. Transition complete, BBO jumps to 25.30.
  EXPECT_TRUE(b.apply(plu('S', 252000, 0, true)));
  EXPECT_FALSE(b.in_transition());
  EXPECT_EQ(b.bbo(), (Bbo{250000, 100, 253000, 100}));
  EXPECT_EQ(b.asks().depth(), 1u);
  EXPECT_EQ(b.bids().depth(), 2u);
}

TEST(Book, SideOrderingAndUpdates) {
  SideBook bids(true), asks(false);
  for (int64_t px : {100, 300, 200}) bids.set(px, 10);
  for (int64_t px : {500, 400, 600}) asks.set(px, 10);
  EXPECT_EQ(bids.best().price, 300);
  EXPECT_EQ(asks.best().price, 400);
  EXPECT_EQ(bids.level(1).price, 200);
  EXPECT_EQ(asks.level(2).price, 600);
  EXPECT_EQ(bids.set(300, 7), LevelChange::Update);
  EXPECT_EQ(bids.best().size, 7u);
  EXPECT_EQ(bids.set(300, 0), LevelChange::Delete);
  EXPECT_EQ(bids.best().price, 200);
  EXPECT_EQ(bids.set(12345, 0), LevelChange::DeleteMissing);
}

TEST(Book, BboChangeOnlyReportedAtEventEnd) {
  SymbolBook b;
  EXPECT_FALSE(b.apply(plu('B', 100, 5, false)));
  EXPECT_FALSE(b.apply(plu('S', 200, 5, false)));
  EXPECT_EQ(b.bbo(), Bbo{});
  EXPECT_TRUE(b.apply(plu('S', 210, 5, true)));
  EXPECT_EQ(b.bbo(), (Bbo{100, 5, 200, 5}));
  // An event that does not change the top reports no change.
  EXPECT_FALSE(b.apply(plu('S', 300, 5, true)));
  EXPECT_EQ(b.counters().events_completed, 2u);
}

TEST(Book, CountsCrossedAndLockedAtEventEnd) {
  SymbolBook b;
  b.apply(plu('B', 100, 5, true));
  b.apply(plu('S', 100, 5, true));
  EXPECT_EQ(b.counters().locked_at_complete, 1u);
  b.apply(plu('S', 90, 5, true));
  EXPECT_EQ(b.counters().crossed_at_complete, 1u);
}

// Randomized comparison against a std::map reference model.
TEST(Book, RandomizedAgainstReferenceModel) {
  std::mt19937_64 rng(7);
  SymbolBook b;
  std::map<int64_t, uint32_t> ref_bid, ref_ask;
  Bbo ref_consistent;
  for (int i = 0; i < 200000; ++i) {
    bool buy = rng() & 1;
    int64_t px = (buy ? 9000 : 10001) + static_cast<int64_t>(rng() % 1000) * (buy ? 1 : 1);
    if (!buy) px = 10001 + static_cast<int64_t>(rng() % 1000);
    uint32_t size = (rng() % 4 == 0) ? 0 : static_cast<uint32_t>(1 + rng() % 500);
    bool complete = (rng() % 3) != 0;
    auto& ref = buy ? ref_bid : ref_ask;
    if (size == 0) ref.erase(px);
    else ref[px] = size;
    b.apply(plu(buy ? 'B' : 'S', px, size, complete));
    if (complete) {
      Bbo r;
      if (!ref_bid.empty()) {
        r.bid_price = ref_bid.rbegin()->first;
        r.bid_size = ref_bid.rbegin()->second;
      }
      if (!ref_ask.empty()) {
        r.ask_price = ref_ask.begin()->first;
        r.ask_size = ref_ask.begin()->second;
      }
      ref_consistent = r;
    }
    ASSERT_EQ(b.bbo(), ref_consistent) << "step " << i;
    ASSERT_EQ(b.bids().depth(), ref_bid.size());
    ASSERT_EQ(b.asks().depth(), ref_ask.size());
  }
}

TEST(SymbolTable, GrowsAndKeepsIds) {
  SymbolTable t;
  std::vector<uint32_t> ids;
  for (int i = 0; i < 50000; ++i) {
    char s[9];
    std::snprintf(s, sizeof s, "S%07d", i);
    ids.push_back(t.get_or_add(make_symbol(s)));
  }
  for (int i = 0; i < 50000; ++i) {
    char s[9];
    std::snprintf(s, sizeof s, "S%07d", i);
    ASSERT_EQ(t.get_or_add(make_symbol(s)), ids[i]);
    ASSERT_EQ(t.symbol(ids[i]), make_symbol(s));
  }
  EXPECT_EQ(t.size(), 50000u);
}
