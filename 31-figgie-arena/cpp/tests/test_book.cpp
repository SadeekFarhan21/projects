#include <gtest/gtest.h>

#include "figgie/book.hpp"

using namespace figgie;

TEST(SuitBook, PriceThenTimePriority) {
  SuitBook b;
  b.insert(SuitBook::kBidSide, 0, 5, 1);
  b.insert(SuitBook::kBidSide, 1, 7, 2);
  b.insert(SuitBook::kBidSide, 2, 5, 3);
  ASSERT_EQ(b.size(SuitBook::kBidSide), 3);
  EXPECT_EQ(b.at(SuitBook::kBidSide, 0).player, 1);
  EXPECT_EQ(b.at(SuitBook::kBidSide, 1).player, 0);  // earlier at equal price
  EXPECT_EQ(b.at(SuitBook::kBidSide, 2).player, 2);
  EXPECT_TRUE(b.well_formed());
}

TEST(SuitBook, AsksSortAscending) {
  SuitBook b;
  b.insert(SuitBook::kAskSide, 0, 9, 1);
  b.insert(SuitBook::kAskSide, 1, 4, 2);
  b.insert(SuitBook::kAskSide, 2, 6, 3);
  EXPECT_EQ(b.best(SuitBook::kAskSide)->price, 4);
  EXPECT_EQ(b.at(SuitBook::kAskSide, 2).price, 9);
  EXPECT_TRUE(b.well_formed());
}

TEST(SuitBook, ReplaceLosesPriority) {
  SuitBook b;
  b.insert(SuitBook::kBidSide, 0, 5, 1);
  b.insert(SuitBook::kBidSide, 1, 5, 2);
  b.insert(SuitBook::kBidSide, 0, 5, 3);  // re-quote at same price
  EXPECT_EQ(b.size(SuitBook::kBidSide), 2);
  EXPECT_EQ(b.best(SuitBook::kBidSide)->player, 1);
}

TEST(SuitBook, CancelAndCross) {
  SuitBook b;
  b.insert(SuitBook::kAskSide, 3, 10, 1);
  EXPECT_TRUE(b.crosses(SuitBook::kBidSide, 10));
  EXPECT_FALSE(b.crosses(SuitBook::kBidSide, 9));
  EXPECT_FALSE(b.cancel(SuitBook::kAskSide, 2));
  EXPECT_TRUE(b.cancel(SuitBook::kAskSide, 3));
  EXPECT_FALSE(b.best(SuitBook::kAskSide).has_value());
}
