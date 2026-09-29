#include <gtest/gtest.h>

#include <algorithm>
#include <map>

#include "figgie/game.hpp"

using namespace figgie;

namespace {
// Spades 12, clubs 8 (goal), hearts 10, diamonds 10.
const Counts kSizes{12, 8, 10, 10};
std::vector<Counts> deal4() {
  return {Counts{3, 3, 2, 2}, Counts{3, 2, 3, 2}, Counts{3, 2, 2, 3}, Counts{3, 1, 3, 3}};
}
Game fixed_game(Config cfg = Config{}) {
  Game g(cfg, 1);
  g.reset_with_deal(kSizes, deal4(), 42);
  return g;
}
}  // namespace

TEST(Deal, InvariantsOverManySeeds) {
  Config cfg;
  std::map<int, int> goal_hist;
  std::map<int, int> twelve_hist;
  for (uint64_t seed = 0; seed < 20000; ++seed) {
    Game g(cfg, seed);
    ASSERT_EQ(g.check_invariants(), "") << "seed " << seed;
    Counts sz = g.suit_counts();
    int twelve = static_cast<int>(std::find(sz.begin(), sz.end(), 12) - sz.begin());
    EXPECT_EQ(g.goal_suit(), partner(twelve));
    for (int p = 0; p < 4; ++p) EXPECT_EQ(total(g.hand_counts(p)), 10);
    goal_hist[g.goal_suit()]++;
    twelve_hist[twelve]++;
  }
  for (int s = 0; s < 4; ++s) {  // each suit is the goal about 25% of the time
    EXPECT_NEAR(goal_hist[s] / 20000.0, 0.25, 0.02);
    EXPECT_NEAR(twelve_hist[s] / 20000.0, 0.25, 0.02);
  }
}

TEST(Deal, FivePlayers) {
  Config cfg;
  cfg.n_players = 5;
  Game g(cfg, 3);
  EXPECT_EQ(cfg.ante(), 40);
  EXPECT_EQ(cfg.pot(), 200);
  for (int p = 0; p < 5; ++p) EXPECT_EQ(total(g.hand_counts(p)), 8);
  EXPECT_EQ(g.check_invariants(), "");
}

TEST(Deal, RejectsBadDeals) {
  Game g;
  EXPECT_THROW(g.reset_with_deal(Counts{12, 12, 8, 8}, deal4(), 1), std::invalid_argument);
  auto bad = deal4();
  bad[0][0]++;
  EXPECT_THROW(g.reset_with_deal(kSizes, bad, 1), std::invalid_argument);
}

TEST(Trading, RestingBidHitByIncomingAsk) {
  Game g = fixed_game();
  EXPECT_EQ(g.goal_suit(), kClubs);
  EXPECT_EQ(g.apply(0, {ActType::kBid, kHearts, 6}), Status::kOk);
  EXPECT_EQ(g.apply(1, {ActType::kAsk, kHearts, 4}), Status::kTraded);  // crosses: trades at 6
  ASSERT_EQ(g.trades().size(), 1u);
  EXPECT_EQ(g.trades()[0].price, 6);
  EXPECT_EQ(g.trades()[0].buyer, 0);
  EXPECT_EQ(g.trades()[0].seller, 1);
  EXPECT_EQ(g.hand(0, kHearts), 3);
  EXPECT_EQ(g.hand(1, kHearts), 2);
  EXPECT_EQ(g.cash(0), 300 - 6);
  EXPECT_EQ(g.cash(1), 300 + 6);
  EXPECT_EQ(g.check_invariants(), "");
}

TEST(Trading, TradeClearsAllQuotesInAllSuits) {
  Game g = fixed_game();
  g.apply(2, {ActType::kBid, kSpades, 3});
  g.apply(3, {ActType::kAsk, kDiamonds, 20});
  g.apply(0, {ActType::kAsk, kHearts, 9});
  EXPECT_EQ(g.apply(1, {ActType::kBuy, kHearts, 0}), Status::kTraded);
  for (int s = 0; s < kSuits; ++s) {
    EXPECT_FALSE(g.book(s).best(SuitBook::kBidSide).has_value());
    EXPECT_FALSE(g.book(s).best(SuitBook::kAskSide).has_value());
  }
}

TEST(Trading, WithoutClearRuleOtherQuotesSurvive) {
  Config cfg;
  cfg.clear_on_trade = false;
  Game g = fixed_game(cfg);
  g.apply(2, {ActType::kBid, kSpades, 3});
  g.apply(0, {ActType::kAsk, kHearts, 9});
  g.apply(1, {ActType::kBuy, kHearts, 0});
  EXPECT_TRUE(g.book(kSpades).best(SuitBook::kBidSide).has_value());
  EXPECT_EQ(g.check_invariants(), "");
}

TEST(Trading, ReconcileCancelsUnbackedOffer) {
  Config cfg;
  cfg.clear_on_trade = false;
  Game g = fixed_game(cfg);
  // Player 3 holds one club. Offer it, then sell it into a bid: the offer must go.
  g.apply(3, {ActType::kAsk, kClubs, 30});
  g.apply(0, {ActType::kBid, kClubs, 10});
  EXPECT_EQ(g.apply(3, {ActType::kSell, kClubs, 0}), Status::kTraded);
  EXPECT_EQ(g.hand(3, kClubs), 0);
  EXPECT_FALSE(g.book(kClubs).best(SuitBook::kAskSide).has_value());
  EXPECT_EQ(g.check_invariants(), "");
}

TEST(Trading, TimePriorityAtEqualPrice) {
  Game g = fixed_game();
  g.apply(2, {ActType::kBid, kSpades, 5});
  g.apply(3, {ActType::kBid, kSpades, 5});
  g.apply(0, {ActType::kSell, kSpades, 0});
  EXPECT_EQ(g.trades().back().buyer, 2);
}

TEST(Trading, SelfTradePreventionAndValidation) {
  Game g = fixed_game();
  g.apply(0, {ActType::kAsk, kHearts, 5});
  EXPECT_EQ(g.apply(0, {ActType::kBid, kHearts, 6}), Status::kRejected);
  EXPECT_EQ(g.apply(0, {ActType::kBuy, kHearts, 0}), Status::kRejected);
  EXPECT_EQ(g.apply(1, {ActType::kBid, kHearts, 0}), Status::kRejected);    // price below 1
  EXPECT_EQ(g.apply(1, {ActType::kBid, kHearts, 101}), Status::kRejected);  // above max
  EXPECT_EQ(g.apply(1, {ActType::kBid, kHearts, 99}), Status::kTraded);     // cash 300 is enough
  EXPECT_EQ(g.apply(1, {ActType::kBid, kHearts, 301}), Status::kRejected);
  EXPECT_EQ(g.apply(3, {ActType::kBuy, kSpades, 0}), Status::kNoop);  // empty book
  EXPECT_EQ(g.apply(3, {ActType::kBid, 7, 5}), Status::kRejected);     // bad suit
  EXPECT_EQ(g.apply(9, {ActType::kPass, 0, 0}), Status::kRejected);    // bad player
  EXPECT_EQ(g.check_invariants(), "");
}

TEST(Trading, CannotOfferCardNotHeld) {
  Game g(Config{}, 1);
  g.reset_with_deal(Counts{12, 8, 10, 10},
                    {Counts{4, 0, 3, 3}, Counts{3, 3, 2, 2}, Counts{3, 3, 2, 2}, Counts{2, 2, 3, 3}}, 1);
  EXPECT_EQ(g.apply(0, {ActType::kAsk, kClubs, 5}), Status::kRejected);
  g.apply(1, {ActType::kBid, kClubs, 5});
  EXPECT_EQ(g.apply(0, {ActType::kSell, kClubs, 0}), Status::kRejected);
}

TEST(Trading, CancelSemantics) {
  Game g = fixed_game();
  EXPECT_EQ(g.apply(0, {ActType::kCancelBid, kHearts, 0}), Status::kNoop);
  g.apply(0, {ActType::kBid, kHearts, 5});
  g.apply(0, {ActType::kAsk, kSpades, 15});
  EXPECT_EQ(g.apply(0, {ActType::kCancelAll, 0, 0}), Status::kOk);
  EXPECT_FALSE(g.book(kHearts).best(SuitBook::kBidSide).has_value());
  EXPECT_FALSE(g.book(kSpades).best(SuitBook::kAskSide).has_value());
}

TEST(Settlement, SingleWinner) {
  Config cfg;
  // Goal clubs with 8 cards: remainder = 200 - 80 = 120.
  auto st = Game::settle(cfg, kClubs, {Counts{0, 4, 3, 3}, Counts{4, 2, 2, 2}, Counts{4, 1, 3, 2}, Counts{4, 1, 2, 3}});
  EXPECT_EQ(st.remainder, 120);
  EXPECT_EQ(st.n_winners, 1);
  EXPECT_EQ(st.payout_x60[0], 60 * (40 + 120));
  EXPECT_EQ(st.payout_x60[1], 60 * 20);
  EXPECT_EQ(st.payout_x60[2], 60 * 10);
}

TEST(Settlement, TwoAndThreeWayTies) {
  Config cfg;
  // Goal hearts with 10 cards: remainder 100.
  auto st2 = Game::settle(cfg, kHearts, {Counts{2, 1, 4, 3}, Counts{2, 1, 4, 3}, Counts{4, 3, 1, 2}, Counts{4, 3, 1, 2}});
  EXPECT_EQ(st2.n_winners, 2);
  EXPECT_EQ(st2.payout_x60[0], 60 * (40 + 50));
  auto st3 = Game::settle(cfg, kHearts, {Counts{3, 1, 3, 3}, Counts{3, 1, 3, 3}, Counts{3, 1, 3, 3}, Counts{3, 5, 1, 1}});
  EXPECT_EQ(st3.n_winners, 3);
  EXPECT_EQ(st3.payout_x60[0], 60 * 30 + 2000);  // 100 / 3 chips, exact in 1/60 units
  int64_t sum = 0;
  for (int p = 0; p < 4; ++p) sum += st3.payout_x60[p];
  EXPECT_EQ(sum, 60 * 200);
}

TEST(FullGame, SettlesAndConservesMoney) {
  Config cfg;
  cfg.ticks = 3;
  Game g = fixed_game(cfg);
  g.apply(0, {ActType::kBid, kClubs, 12});
  g.apply(3, {ActType::kSell, kClubs, 0});
  for (int t = 0; t < 3; ++t) g.end_tick();
  ASSERT_TRUE(g.done());
  EXPECT_EQ(g.check_invariants(), "");
  double sum = 0;
  for (int p = 0; p < 4; ++p) sum += g.pnl(p);
  EXPECT_NEAR(sum, 0.0, 1e-9);
  // Player 0 ends with 4 clubs: alone at the top, wins 40 + 120, paid 12, anted 50.
  EXPECT_DOUBLE_EQ(g.pnl(0), -50 - 12 + 40 + 120);
  EXPECT_EQ(g.apply(0, {ActType::kBid, kClubs, 5}), Status::kRejected);  // no trading after end
}
