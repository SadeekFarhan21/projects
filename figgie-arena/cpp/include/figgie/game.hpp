// Exact Figgie rules engine.
//
// Rules (figgie.com/how-to-play): 40 cards, suits of 12, 10, 10 and 8 cards
// assigned uniformly at random. The goal suit is the other suit of the same
// colour as the 12-card suit (so it has 10 or 8 cards). Every player antes an
// equal share of a 200 pot and is dealt 40 / n cards. Trading is a continuous
// one-lot double auction per suit; after every trade all quotes in all suits
// are cancelled. At the end each goal-suit card pays 10 from the pot and the
// remainder goes to whoever holds the most goal-suit cards, ties split evenly.
//
// Time is discretised into ticks. In each tick every player gets exactly one
// action, and the order in which players act is a fresh uniform permutation
// drawn from the game's RNG (so no seat has a systematic speed advantage).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "figgie/book.hpp"
#include "figgie/rng.hpp"
#include "figgie/types.hpp"

namespace figgie {

struct Settlement {
  // Payouts in units of 1/60 of a chip, so every tie split (by 1..5 players)
  // is exact and money is conserved without rounding.
  std::array<int64_t, kMaxPlayers> payout_x60{};
  std::array<int, kMaxPlayers> goal_cards{};
  int remainder = 0;  // pot minus 10 per goal card
  int n_winners = 0;
};

class Game {
 public:
  explicit Game(Config cfg = Config{}, uint64_t seed = 1);

  // Deal a fresh random game.
  void reset(uint64_t seed);
  // Deal a specific game (for tests): suit counts must be a permutation of
  // {12, 10, 10, 8} and hands must sum to them.
  void reset_with_deal(const Counts& suit_counts, const std::vector<Counts>& hands, uint64_t seed);

  Status apply(int player, const Action& a);

  // Advance the clock by one tick. When the last tick ends the game settles.
  void end_tick();
  bool done() const { return done_; }

  // Random permutation for this tick's action order.
  std::array<int, kMaxPlayers> action_order();

  // ---- public information ----
  const Config& config() const { return cfg_; }
  int n_players() const { return cfg_.n_players; }
  int tick() const { return tick_; }
  const SuitBook& book(int suit) const { return books_[suit]; }
  const std::vector<Trade>& trades() const { return trades_; }
  int cash(int p) const { return cash_[p]; }  // chips are public in Figgie
  int pot() const { return pot_; }

  // ---- private information (engine, tests and a player's own view) ----
  int hand(int p, int s) const { return hands_[p][s]; }
  const Counts& hand_counts(int p) const { return hands_[p]; }
  const Counts& initial_hand(int p) const { return initial_hands_[p]; }
  const Counts& suit_counts() const { return suit_counts_; }
  int goal_suit() const { return goal_; }

  // ---- settlement ----
  const Settlement& settlement() const { return settlement_; }
  // Net chip result for the game including the ante, in chips.
  double pnl(int p) const;
  // Pure function used by the engine, exposed for property tests.
  static Settlement settle(const Config& cfg, int goal, const std::vector<Counts>& hands);

  // Checks every rule invariant; returns an empty string when all hold.
  std::string check_invariants() const;

  uint64_t rejected() const { return rejected_; }

 private:
  Status place(int player, int suit, SuitBook::Side side, int price);
  Status take(int player, int suit, SuitBook::Side resting_side);
  void execute(int suit, int buyer, int seller, int price);
  void reconcile_after_trade(int buyer, int seller, int suit);
  void finalize();

  Config cfg_;
  Rng rng_;
  Counts suit_counts_{};
  int goal_ = 0;
  std::array<Counts, kMaxPlayers> hands_{};
  std::array<Counts, kMaxPlayers> initial_hands_{};
  std::array<int, kMaxPlayers> cash_{};
  int pot_ = 0;
  std::array<SuitBook, kSuits> books_{};
  std::vector<Trade> trades_;
  uint64_t seq_ = 0;
  int tick_ = 0;
  bool done_ = false;
  uint64_t rejected_ = 0;
  Settlement settlement_{};
};

}  // namespace figgie
