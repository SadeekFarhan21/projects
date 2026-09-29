// Bots see the game only through View, which exposes public information plus
// the viewing player's own hand. This keeps scripted bots honest: nothing in
// a bot can read another player's cards or the true goal suit.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "figgie/game.hpp"
#include "figgie/posterior.hpp"

namespace figgie {

class View {
 public:
  View(const Game& g, int seat) : g_(g), seat_(seat) {}
  int seat() const { return seat_; }
  int n_players() const { return g_.n_players(); }
  const Config& config() const { return g_.config(); }
  int tick() const { return g_.tick(); }
  int my_hand(int s) const { return g_.hand(seat_, s); }
  const Counts& my_hand_counts() const { return g_.hand_counts(seat_); }
  const Counts& my_initial_hand() const { return g_.initial_hand(seat_); }
  int my_cash() const { return g_.cash(seat_); }
  int cash(int p) const { return g_.cash(p); }
  const SuitBook& book(int s) const { return g_.book(s); }
  const std::vector<Trade>& trades() const { return g_.trades(); }

 private:
  const Game& g_;
  int seat_;
};

class Bot {
 public:
  virtual ~Bot() = default;
  virtual std::string name() const = 0;
  virtual void reset(const View& v, uint64_t seed) = 0;
  virtual Action act(const View& v) = 0;
  // The bot's current belief that each suit is the goal suit (sums to 1).
  virtual std::array<double, kSuits> goal_belief(const View& v) = 0;
};

// Names: "random", "passive", "taker", "bayes".
std::unique_ptr<Bot> make_bot(const std::string& name);
std::vector<std::string> bot_names();

// Tracks per-player lower bounds on dealt holdings from the public tape and
// the exact posterior that follows. Shared by the Bayesian bot and the env.
class TapeTracker {
 public:
  void reset(int n_players, int seat);
  // Consume any new trades; returns true when some lower bound changed.
  bool update(const std::vector<Trade>& trades);
  std::vector<Counts> others_min() const;
  const std::array<Counts, kMaxPlayers>& net_bought() const { return net_; }

 private:
  int n_ = 4, seat_ = 0;
  size_t seen_ = 0;
  std::array<Counts, kMaxPlayers> net_{};      // bought minus sold
  std::array<Counts, kMaxPlayers> min_init_{}; // running max of (sold minus bought)
};

}  // namespace figgie
