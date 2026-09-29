// Tournament runner, invariant fuzzer and vectorised RL environment.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "figgie/bots.hpp"
#include "figgie/game.hpp"

namespace figgie {

struct MatchResults {
  int n_games = 0;
  int n_bots = 0;
  int n_checkpoints = 0;
  std::vector<int> deal;            // [G] deal index (rotations share a deal)
  std::vector<int> rotation;        // [G]
  std::vector<double> pnl;          // [G, n_bots], indexed by lineup position
  std::vector<double> brier;        // [G, n_bots, n_checkpoints]
  std::vector<int> goal_cards_end;  // [G, n_bots]
  std::vector<int> n_trades;        // [G]
  uint64_t invariant_violations = 0;
  std::string first_violation;
};

// Plays n_deals deals. With rotate, each deal is replayed n_players times with
// the lineup cyclically shifted through the seats (duplicate format), so every
// bot sees every seat and every hand. Deterministic for a given seed
// regardless of thread count.
MatchResults run_games(const std::vector<std::string>& lineup, int n_deals, uint64_t seed, const Config& cfg,
                       int n_checkpoints, bool rotate, bool check_invariants, int threads);

struct FuzzResults {
  uint64_t games = 0;
  uint64_t actions = 0;
  uint64_t trades = 0;
  uint64_t rejected = 0;
  uint64_t violations = 0;
  std::string first_violation;
};

// Random configs (4 or 5 players, clear-on-trade on or off, random length),
// a mix of fully random (often invalid) actions and scripted bots, with every
// invariant checked after every single action.
// With allow_bayes false the Bayesian bot is excluded from the seat mix, which
// makes the fuzzer about 100x faster (the exact posterior dominates the cost).
FuzzResults fuzz(uint64_t n_games, uint64_t seed, int threads, bool allow_bayes = true);

class VecEnv {
 public:
  static constexpr int kObsDim = 62;
  VecEnv(int num_envs, int n_learners, const std::vector<std::string>& opponents, const Config& cfg, uint64_t seed,
         bool check_invariants);

  int num_envs() const { return num_envs_; }
  int n_learners() const { return n_learners_; }
  int obs_dim() const { return kObsDim; }

  // obs: [N, L, D]
  void reset(float* obs);
  // actions: [N, L, 3] as (type, suit, price); reward: [N, L]; done: [N]
  void step(const int32_t* actions, float* obs, float* reward, uint8_t* done);

  uint64_t steps() const { return steps_; }
  uint64_t episodes() const { return episodes_; }
  uint64_t violations() const { return violations_; }
  uint64_t rejected_learner_actions() const { return rejected_; }
  std::array<double, kSuits> bayes_goal_probs(int env, int learner) const;
  const Game& game(int env) const { return *games_[env]; }

 private:
  void reset_env(int e);
  void write_obs(int e, float* obs) const;

  int num_envs_, n_learners_;
  Config cfg_;
  uint64_t seed_;
  bool check_;
  std::vector<std::unique_ptr<Game>> games_;
  std::vector<std::vector<std::unique_ptr<Bot>>> bots_;  // [N][n_players - L]
  std::vector<std::string> opponents_;
  uint64_t episode_counter_ = 0, steps_ = 0, episodes_ = 0, violations_ = 0, rejected_ = 0;
};

}  // namespace figgie
