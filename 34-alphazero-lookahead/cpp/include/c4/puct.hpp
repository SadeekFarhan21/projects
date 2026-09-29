// PUCT search (AlphaZero style) with the network evaluation pulled out of the
// search loop. A PuctTree never calls a network. It hands out one leaf at a
// time (select_leaf) and later receives the prior and value for that leaf
// (expand_and_backup). BatchedGames runs many trees side by side and packs one
// leaf per tree into a single batch, which Python sends through the network
// in one forward pass.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "c4/position.hpp"
#include "c4/pure_mcts.hpp"

namespace c4 {

struct PuctConfig {
  int num_sims = 100;           // leaf evaluations per move (root expansion counts as one)
  float c_puct = 1.5f;          // exploration constant
  float dirichlet_alpha = 1.0f; // root noise concentration
  float dirichlet_eps = 0.25f;  // root noise weight (0 disables noise)
  int temp_moves = 10;          // plies (whole game count) sampled from visit counts
};

class PuctTree {
 public:
  struct Node {
    Position pos;
    int parent;
    int first_child;  // -1 until expanded
    std::int8_t num_children;
    std::int8_t move;
    std::int8_t terminal;
    float terminal_value;  // for the player who moved into this node
    float prior;
    std::uint32_t visits;
    float value_sum;  // for the player who moved into this node
  };

  void reset(const Position& root);

  // Descends from the root with PUCT until it reaches an unexpanded,
  // non-terminal node, backing up terminal nodes along the way (each counts as
  // a simulation). Returns that node's index, or -1 once `target_sims`
  // simulations are done.
  int select_leaf(int target_sims, const PuctConfig& cfg);

  // `policy` holds 7 probabilities for the leaf's side to move (illegal
  // entries are ignored and the rest renormalized); `value` is in [-1, 1]
  // from the same side's view.
  void expand_and_backup(int leaf, const float* policy, float value, const PuctConfig& cfg,
                         std::mt19937_64& rng);

  int sims() const { return sims_; }
  const Node& node(int i) const { return nodes_[i]; }
  const Position& root_position() const { return nodes_[0].pos; }

  // Visit share per column (priors if nothing was visited, e.g. num_sims=1).
  std::array<float, kWidth> policy_target() const;
  // Samples from visit counts if `sample`, otherwise the most visited move
  // (ties broken by prior).
  int choose_move(bool sample, std::mt19937_64& rng) const;
  // Mean value of the root from the side to move's view.
  float root_value() const;

 private:
  void backup(int idx, float value_for_mover);
  int select_child(int idx, const PuctConfig& cfg) const;
  std::vector<Node> nodes_;
  int sims_ = 0;
};

struct GameRecord {
  int winner;    // -1 draw, 0 first player, 1 second player
  int net_side;  // side the network played (-1 in self-play)
  std::string moves;  // 1-based column digits
};

// Many games in lockstep. In self-play mode (opponent_rollouts == 0) the
// network plays both sides and every network move becomes a training sample.
// In evaluation mode the network plays side (game_index % 2) and a PureMCTS
// with `opponent_rollouts` rollouts plays the other side.
class BatchedGames {
 public:
  BatchedGames(int num_slots, int max_games, PuctConfig cfg, std::uint64_t seed,
               int opponent_rollouts = 0, std::vector<std::string> openings = {});

  // Writes up to num_slots observations (kObsSize floats each) and returns how
  // many were written. 0 means all games are finished.
  int gather(float* obs_out);
  // One policy row (7 floats) and one value per gathered observation.
  void scatter(const float* policy, const float* value);

  bool done() const { return finished_games_ >= max_games_; }
  int capacity() const { return static_cast<int>(slots_.size()); }
  int finished_games() const { return finished_games_; }
  std::int64_t total_moves() const { return total_moves_; }
  std::int64_t total_evals() const { return total_evals_; }

  // Training samples produced since the last drain.
  struct Samples {
    std::vector<float> obs;  // n * kObsSize
    std::vector<float> pi;   // n * kWidth
    std::vector<float> z;    // n
  };
  Samples drain_samples();
  std::vector<GameRecord> drain_games();

 private:
  struct Slot {
    PuctTree tree;
    Position pos;
    int game_index = -1;
    int net_side = -1;
    bool active = false;
    int pending = -1;
    std::string moves;
    std::vector<float> obs;
    std::vector<float> pi;
    std::vector<int> side;
    std::unique_ptr<PureMCTS> opponent;
  };
  void start_game(Slot& s);
  // Applies a move to the slot's game; returns true if the game ended.
  bool apply_move(Slot& s, int col);
  void finish_game(Slot& s, int winner);
  // Advances a slot until it needs a network evaluation. Returns the leaf.
  int advance(Slot& s);

  PuctConfig cfg_;
  int max_games_;
  int opponent_rollouts_;
  std::vector<std::string> openings_;
  std::mt19937_64 rng_;
  std::vector<Slot> slots_;
  std::vector<int> row_slot_;
  int started_games_ = 0;
  int finished_games_ = 0;
  std::int64_t total_moves_ = 0;
  std::int64_t total_evals_ = 0;
  Samples samples_;
  std::vector<GameRecord> games_;
};

// Runs a PUCT search for each of a fixed list of positions and records the
// root visit distribution. Same gather/scatter protocol as BatchedGames.
class BatchedAnalysis {
 public:
  BatchedAnalysis(std::vector<Position> positions, PuctConfig cfg, std::uint64_t seed);
  int gather(float* obs_out);
  void scatter(const float* policy, const float* value);
  bool done() const;
  int capacity() const { return static_cast<int>(trees_.size()); }
  std::vector<std::array<float, kWidth>> policies() const;
  std::vector<int> best_moves() const;
  std::vector<float> root_values() const;

 private:
  PuctConfig cfg_;
  std::mt19937_64 rng_;
  std::vector<PuctTree> trees_;
  std::vector<int> pending_;
  std::vector<int> row_tree_;
};

}  // namespace c4
