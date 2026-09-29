// Baseline player: UCT tree search with uniformly random rollouts and no
// learned knowledge. "MCTS-N" in the results means N rollouts per move.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "c4/position.hpp"

namespace c4 {

class PureMCTS {
 public:
  PureMCTS(int rollouts, std::uint64_t seed, double c_uct = 1.41421356);
  // Most visited root child after `rollouts` iterations.
  int choose_move(const Position& root);

 private:
  struct Node {
    int parent;
    int first_child;  // -1 until expanded
    std::int8_t num_children;
    std::int8_t move;
    std::int8_t terminal;  // 1 if the move into this node ended the game
    float terminal_value;  // value for the player who moved into the node
    std::uint32_t visits;
    float value_sum;  // from the view of the player who moved into the node
  };
  float rollout(Position p);
  void expand(int idx, const Position& p);
  int select_child(int idx);
  int rollouts_;
  double c_uct_;
  std::mt19937_64 rng_;
  std::vector<Node> nodes_;
};

// Uniform random legal move.
int random_move(const Position& p, std::mt19937_64& rng);

}  // namespace c4
