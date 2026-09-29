#include "c4/pure_mcts.hpp"

#include <cmath>
#include <stdexcept>

namespace c4 {

int random_move(const Position& p, std::mt19937_64& rng) {
  int legal[kWidth];
  int n = 0;
  for (int c = 0; c < kWidth; ++c)
    if (p.can_play(c)) legal[n++] = c;
  if (n == 0) throw std::logic_error("random_move on full board");
  return legal[std::uniform_int_distribution<int>(0, n - 1)(rng)];
}

PureMCTS::PureMCTS(int rollouts, std::uint64_t seed, double c_uct)
    : rollouts_(rollouts), c_uct_(c_uct), rng_(seed) {}

// Plays random moves to the end. Returns the result for the player who is to
// move in `p`: +1 win, -1 loss, 0 draw.
float PureMCTS::rollout(Position p) {
  const int start_side = p.side_to_move();
  while (true) {
    const int c = random_move(p, rng_);
    p.play(c);
    if (p.last_mover_won()) {
      // last mover is the side that was to move before this play
      const int mover = 1 - p.side_to_move();
      return mover == start_side ? 1.0f : -1.0f;
    }
    if (p.is_full()) return 0.0f;
  }
}

void PureMCTS::expand(int idx, const Position& p) {
  const int first = static_cast<int>(nodes_.size());
  int count = 0;
  for (int c = 0; c < kWidth; ++c) {
    if (!p.can_play(c)) continue;
    Position q = p;
    q.play(c);
    Node ch{idx, -1, 0, static_cast<std::int8_t>(c), 0, 0.0f, 0, 0.0f};
    if (q.last_mover_won()) {
      ch.terminal = 1;
      ch.terminal_value = 1.0f;
    } else if (q.is_full()) {
      ch.terminal = 1;
      ch.terminal_value = 0.0f;
    }
    nodes_.push_back(ch);
    ++count;
  }
  nodes_[idx].first_child = first;
  nodes_[idx].num_children = static_cast<std::int8_t>(count);
}

int PureMCTS::select_child(int idx) {
  const Node& n = nodes_[idx];
  int best = -1;
  double best_score = -1e300;
  const double log_n = std::log(static_cast<double>(n.visits) + 1.0);
  for (int k = 0; k < n.num_children; ++k) {
    const int ci = n.first_child + k;
    const Node& ch = nodes_[ci];
    double score;
    if (ch.visits == 0) {
      // unvisited children first, in random order
      score = 1e9 + std::uniform_real_distribution<double>(0, 1)(rng_);
    } else {
      score = ch.value_sum / ch.visits + c_uct_ * std::sqrt(log_n / ch.visits);
    }
    if (score > best_score) {
      best_score = score;
      best = ci;
    }
  }
  return best;
}

int PureMCTS::choose_move(const Position& root) {
  if (root.is_terminal()) throw std::invalid_argument("choose_move on terminal position");
  nodes_.clear();
  nodes_.reserve(static_cast<std::size_t>(rollouts_) * 2 + 16);
  nodes_.push_back({-1, -1, 0, -1, 0, 0.0f, 0, 0.0f});

  for (int it = 0; it < rollouts_; ++it) {
    int idx = 0;
    Position p = root;
    float value;  // from the view of the player who moved into nodes_[idx]
    while (true) {
      if (nodes_[idx].terminal) {
        value = nodes_[idx].terminal_value;
        break;
      }
      if (nodes_[idx].first_child < 0) {
        expand(idx, p);
        const Node& n = nodes_[idx];
        idx = n.first_child + std::uniform_int_distribution<int>(0, n.num_children - 1)(rng_);
        p.play(nodes_[idx].move);
        // rollout() scores for the side to move in p, the opponent of the
        // player who moved into idx.
        value = nodes_[idx].terminal ? nodes_[idx].terminal_value : -rollout(p);
        break;
      }
      idx = select_child(idx);
      p.play(nodes_[idx].move);
    }

    while (idx >= 0) {
      nodes_[idx].visits += 1;
      nodes_[idx].value_sum += value;
      value = -value;
      idx = nodes_[idx].parent;
    }
  }

  const Node& r = nodes_[0];
  int best_move = -1;
  std::uint32_t best_visits = 0;
  for (int k = 0; k < r.num_children; ++k) {
    const Node& ch = nodes_[r.first_child + k];
    if (best_move < 0 || ch.visits > best_visits) {
      best_visits = ch.visits;
      best_move = ch.move;
    }
  }
  return best_move;
}

}  // namespace c4
