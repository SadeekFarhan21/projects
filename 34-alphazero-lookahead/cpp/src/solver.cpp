#include "c4/solver.hpp"

#include <algorithm>
#include <stdexcept>

namespace c4 {

TranspositionTable::TranspositionTable(int log2_size)
    : shift_(64 - log2_size),
      keys_(std::size_t{1} << log2_size, 0),
      values_(std::size_t{1} << log2_size, 0) {}

void TranspositionTable::reset() {
  std::fill(keys_.begin(), keys_.end(), 0);
  std::fill(values_.begin(), values_.end(), 0);
}

namespace {

// Small insertion-sorted list. getNext pops the highest score; among equal
// scores the most recently added entry comes out first.
class MoveSorter {
 public:
  void add(u64 move, int score) {
    int pos = size_++;
    for (; pos && entries_[pos - 1].score > score; --pos) entries_[pos] = entries_[pos - 1];
    entries_[pos] = {move, score};
  }
  u64 next() { return size_ ? entries_[--size_].move : 0; }

 private:
  struct Entry {
    u64 move;
    int score;
  };
  std::array<Entry, kWidth> entries_{};
  int size_ = 0;
};

}  // namespace

// Table encoding (value 0 means empty):
//   upper bound b stored as b - kMinScore + 1            in [1, kMaxScore-kMinScore+1]
//   lower bound b stored as b + kMaxScore - 2*kMinScore + 2 above that range
int Solver::negamax(const Position& p, int alpha, int beta) {
  ++nodes_;
  const u64 next = p.possible_non_losing_moves();
  if (next == 0) return -(kCells - p.moves()) / 2;  // every move loses next turn
  if (p.moves() >= kCells - 2) return 0;            // board fills with no win

  int min = -(kCells - 2 - p.moves()) / 2;  // opponent cannot win next move
  if (alpha < min) {
    alpha = min;
    if (alpha >= beta) return alpha;
  }
  int max = (kCells - 1 - p.moves()) / 2;  // we cannot win next move
  if (const int val = tt_.get(p.key())) {
    if (val > kMaxScore - kMinScore + 1) {
      min = val + 2 * kMinScore - kMaxScore - 2;
      if (alpha < min) {
        alpha = min;
        if (alpha >= beta) return alpha;
      }
    } else {
      max = val + kMinScore - 1;
      if (beta > max) {
        beta = max;
        if (alpha >= beta) return beta;
      }
    }
  }
  if (beta > max) {
    beta = max;
    if (alpha >= beta) return beta;
  }

  MoveSorter moves;
  for (int i = kWidth; i--;) {
    if (const u64 move = next & column_mask(kColumnOrder[i])) moves.add(move, p.move_score(move));
  }
  while (const u64 move = moves.next()) {
    Position child(p);
    child.play_bits(move);
    const int score = -negamax(child, -beta, -alpha);
    if (score >= beta) {
      tt_.put(p.key(), static_cast<std::uint8_t>(score + kMaxScore - 2 * kMinScore + 2));
      return score;
    }
    if (score > alpha) alpha = score;
  }
  tt_.put(p.key(), static_cast<std::uint8_t>(alpha - kMinScore + 1));
  return alpha;
}

int Solver::solve(const Position& p) {
  if (p.is_terminal()) throw std::invalid_argument("solve called on terminal position");
  if (p.can_win_next()) return (kCells + 1 - p.moves()) / 2;
  int min = -(kCells - p.moves()) / 2;
  int max = (kCells + 1 - p.moves()) / 2;
  while (min < max) {
    int med = min + (max - min) / 2;
    if (med <= 0 && min / 2 < med)
      med = min / 2;
    else if (med >= 0 && max / 2 > med)
      med = max / 2;
    const int r = negamax(p, med, med + 1);
    if (r <= med)
      max = r;
    else
      min = r;
  }
  return min;
}

std::array<int, kWidth> Solver::analyze(const Position& p) {
  std::array<int, kWidth> scores;
  scores.fill(kInvalidScore);
  for (int c = 0; c < kWidth; ++c) {
    if (!p.can_play(c)) continue;
    if (p.is_winning_move(c)) {
      scores[c] = (kCells + 1 - p.moves()) / 2;
      continue;
    }
    Position child(p);
    child.play(c);
    scores[c] = child.is_full() ? 0 : -solve(child);
  }
  return scores;
}

int Solver::best_move(const Position& p) {
  const auto scores = analyze(p);
  int best = -1;
  for (int c : kColumnOrder) {
    if (scores[c] == kInvalidScore) continue;
    if (best < 0 || scores[c] > scores[best]) best = c;
  }
  return best;
}

int brute_force_score(const Position& p) {
  // p is non-terminal. Score as in Solver: win with k-th own stone => 22 - k.
  int best = -1000;
  for (int c = 0; c < kWidth; ++c) {
    if (!p.can_play(c)) continue;
    Position child(p);
    child.play(c);
    int s;
    if (child.last_mover_won())
      s = (kCells + 1 - p.moves()) / 2;
    else if (child.is_full())
      s = 0;
    else
      s = -brute_force_score(child);
    best = std::max(best, s);
  }
  return best;
}

}  // namespace c4
