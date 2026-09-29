// Perfect Connect Four solver: negamax with alpha-beta, a transposition table
// keyed on the bitboard key, threat-based move ordering, and a null-window
// search that binary searches the exact score.
//
// Score convention (from the side to move): 0 is a draw. A positive score s
// means the side to move wins with its k-th stone where s = 22 - k, so faster
// wins score higher. A negative score means the side to move loses.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "c4/position.hpp"

namespace c4 {

inline constexpr int kMinScore = -(kCells / 2) + 3;
inline constexpr int kMaxScore = (kCells + 1) / 2 - 3;
// Sentinel used by analyze() for full columns.
inline constexpr int kInvalidScore = -1000;

class TranspositionTable {
 public:
  explicit TranspositionTable(int log2_size = 23);
  void reset();
  void put(u64 key, std::uint8_t value) {
    const std::size_t i = index(key);
    keys_[i] = key;
    values_[i] = value;
  }
  std::uint8_t get(u64 key) const {
    const std::size_t i = index(key);
    return keys_[i] == key ? values_[i] : 0;
  }

 private:
  std::size_t index(u64 key) const {
    return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> shift_);
  }
  int shift_;
  std::vector<u64> keys_;
  std::vector<std::uint8_t> values_;
};

class Solver {
 public:
  explicit Solver(int tt_log2_size = 23) : tt_(tt_log2_size) {}

  // Exact score of the position. The position must not be terminal.
  int solve(const Position& p);

  // Score of playing each column (from the mover's view), kInvalidScore for
  // full columns.
  std::array<int, kWidth> analyze(const Position& p);

  // Best column by score, ties broken toward the center.
  int best_move(const Position& p);

  std::uint64_t node_count() const { return nodes_; }
  void reset() {
    nodes_ = 0;
    tt_.reset();
  }

 private:
  int negamax(const Position& p, int alpha, int beta);
  TranspositionTable tt_;
  std::uint64_t nodes_ = 0;
};

// Plain negamax with no pruning and no table. Exponential; only for testing
// the solver on positions with few empty cells.
int brute_force_score(const Position& p);

}  // namespace c4
