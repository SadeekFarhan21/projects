#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "c4/solver.hpp"

using namespace c4;

namespace {
// Random non-terminal position with exactly `stones` stones.
bool random_position(std::mt19937_64& rng, int stones, Position& out) {
  Position p;
  while (p.moves() < stones) {
    int legal[kWidth], n = 0;
    for (int c = 0; c < kWidth; ++c)
      if (p.can_play(c)) legal[n++] = c;
    p.play(legal[rng() % n]);
    if (p.is_terminal()) return false;
  }
  out = p;
  return true;
}
}  // namespace

TEST(Solver, ImmediateWinScore) {
  Solver s(16);
  Position p = Position::from_moves("121212");  // X wins with 7th move (its 4th stone)
  EXPECT_EQ(s.solve(p), (kCells + 1 - 6) / 2);   // 18
  EXPECT_TRUE(p.is_winning_move(0));
}

TEST(Solver, MatchesBruteForceLateGame) {
  std::mt19937_64 rng(11);
  Solver s(18);
  int checked = 0;
  while (checked < 150) {
    Position p;
    if (!random_position(rng, 32 + static_cast<int>(rng() % 6), p)) continue;
    EXPECT_EQ(s.solve(p), brute_force_score(p)) << p.to_string();
    ++checked;
  }
}

TEST(Solver, AnalyzeConsistentWithSolve) {
  std::mt19937_64 rng(5);
  Solver s(20);
  int checked = 0;
  while (checked < 40) {
    Position p;
    if (!random_position(rng, 18 + static_cast<int>(rng() % 12), p)) continue;
    auto sc = s.analyze(p);
    EXPECT_EQ(*std::max_element(sc.begin(), sc.end()), s.solve(p));
    for (int c = 0; c < kWidth; ++c) EXPECT_EQ(sc[c] == kInvalidScore, !p.can_play(c));
    ++checked;
  }
}

TEST(Solver, MirrorSymmetry) {
  std::mt19937_64 rng(9);
  Solver s(20);
  int checked = 0;
  while (checked < 30) {
    Position p;
    if (!random_position(rng, 16 + static_cast<int>(rng() % 10), p)) continue;
    EXPECT_EQ(s.solve(p), s.solve(p.mirrored()));
    ++checked;
  }
}

TEST(Solver, TableDoesNotChangeAnswers) {
  // Solving with a warm table must give the same scores as a cold one.
  std::mt19937_64 rng(21);
  Solver warm(20);
  int checked = 0;
  while (checked < 20) {
    Position p;
    if (!random_position(rng, 20 + static_cast<int>(rng() % 8), p)) continue;
    Solver cold(20);
    EXPECT_EQ(warm.solve(p), cold.solve(p));
    ++checked;
  }
}
