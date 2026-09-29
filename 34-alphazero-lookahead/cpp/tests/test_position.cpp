#include <gtest/gtest.h>

#include <random>
#include <set>

#include "c4/position.hpp"

using namespace c4;

TEST(Position, EmptyBoard) {
  Position p;
  EXPECT_EQ(p.moves(), 0);
  EXPECT_EQ(p.side_to_move(), 0);
  for (int c = 0; c < kWidth; ++c) EXPECT_TRUE(p.can_play(c));
  EXPECT_FALSE(p.is_terminal());
  EXPECT_EQ(p.possible(), kBottomMask);
}

TEST(Position, ColumnFillsAfterSixStones) {
  Position p = Position::from_moves("444444");
  EXPECT_FALSE(p.can_play(3));
  EXPECT_TRUE(p.can_play(2));
  EXPECT_FALSE(p.is_terminal());
  EXPECT_THROW(Position::from_moves("4444444"), std::invalid_argument);
}

TEST(Position, VerticalWin) {
  Position p = Position::from_moves("1212121");
  EXPECT_TRUE(p.last_mover_won());
  EXPECT_TRUE(p.is_terminal());
  EXPECT_THROW(Position::from_moves("12121212"), std::invalid_argument);
}

TEST(Position, HorizontalWin) {
  Position p = Position::from_moves("1122334");
  EXPECT_TRUE(p.last_mover_won());
  Position q = Position::from_moves("112233");
  EXPECT_TRUE(q.is_winning_move(3));
  EXPECT_FALSE(q.is_winning_move(4));
}

TEST(Position, DiagonalWins) {
  // X at (0,0),(1,1),(2,2),(3,3)
  Position up = Position::from_moves("12233434544");
  EXPECT_TRUE(up.last_mover_won());
  // mirrored diagonal
  Position down = Position::from_moves("76655454344");
  EXPECT_TRUE(down.last_mover_won());
}

TEST(Position, NoFalseWinAcrossColumnBoundary) {
  // Stones at top of column 1 and bottom of column 2 must not connect.
  Position p = Position::from_moves("1111122");
  EXPECT_FALSE(p.last_mover_won());
}

TEST(Position, CellAndEncode) {
  Position p = Position::from_moves("45");
  EXPECT_EQ(p.cell(3, 0), 1);
  EXPECT_EQ(p.cell(4, 0), 2);
  EXPECT_EQ(p.cell(0, 0), 0);
  float obs[kObsSize];
  p.encode(obs);
  // first player to move: plane 0 holds X stones
  EXPECT_EQ(obs[0 * kPlaneSize + 0 * kWidth + 3], 1.0f);
  EXPECT_EQ(obs[1 * kPlaneSize + 0 * kWidth + 4], 1.0f);
  EXPECT_EQ(obs[2 * kPlaneSize], 1.0f);
  p.play(0);
  p.encode(obs);
  EXPECT_EQ(obs[0 * kPlaneSize + 0 * kWidth + 4], 1.0f);  // O to move, O stones in plane 0
  EXPECT_EQ(obs[2 * kPlaneSize], 0.0f);
}

TEST(Position, MirrorIsInvolutionAndFlipsColumns) {
  std::mt19937_64 rng(7);
  for (int t = 0; t < 200; ++t) {
    Position p;
    std::string seq;
    for (int k = 0; k < 20 && !p.is_terminal(); ++k) {
      int c = static_cast<int>(rng() % kWidth);
      if (!p.can_play(c)) continue;
      p.play(c);
      seq += static_cast<char>('1' + (kWidth - 1 - c));
    }
    Position m = p.mirrored();
    EXPECT_EQ(m.mirrored().key(), p.key());
    EXPECT_EQ(m.key(), Position::from_moves(seq).key());
    for (int c = 0; c < kWidth; ++c)
      for (int r = 0; r < kHeight; ++r) EXPECT_EQ(p.cell(c, r), m.cell(kWidth - 1 - c, r));
  }
}

TEST(Position, KeysAreUniqueAcrossDistinctBoards) {
  std::mt19937_64 rng(3);
  std::set<std::uint64_t> keys;
  std::set<std::string> boards;
  for (int t = 0; t < 3000; ++t) {
    Position p;
    int len = static_cast<int>(rng() % 12);
    for (int k = 0; k < len && !p.is_terminal(); ++k) {
      int c = static_cast<int>(rng() % kWidth);
      if (p.can_play(c)) p.play(c);
    }
    keys.insert(p.key());
    boards.insert(p.to_string() + std::to_string(p.side_to_move()));
  }
  EXPECT_EQ(keys.size(), boards.size());
}

TEST(Position, NonLosingMovesBlocksThreat) {
  // X threatens to complete the bottom row at column 4 (index 3).
  Position p = Position::from_moves("1627");  // X: 1,2  O: 6,7
  p.play(2);                                   // X plays column 3 -> threat at 4
  // O to move, must block column index 3 (and 0 is also a threat? no: X has 1,2,3)
  const u64 nl = p.possible_non_losing_moves();
  EXPECT_NE(nl & column_mask(3), 0u);
  EXPECT_EQ(nl & ~column_mask(3), 0u);
}
