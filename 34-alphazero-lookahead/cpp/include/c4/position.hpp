// Connect Four position on two 64-bit bitboards.
//
// Layout: column c occupies bits [c*7, c*7+6). Bit c*7+6 is a sentinel row that
// is always zero, which stops shifts from leaking between columns. Row 0 is the
// bottom row. `current` holds the stones of the player to move, `mask` holds
// every stone. The opponent's stones are `current ^ mask`.
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <string_view>

namespace c4 {

using u64 = std::uint64_t;

inline constexpr int kWidth = 7;
inline constexpr int kHeight = 6;
inline constexpr int kCells = kWidth * kHeight;
inline constexpr int kColStride = kHeight + 1;

// Center first. Used for move ordering in the solver and for tie breaks.
inline constexpr std::array<int, kWidth> kColumnOrder = {3, 2, 4, 1, 5, 0, 6};

namespace detail {
constexpr u64 bottom_mask() {
  u64 m = 0;
  for (int c = 0; c < kWidth; ++c) m |= u64{1} << (c * kColStride);
  return m;
}
}  // namespace detail

inline constexpr u64 kBottomMask = detail::bottom_mask();
inline constexpr u64 kBoardMask = kBottomMask * ((u64{1} << kHeight) - 1);

constexpr u64 top_mask_col(int c) { return u64{1} << (kHeight - 1 + c * kColStride); }
constexpr u64 bottom_mask_col(int c) { return u64{1} << (c * kColStride); }
constexpr u64 column_mask(int c) { return ((u64{1} << kHeight) - 1) << (c * kColStride); }

// True when `pos` contains four in a row in any direction.
constexpr bool has_alignment(u64 pos) {
  // horizontal
  u64 m = pos & (pos >> kColStride);
  if (m & (m >> (2 * kColStride))) return true;
  // diagonal /
  m = pos & (pos >> (kHeight + 2));
  if (m & (m >> (2 * (kHeight + 2)))) return true;
  // diagonal backslash
  m = pos & (pos >> kHeight);
  if (m & (m >> (2 * kHeight))) return true;
  // vertical
  m = pos & (pos >> 1);
  if (m & (m >> 2)) return true;
  return false;
}

// Empty cells that would complete four in a row for the stones in `position`.
constexpr u64 compute_winning_position(u64 position, u64 mask) {
  constexpr int H = kHeight;
  // vertical: three stacked stones below an empty cell
  u64 r = (position << 1) & (position << 2) & (position << 3);

  // horizontal (shift H+1) and both diagonals (shift H and H+2)
  auto line = [&](int s) {
    u64 p = (position << s) & (position << 2 * s);
    u64 out = p & (position << 3 * s);
    out |= p & (position >> s);
    p = (position >> s) & (position >> 2 * s);
    out |= p & (position << s);
    out |= p & (position >> 3 * s);
    return out;
  };
  r |= line(H + 1);
  r |= line(H);
  r |= line(H + 2);
  return r & (kBoardMask ^ mask);
}

class Position {
 public:
  Position() = default;

  // Build from a move string of 1-based column digits, e.g. "4453". Throws on
  // an illegal sequence or a sequence that plays past a finished game.
  static Position from_moves(std::string_view seq);

  bool can_play(int col) const { return (mask_ & top_mask_col(col)) == 0; }

  // Plays a column. Caller must check can_play.
  void play(int col) { play_bits((mask_ + bottom_mask_col(col)) & column_mask(col)); }

  // Plays a single-bit move taken from possible().
  void play_bits(u64 move) {
    current_ ^= mask_;
    mask_ |= move;
    ++moves_;
  }

  // Bits of every legal move (one per non-full column).
  u64 possible() const { return (mask_ + kBottomMask) & kBoardMask; }

  bool is_winning_move(int col) const {
    return (winning_position() & possible() & column_mask(col)) != 0;
  }
  bool can_win_next() const { return (winning_position() & possible()) != 0; }

  // Moves that do not hand the opponent an immediate win. Returns 0 when every
  // move loses (the opponent has two immediate threats, or a threat sits under
  // a threat).
  u64 possible_non_losing_moves() const {
    u64 poss = possible();
    const u64 opp_win = opponent_winning_position();
    const u64 forced = poss & opp_win;
    if (forced) {
      if (forced & (forced - 1)) return 0;
      poss = forced;
    }
    return poss & ~(opp_win >> 1);
  }

  // Number of winning cells the current player would own after `move`.
  int move_score(u64 move) const {
    return std::popcount(compute_winning_position(current_ | move, mask_));
  }

  // The player who made the last move has four in a row.
  bool last_mover_won() const { return has_alignment(current_ ^ mask_); }
  bool is_full() const { return moves_ == kCells; }
  bool is_terminal() const { return last_mover_won() || is_full(); }

  // Unique key: current + mask sets the bit above every column's top stone
  // and encodes whose stones are whose.
  u64 key() const { return current_ + mask_; }

  // Mirror the board left to right.
  Position mirrored() const;

  u64 current() const { return current_; }
  u64 mask() const { return mask_; }
  int moves() const { return moves_; }
  // 0 if the first player is to move, 1 otherwise.
  int side_to_move() const { return moves_ & 1; }

  // Cell accessor: 0 empty, 1 first player, 2 second player.
  int cell(int col, int row) const;

  // Writes 3x6x7 float planes: [current player stones, opponent stones,
  // ones if first player to move]. Index [plane][row][col], row 0 = bottom.
  void encode(float* out) const;

  std::string to_string() const;

 private:
  u64 winning_position() const { return compute_winning_position(current_, mask_); }
  u64 opponent_winning_position() const {
    return compute_winning_position(current_ ^ mask_, mask_);
  }

  u64 current_ = 0;
  u64 mask_ = 0;
  int moves_ = 0;
};

inline constexpr int kPlaneSize = kHeight * kWidth;
inline constexpr int kObsSize = 3 * kPlaneSize;

}  // namespace c4
