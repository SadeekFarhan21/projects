#include "c4/position.hpp"

#include <stdexcept>

namespace c4 {

Position Position::from_moves(std::string_view seq) {
  Position p;
  for (char ch : seq) {
    const int col = ch - '1';
    if (col < 0 || col >= kWidth) throw std::invalid_argument("bad column in move string");
    if (p.is_terminal()) throw std::invalid_argument("move after game end");
    if (!p.can_play(col)) throw std::invalid_argument("move into full column");
    p.play(col);
  }
  return p;
}

Position Position::mirrored() const {
  Position m;
  for (int c = 0; c < kWidth; ++c) {
    const int src = c * kColStride;
    const int dst = (kWidth - 1 - c) * kColStride;
    const u64 colbits = column_mask(0);
    m.current_ |= ((current_ >> src) & colbits) << dst;
    m.mask_ |= ((mask_ >> src) & colbits) << dst;
  }
  m.moves_ = moves_;
  return m;
}

int Position::cell(int col, int row) const {
  const u64 bit = u64{1} << (col * kColStride + row);
  if (!(mask_ & bit)) return 0;
  const bool is_current = (current_ & bit) != 0;
  // current belongs to the side to move
  const int mover = side_to_move();  // 0 = first player to move
  const int owner_if_current = mover == 0 ? 1 : 2;
  const int owner_if_other = mover == 0 ? 2 : 1;
  return is_current ? owner_if_current : owner_if_other;
}

void Position::encode(float* out) const {
  const u64 opp = current_ ^ mask_;
  const float first = side_to_move() == 0 ? 1.0f : 0.0f;
  for (int r = 0; r < kHeight; ++r) {
    for (int c = 0; c < kWidth; ++c) {
      const u64 bit = u64{1} << (c * kColStride + r);
      const int idx = r * kWidth + c;
      out[idx] = (current_ & bit) ? 1.0f : 0.0f;
      out[kPlaneSize + idx] = (opp & bit) ? 1.0f : 0.0f;
      out[2 * kPlaneSize + idx] = first;
    }
  }
}

std::string Position::to_string() const {
  std::string s;
  for (int r = kHeight - 1; r >= 0; --r) {
    for (int c = 0; c < kWidth; ++c) {
      const int v = cell(c, r);
      s += v == 0 ? '.' : (v == 1 ? 'X' : 'O');
    }
    s += '\n';
  }
  return s;
}

}  // namespace c4
