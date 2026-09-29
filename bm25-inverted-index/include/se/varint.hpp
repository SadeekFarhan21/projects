#pragma once
// Variable-byte (LEB128-style) integer coding for postings.
// Each byte carries 7 payload bits, least significant group first.
// The high bit is set on every byte except the last one of a value.
#include <cstdint>
#include <vector>

namespace se {

inline void vbyte_encode(uint32_t v, std::vector<uint8_t>& out) {
  while (v >= 0x80) {
    out.push_back(static_cast<uint8_t>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<uint8_t>(v));
}

// Unchecked decode for the hot path. The index validates every postings list
// at open time (see Index::validate), so readers never run off the end.
inline uint32_t vbyte_decode(const uint8_t*& p) {
  uint32_t b = *p++;
  if (b < 0x80) return b;  // fast path: most gaps and tfs fit in one byte
  uint32_t v = b & 0x7F;
  int shift = 7;
  for (;;) {
    b = *p++;
    v |= (b & 0x7F) << shift;
    if (b < 0x80) return v;
    shift += 7;
  }
}

// Skip n encoded values without materialising them.
inline void vbyte_skip(const uint8_t*& p, uint32_t n) {
  while (n) {
    if (*p++ < 0x80) --n;
  }
}

// Bounds-checked decode used for validation. Rejects encodings longer than
// 5 bytes or values that overflow 32 bits.
bool vbyte_decode_checked(const uint8_t*& p, const uint8_t* end, uint32_t& out);

}  // namespace se
