#include "se/varint.hpp"

namespace se {

bool vbyte_decode_checked(const uint8_t*& p, const uint8_t* end, uint32_t& out) {
  uint64_t v = 0;
  for (int i = 0; i < 5; ++i) {
    if (p >= end) return false;
    uint8_t b = *p++;
    v |= static_cast<uint64_t>(b & 0x7F) << (7 * i);
    if (b < 0x80) {
      if (v > 0xFFFFFFFFull) return false;
      out = static_cast<uint32_t>(v);
      return true;
    }
  }
  return false;
}

}  // namespace se
