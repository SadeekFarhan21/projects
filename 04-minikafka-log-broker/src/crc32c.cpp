#include "minikafka/crc32c.h"

#include <array>
#include <cstring>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace mk {
namespace {

constexpr uint32_t kPoly = 0x82F63B78u;  // reflected Castagnoli polynomial

constexpr std::array<uint32_t, 256> make_table() {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ kPoly : (c >> 1);
    t[i] = c;
  }
  return t;
}
constexpr auto kTable = make_table();

}  // namespace

uint32_t crc32c_sw(const void* data, size_t n, uint32_t seed) {
  auto* p = static_cast<const uint8_t*>(data);
  uint32_t c = ~seed;
  for (size_t i = 0; i < n; ++i) c = kTable[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return ~c;
}

uint32_t crc32c(const void* data, size_t n, uint32_t seed) {
#if defined(__ARM_FEATURE_CRC32)
  auto* p = static_cast<const uint8_t*>(data);
  uint32_t c = ~seed;
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    c = __crc32cd(c, v);
    p += 8;
    n -= 8;
  }
  while (n > 0) {
    c = __crc32cb(c, *p++);
    --n;
  }
  return ~c;
#else
  return crc32c_sw(data, n, seed);
#endif
}

}  // namespace mk
