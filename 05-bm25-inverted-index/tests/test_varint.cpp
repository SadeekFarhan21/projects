#include <gtest/gtest.h>

#include <random>

#include "se/varint.hpp"

using namespace se;

TEST(Varint, EncodedLengthsAtBoundaries) {
  struct Case { uint32_t v; size_t len; };
  for (auto c : {Case{0, 1}, Case{127, 1}, Case{128, 2}, Case{16383, 2}, Case{16384, 3},
                 Case{(1u << 21) - 1, 3}, Case{1u << 21, 4}, Case{(1u << 28) - 1, 4},
                 Case{1u << 28, 5}, Case{0xFFFFFFFFu, 5}}) {
    std::vector<uint8_t> buf;
    vbyte_encode(c.v, buf);
    EXPECT_EQ(buf.size(), c.len) << c.v;
    const uint8_t* p = buf.data();
    EXPECT_EQ(vbyte_decode(p), c.v);
    EXPECT_EQ(p, buf.data() + buf.size());
  }
}

TEST(Varint, RandomRoundTripAndSkip) {
  std::mt19937 rng(42);
  std::vector<uint32_t> vals;
  std::vector<uint8_t> buf;
  for (int i = 0; i < 10000; ++i) {
    // mix of magnitudes so every length is exercised
    uint32_t v = rng() >> (rng() % 32);
    vals.push_back(v);
    vbyte_encode(v, buf);
  }
  const uint8_t* p = buf.data();
  for (uint32_t v : vals) ASSERT_EQ(vbyte_decode(p), v);
  EXPECT_EQ(p, buf.data() + buf.size());

  const uint8_t* q = buf.data();
  vbyte_skip(q, 5000);
  EXPECT_EQ(vbyte_decode(q), vals[5000]);

  const uint8_t* c = buf.data();
  for (uint32_t v : vals) {
    uint32_t out;
    ASSERT_TRUE(vbyte_decode_checked(c, buf.data() + buf.size(), out));
    ASSERT_EQ(out, v);
  }
}

TEST(Varint, CheckedDecodeRejectsBadInput) {
  uint32_t out;
  std::vector<uint8_t> truncated = {0x80, 0x80};
  const uint8_t* p = truncated.data();
  EXPECT_FALSE(vbyte_decode_checked(p, truncated.data() + truncated.size(), out));

  std::vector<uint8_t> overlong = {0x80, 0x80, 0x80, 0x80, 0x80, 0x01};
  p = overlong.data();
  EXPECT_FALSE(vbyte_decode_checked(p, overlong.data() + overlong.size(), out));

  std::vector<uint8_t> overflow = {0xFF, 0xFF, 0xFF, 0xFF, 0x7F};  // 35 bits
  p = overflow.data();
  EXPECT_FALSE(vbyte_decode_checked(p, overflow.data() + overflow.size(), out));
}
