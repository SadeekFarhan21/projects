#include <gtest/gtest.h>

#include <random>
#include <string>

#include "minikafka/bytes.h"
#include "minikafka/crc32c.h"
#include "minikafka/record.h"

using namespace mk;

TEST(Crc32c, KnownVectors) {
  // Standard CRC-32C check value.
  EXPECT_EQ(crc32c("123456789", 9), 0xE3069283u);
  EXPECT_EQ(crc32c("", 0), 0u);
  // 32 bytes of zeros, from RFC 3720 appendix B.4.
  uint8_t zeros[32] = {};
  EXPECT_EQ(crc32c(zeros, 32), 0x8A9136AAu);
}

TEST(Crc32c, HardwareMatchesSoftware) {
  std::mt19937 rng(42);
  for (size_t n : {0u, 1u, 7u, 8u, 9u, 63u, 64u, 1000u, 4097u}) {
    std::vector<uint8_t> buf(n);
    for (auto& b : buf) b = static_cast<uint8_t>(rng());
    EXPECT_EQ(crc32c(buf.data(), n), crc32c_sw(buf.data(), n)) << "n=" << n;
  }
}

TEST(Record, RoundTrip) {
  std::vector<uint8_t> batch;
  append_record(batch, 123, "k1", "hello");
  append_record(batch, 456, "", "");
  append_record(batch, 789, "key", std::string(10000, 'x'));
  std::string err;
  EXPECT_EQ(validate_batch(batch.data(), batch.size(), &err), 3) << err;
  auto recs = decode_records(batch.data(), batch.size());
  ASSERT_EQ(recs.size(), 3u);
  EXPECT_EQ(recs[0].timestamp_ms, 123);
  EXPECT_EQ(recs[0].key, "k1");
  EXPECT_EQ(recs[0].value, "hello");
  EXPECT_EQ(recs[1].key, "");
  EXPECT_EQ(recs[1].value, "");
  EXPECT_EQ(recs[2].value.size(), 10000u);
}

TEST(Record, ValidateRejectsCorruption) {
  std::vector<uint8_t> batch;
  append_record(batch, 1, "k", "value");
  append_record(batch, 2, "k", "value2");
  std::string err;

  auto flipped = batch;
  flipped[kRecordHeaderSize + 14] ^= 0x01;  // a byte inside the first value
  EXPECT_EQ(validate_batch(flipped.data(), flipped.size(), &err), -1);
  EXPECT_EQ(err, "crc mismatch");

  auto truncated = batch;
  truncated.resize(truncated.size() - 3);
  EXPECT_EQ(validate_batch(truncated.data(), truncated.size(), &err), -1);

  auto badlen = batch;
  store_u32(badlen.data() + 8, 0xFFFFFFFFu);
  EXPECT_EQ(validate_batch(badlen.data(), badlen.size(), &err), -1);
}

TEST(Bytes, ReaderThrowsOnTruncation) {
  Writer w;
  w.str("hello");
  w.u64(7);
  Reader r(w.buf.data(), w.buf.size() - 1);
  EXPECT_EQ(r.str(), "hello");
  EXPECT_THROW(r.u64(), DecodeError);

  // A string length prefix that claims more bytes than exist.
  Writer bad;
  bad.u32(1000);
  Reader r2(bad.buf.data(), bad.buf.size());
  EXPECT_THROW(r2.str(), DecodeError);
}
