// Storage-layer tests: segments, the sparse index, recovery and retention.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <fstream>
#include <random>
#include <string>

#include "minikafka/bytes.h"
#include "minikafka/offset_store.h"
#include "minikafka/partition_log.h"
#include "minikafka/record.h"
#include "minikafka/segment.h"
#include "test_util.h"

using namespace mk;
namespace fs = std::filesystem;

namespace {

std::string value_for(uint64_t i) { return "value-" + std::to_string(i) + std::string(i % 50, 'z'); }

// Appends records [first, first+count) as a single batch.
uint64_t append_n(PartitionLog& log, uint64_t first, uint32_t count) {
  std::vector<uint8_t> batch;
  for (uint32_t i = 0; i < count; ++i)
    append_record(batch, static_cast<int64_t>(first + i), "k" + std::to_string(first + i),
                  value_for(first + i));
  return log.append(batch.data(), batch.size(), count);
}

// Reads the whole log from `from` and returns the decoded records.
std::vector<Record> read_all(const PartitionLog& log, uint64_t from, uint32_t max_bytes = 8192) {
  std::vector<Record> out;
  uint64_t pos = from;
  for (;;) {
    ReadResult r = log.read(pos, max_bytes);
    EXPECT_EQ(r.error, ErrorCode::None);
    if (r.data.empty()) break;
    for (auto& rec : decode_records(r.data.data(), r.data.size())) {
      pos = rec.offset + 1;
      out.push_back(std::move(rec));
    }
  }
  return out;
}

LogConfig small_segments() {
  LogConfig c;
  c.segment_bytes = 16 * 1024;
  c.index_interval_bytes = 512;
  return c;
}

}  // namespace

TEST(PartitionLog, AssignsContiguousOffsetsAndReadsBackInOrder) {
  test::TempDir dir;
  PartitionLog log(dir.path() / "t-0", small_segments());
  uint64_t next = 0;
  std::mt19937 rng(1);
  while (next < 5000) {
    uint32_t n = 1 + rng() % 40;
    EXPECT_EQ(append_n(log, next, n), next);
    next += n;
  }
  EXPECT_EQ(log.log_end(), next);
  EXPECT_GT(log.segment_count(), 10u) << "test should span many segments";

  auto recs = read_all(log, 0);
  ASSERT_EQ(recs.size(), next);
  for (uint64_t i = 0; i < next; ++i) {
    ASSERT_EQ(recs[i].offset, i);
    ASSERT_EQ(recs[i].value, value_for(i));
    ASSERT_EQ(recs[i].timestamp_ms, static_cast<int64_t>(i));
  }
}

TEST(PartitionLog, RandomOffsetLookupsUseSparseIndexCorrectly) {
  test::TempDir dir;
  PartitionLog log(dir.path() / "t-0", small_segments());
  for (uint64_t i = 0; i < 3000; i += 7) append_n(log, i, 7);
  std::mt19937 rng(7);
  for (int trial = 0; trial < 2000; ++trial) {
    uint64_t off = rng() % log.log_end();
    ReadResult r = log.read(off, 1);  // max_bytes=1 still returns one whole record
    ASSERT_EQ(r.error, ErrorCode::None);
    auto recs = decode_records(r.data.data(), r.data.size());
    ASSERT_EQ(recs.size(), 1u);
    ASSERT_EQ(recs[0].offset, off);
    ASSERT_EQ(recs[0].value, value_for(off));
  }
}

TEST(PartitionLog, ReadBoundaries) {
  test::TempDir dir;
  PartitionLog log(dir.path() / "t-0", small_segments());
  append_n(log, 0, 10);
  EXPECT_TRUE(log.read(10, 1024).data.empty());                       // caught up
  EXPECT_EQ(log.read(11, 1024).error, ErrorCode::OffsetOutOfRange);   // past the end
  // max_bytes cuts at record boundaries.
  ReadResult r = log.read(0, 100);
  auto recs = decode_records(r.data.data(), r.data.size());
  EXPECT_GE(recs.size(), 1u);
  EXPECT_LE(r.data.size(), 100u + 200u);
  EXPECT_EQ(validate_batch(r.data.data(), r.data.size(), nullptr), static_cast<int64_t>(recs.size()));
}

TEST(PartitionLog, RecoversAfterReopen) {
  test::TempDir dir;
  {
    PartitionLog log(dir.path() / "t-0", small_segments());
    for (uint64_t i = 0; i < 2000; i += 10) append_n(log, i, 10);
  }
  PartitionLog log(dir.path() / "t-0", small_segments());
  EXPECT_EQ(log.log_end(), 2000u);
  EXPECT_EQ(log.recovery_stats().bytes_truncated, 0u);
  auto recs = read_all(log, 0);
  ASSERT_EQ(recs.size(), 2000u);
  for (uint64_t i = 0; i < 2000; ++i) ASSERT_EQ(recs[i].offset, i);
  // And appending continues from the recovered end.
  EXPECT_EQ(append_n(log, 2000, 5), 2000u);
}

TEST(PartitionLog, TornTailIsTruncatedOnRecovery) {
  test::TempDir dir;
  fs::path pdir = dir.path() / "t-0";
  {
    PartitionLog log(pdir, LogConfig{});
    append_n(log, 0, 100);
  }
  fs::path seg = pdir / "00000000000000000000.log";
  const auto good_size = fs::file_size(seg);
  // Simulate a crash mid-write: half a record header plus garbage at the end.
  {
    std::vector<uint8_t> partial;
    append_record(partial, 0, "k", "a record that never finished writing");
    store_u64(partial.data(), 100);
    std::ofstream f(seg, std::ios::binary | std::ios::app);
    f.write(reinterpret_cast<const char*>(partial.data()), static_cast<std::streamsize>(partial.size() / 2));
  }
  ASSERT_GT(fs::file_size(seg), good_size);
  PartitionLog log(pdir, LogConfig{});
  EXPECT_EQ(log.log_end(), 100u);
  EXPECT_GT(log.recovery_stats().bytes_truncated, 0u);
  EXPECT_EQ(fs::file_size(seg), good_size);
  EXPECT_EQ(read_all(log, 0).size(), 100u);
  EXPECT_EQ(append_n(log, 100, 1), 100u);
}

TEST(PartitionLog, CorruptTailRecordIsTruncatedOnRecovery) {
  test::TempDir dir;
  fs::path pdir = dir.path() / "t-0";
  uint64_t size_before_last = 0;
  {
    PartitionLog log(pdir, LogConfig{});
    append_n(log, 0, 50);
    size_before_last = log.size_bytes();
    append_n(log, 50, 1);
  }
  // Flip one byte inside the last record's payload: its CRC no longer matches.
  fs::path seg = pdir / "00000000000000000000.log";
  int fd = ::open(seg.c_str(), O_RDWR);
  ASSERT_GE(fd, 0);
  uint8_t b;
  const off_t at = static_cast<off_t>(size_before_last + kRecordHeaderSize + 13);
  ASSERT_EQ(::pread(fd, &b, 1, at), 1);
  b ^= 0xFF;
  ASSERT_EQ(::pwrite(fd, &b, 1, at), 1);
  ::close(fd);

  PartitionLog log(pdir, LogConfig{});
  EXPECT_EQ(log.log_end(), 50u);
  EXPECT_EQ(fs::file_size(seg), size_before_last);
}

TEST(PartitionLog, RebuildsMissingIndexOfSealedSegment) {
  test::TempDir dir;
  fs::path pdir = dir.path() / "t-0";
  {
    PartitionLog log(pdir, small_segments());
    for (uint64_t i = 0; i < 1000; i += 10) append_n(log, i, 10);
  }
  // Corrupt the first (sealed) segment's index: out-of-order garbage entries.
  {
    std::ofstream f(pdir / "00000000000000000000.index", std::ios::binary | std::ios::trunc);
    uint32_t junk[4] = {50, 999999, 10, 5};
    f.write(reinterpret_cast<const char*>(junk), sizeof(junk));
  }
  PartitionLog log(pdir, small_segments());
  auto recs = read_all(log, 0);
  ASSERT_EQ(recs.size(), 1000u);
  for (uint64_t i = 0; i < 1000; ++i) ASSERT_EQ(recs[i].offset, i);
}

TEST(PartitionLog, SizeRetentionDeletesOldestSegments) {
  test::TempDir dir;
  LogConfig cfg = small_segments();
  cfg.retention_bytes = 64 * 1024;
  PartitionLog log(dir.path() / "t-0", cfg);
  for (uint64_t i = 0; i < 20000; i += 20) append_n(log, i, 20);

  // Size stays within [retention, retention + one segment].
  EXPECT_GE(log.size_bytes(), static_cast<uint64_t>(cfg.retention_bytes));
  EXPECT_LE(log.size_bytes(), cfg.retention_bytes + cfg.segment_bytes + 4096);
  EXPECT_GT(log.log_start(), 0u);
  EXPECT_EQ(log.log_end(), 20000u);

  // Reading below the log start is an error; reading from it works.
  EXPECT_EQ(log.read(0, 1024).error, ErrorCode::OffsetOutOfRange);
  auto recs = read_all(log, log.log_start());
  EXPECT_EQ(recs.size(), log.log_end() - log.log_start());
  EXPECT_EQ(recs.back().offset, 19999u);

  // Deleted segment files are really gone from disk.
  size_t files = 0;
  for (const auto& e : fs::directory_iterator(dir.path() / "t-0"))
    if (e.path().extension() == ".log") ++files;
  EXPECT_EQ(files, log.segment_count());

  // And retention survives a restart: log start is recomputed from the files.
  const uint64_t start = log.log_start();
  PartitionLog reopened(dir.path() / "t-0", cfg);
  EXPECT_EQ(reopened.log_start(), start);
  EXPECT_EQ(reopened.log_end(), 20000u);
}

TEST(OffsetStore, PersistsAcrossReopen) {
  test::TempDir dir;
  {
    OffsetStore s(dir.path(), LogConfig{});
    s.commit("g1", "orders", 0, 10);
    s.commit("g1", "orders", 1, 20);
    s.commit("g1", "orders", 0, 15);  // later commit wins
    s.commit("g2", "orders", 0, 3);
    EXPECT_EQ(s.fetch("g1", "orders", 0), 15u);
  }
  OffsetStore s(dir.path(), LogConfig{});
  EXPECT_EQ(s.replayed_records(), 4u);
  EXPECT_EQ(s.fetch("g1", "orders", 0), 15u);
  EXPECT_EQ(s.fetch("g1", "orders", 1), 20u);
  EXPECT_EQ(s.fetch("g2", "orders", 0), 3u);
  EXPECT_FALSE(s.fetch("g3", "orders", 0).has_value());
}

TEST(Segment, IndexHasEntriesInsideLargeBatches) {
  // Regression: the index used to get at most one entry per append, so a 1 MiB
  // batch left 1 MiB to scan for any fetch starting in the middle of it.
  test::TempDir dir;
  auto seg = Segment::create(dir.path(), 0, 4096);
  std::vector<uint8_t> batch;
  const std::string value(1000, 'v');
  for (int i = 0; i < 1000; ++i) append_record(batch, i, "", value);  // ~1 MiB, one append
  for (uint64_t i = 0; i < 1000; ++i) store_u64(batch.data() + i * (kRecordHeaderSize + 12 + 1000), i);
  seg->append(batch.data(), batch.size(), 0, 1000);
  EXPECT_GE(seg->index_entries(), batch.size() / 4096 - 1);
  // The floor entry for any offset is within one interval (plus one record) of it.
  const uint64_t rec = kRecordHeaderSize + 12 + 1000;
  for (uint64_t off : {1u, 500u, 999u}) {
    const uint64_t pos = seg->floor_position(off);
    EXPECT_LE(off * rec - pos, 4096 + rec) << off;
    auto got = decode_records(seg->read_from(pos, off, 1, seg->size()).data(), rec);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].offset, off);
  }
}
