#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <vector>

#include "kvdb/wal.h"
#include "kvdb/write_batch.h"
#include "test_util.h"

using namespace kvdb;

namespace {
std::vector<std::pair<uint64_t, std::string>> read_all(Wal& w) {
  std::vector<std::pair<uint64_t, std::string>> out;
  w.replay([&](uint64_t lsn, std::string_view p) { out.emplace_back(lsn, std::string(p)); });
  return out;
}
off_t file_size(const std::string& p) {
  int fd = ::open(p.c_str(), O_RDONLY);
  off_t s = ::lseek(fd, 0, SEEK_END);
  ::close(fd);
  return s;
}
}  // namespace

TEST(Wal, RoundTrip) {
  test::TempDir dir;
  const auto path = dir.file("w.log");
  {
    Wal w(path);
    for (uint64_t i = 1; i <= 100; ++i) w.append(i, "payload" + std::to_string(i), SyncMode::kNone);
  }
  Wal w(path);
  auto recs = read_all(w);
  ASSERT_EQ(recs.size(), 100u);
  EXPECT_EQ(recs[41].first, 42u);
  EXPECT_EQ(recs[41].second, "payload42");
}

TEST(Wal, TornTailIsTruncatedAtEveryCutPoint) {
  test::TempDir dir;
  const auto path = dir.file("w.log");
  off_t full = 0;
  {
    Wal w(path);
    for (uint64_t i = 1; i <= 3; ++i) w.append(i, std::string(50, 'a' + i), SyncMode::kNone);
    full = static_cast<off_t>(w.size());
    EXPECT_EQ(file_size(path), static_cast<off_t>(Wal::kPreallocBytes));
  }
  ASSERT_EQ(::truncate(path.c_str(), full), 0);  // drop the zero tail
  const off_t rec = full / 3;
  // Simulate a crash that cut the file at every possible byte offset.
  for (off_t cut = 0; cut <= full; ++cut) {
    const auto copy = dir.file("cut.log");
    std::filesystem::copy_file(path, copy, std::filesystem::copy_options::overwrite_existing);
    ASSERT_EQ(::truncate(copy.c_str(), cut), 0);
    Wal w(copy);
    auto recs = read_all(w);
    ASSERT_EQ(recs.size(), static_cast<size_t>(cut / rec)) << "cut=" << cut;
    EXPECT_EQ(w.size(), static_cast<uint64_t>((cut / rec) * rec));
    // Appending after recovery must produce a readable log.
    w.append(10, "after", SyncMode::kNone);
    auto again = read_all(w);
    ASSERT_EQ(again.size(), recs.size() + 1);
    EXPECT_EQ(again.back().second, "after");
  }
}

TEST(Wal, PreallocatedZeroTailIsIgnoredAndTrimmed) {
  test::TempDir dir;
  const auto path = dir.file("w.log");
  uint64_t logical = 0;
  {
    Wal w(path);
    for (uint64_t i = 1; i <= 5; ++i) w.append(i, "rec", SyncMode::kNone);
    logical = w.size();
  }
  EXPECT_EQ(file_size(path), static_cast<off_t>(Wal::kPreallocBytes));
  Wal w(path);
  EXPECT_EQ(read_all(w).size(), 5u);
  EXPECT_EQ(file_size(path), static_cast<off_t>(logical));
}

TEST(Wal, CorruptByteStopsReplayThere) {
  test::TempDir dir;
  const auto path = dir.file("w.log");
  off_t rec = 0;
  {
    Wal w(path);
    for (uint64_t i = 1; i <= 10; ++i) w.append(i, std::string(30, 'x'), SyncMode::kNone);
    rec = static_cast<off_t>(w.size() / 10);
  }
  int fd = ::open(path.c_str(), O_RDWR);
  char c = 'Y';
  ASSERT_EQ(::pwrite(fd, &c, 1, rec * 6 + 20), 1);  // inside record 7's payload
  ::close(fd);
  Wal w(path);
  EXPECT_EQ(read_all(w).size(), 6u);
}

TEST(WriteBatch, EncodeDecode) {
  WriteBatch b;
  b.put("k1", "v1");
  b.del("k2");
  b.put("k3", "");
  std::vector<std::string> seen;
  ASSERT_TRUE(WriteBatch::iterate(b.rep(), [&](WriteBatch::Kind kind, std::string_view k, std::string_view v) {
    seen.push_back(std::to_string(kind) + ":" + std::string(k) + "=" + std::string(v));
  }));
  ASSERT_EQ(seen, (std::vector<std::string>{"1:k1=v1", "2:k2=", "1:k3="}));
  std::string bad(b.rep());
  bad.pop_back();
  bad.pop_back();
  bad.pop_back();
  EXPECT_FALSE(WriteBatch::iterate(bad, [](auto, auto, auto) {}));
}

TEST(Crc32, KnownVectorAndHardwareMatchesTable) {
  EXPECT_EQ(crc32("123456789", 9), 0xCBF43926u);  // standard check value
  std::string buf(10007, '\0');
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<char>(i * 131 + 7);
  for (size_t len : {0, 1, 7, 8, 9, 63, 4096, 10007}) {
    EXPECT_EQ(crc32(buf.data(), len), crc32_portable(buf.data(), len)) << len;
  }
  // Chaining.
  EXPECT_EQ(crc32(buf.data() + 100, 900, crc32(buf.data(), 100)), crc32(buf.data(), 1000));
}
