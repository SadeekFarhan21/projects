// B+ tree behaviour through the DB API, checked against std::map.
#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <random>

#include "kvdb/db.h"
#include "test_util.h"

using namespace kvdb;
using test::key_of;
using test::value_of;

namespace {
Options fast_opts(size_t pool = 1024) {
  Options o;
  o.sync = SyncMode::kNone;
  o.pool_pages = pool;
  return o;
}

void expect_same(DB& db, const std::map<std::string, std::string>& ref) {
  for (auto& [k, v] : ref) {
    std::string got;
    ASSERT_TRUE(db.get(k, &got)) << k;
    ASSERT_EQ(got, v) << k;
  }
  // Full scan must return exactly the reference, in order.
  auto it = ref.begin();
  size_t n = 0;
  db.scan("", "", [&](std::string_view k, std::string_view v) {
    EXPECT_NE(it, ref.end());
    if (it == ref.end()) return false;
    EXPECT_EQ(k, it->first);
    EXPECT_EQ(v, it->second);
    ++it;
    ++n;
    return true;
  });
  EXPECT_EQ(n, ref.size());
  const auto st = db.verify();
  EXPECT_EQ(st.entries, ref.size());
}
}  // namespace

TEST(BTree, EmptyTree) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  std::string v;
  EXPECT_FALSE(db->get("nope", &v));
  EXPECT_FALSE(db->del("nope"));
  int n = 0;
  db->scan("", "", [&](auto, auto) { return ++n, true; });
  EXPECT_EQ(n, 0);
  EXPECT_EQ(db->height(), 1u);
}

TEST(BTree, SequentialInsertsSplitAndStayFull) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  std::map<std::string, std::string> ref;
  for (int i = 0; i < 20000; ++i) {
    ref[key_of(i)] = value_of(i);
    db->put(key_of(i), value_of(i));
  }
  EXPECT_GE(db->height(), 3u);
  expect_same(*db, ref);
  // Rightmost-append splits should keep leaves nearly full.
  EXPECT_GT(db->verify().leaf_fill, 0.9);
}

TEST(BTree, RandomOpsMatchStdMap) {
  test::TempDir dir;
  // Small pool: forces evictions and mid-run checkpoints.
  auto db = DB::open(dir.file("t.db"), fast_opts(256));
  std::map<std::string, std::string> ref;
  std::mt19937_64 rng(42);
  for (int step = 0; step < 60000; ++step) {
    const uint64_t i = rng() % 15000;
    const int op = static_cast<int>(rng() % 10);
    if (op < 6) {
      auto v = value_of(i, step);
      db->put(key_of(i), v);
      ref[key_of(i)] = v;
    } else if (op < 8) {
      EXPECT_EQ(db->del(key_of(i)), ref.erase(key_of(i)) == 1);
    } else {
      std::string got;
      auto it = ref.find(key_of(i));
      ASSERT_EQ(db->get(key_of(i), &got), it != ref.end());
      if (it != ref.end()) ASSERT_EQ(got, it->second);
    }
  }
  expect_same(*db, ref);
  EXPECT_GT(db->stats().checkpoints, 1u);
}

TEST(BTree, RangeScanBounds) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  std::vector<int> ids(5000);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), std::mt19937(1));
  for (int i : ids) db->put(key_of(i), value_of(i));
  std::vector<std::string> got;
  db->scan(key_of(1234), key_of(1300), [&](std::string_view k, std::string_view) {
    got.emplace_back(k);
    return true;
  });
  ASSERT_EQ(got.size(), 66u);
  EXPECT_EQ(got.front(), key_of(1234));
  EXPECT_EQ(got.back(), key_of(1299));
  // Early stop.
  int n = 0;
  db->scan(key_of(10), "", [&](auto, auto) { return ++n < 7; });
  EXPECT_EQ(n, 7);
  // Start key between existing keys.
  got.clear();
  db->scan(key_of(4999) + "0", "", [&](std::string_view k, std::string_view) {
    got.emplace_back(k);
    return true;
  });
  EXPECT_TRUE(got.empty());
}

TEST(BTree, DeleteEverythingThenReinsert) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  for (int i = 0; i < 8000; ++i) db->put(key_of(i), value_of(i));
  for (int i = 0; i < 8000; ++i) ASSERT_TRUE(db->del(key_of(i)));
  int n = 0;
  db->scan("", "", [&](auto, auto) { return ++n, true; });
  EXPECT_EQ(n, 0);  // lazy delete leaves empty leaves, scan skips them
  std::map<std::string, std::string> ref;
  for (int i = 7999; i >= 0; i -= 3) {
    db->put(key_of(i), value_of(i, 1));
    ref[key_of(i)] = value_of(i, 1);
  }
  expect_same(*db, ref);
}

TEST(BTree, MaxSizeKeysAndValuesSplitCorrectly) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  std::map<std::string, std::string> ref;
  std::mt19937 rng(3);
  for (int i = 0; i < 3000; ++i) {
    std::string k(1 + rng() % kMaxKeySize, static_cast<char>('a' + rng() % 26));
    k += std::to_string(i);
    if (k.size() > kMaxKeySize) k = k.substr(k.size() - kMaxKeySize);
    std::string v(rng() % (kMaxValueSize + 1), 'x');
    db->put(k, v);
    ref[k] = v;
  }
  expect_same(*db, ref);
  EXPECT_THROW(db->put(std::string(kMaxKeySize + 1, 'k'), "v"), std::invalid_argument);
  EXPECT_THROW(db->put("k", std::string(kMaxValueSize + 1, 'v')), std::invalid_argument);
  EXPECT_THROW(db->put("", "v"), std::invalid_argument);
}

TEST(BTree, GrowingAndShrinkingValuesForceSplitsOnUpdate) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  std::map<std::string, std::string> ref;
  for (int i = 0; i < 3000; ++i) {
    db->put(key_of(i), "s");
    ref[key_of(i)] = "s";
  }
  for (int i = 0; i < 3000; ++i) {  // every update grows the cell
    std::string v(200 + i % 300, 'L');
    db->put(key_of(i), v);
    ref[key_of(i)] = v;
  }
  expect_same(*db, ref);
}

TEST(BTree, PersistsAcrossCleanReopen) {
  test::TempDir dir;
  std::map<std::string, std::string> ref;
  {
    auto db = DB::open(dir.file("t.db"), fast_opts());
    for (int i = 0; i < 10000; ++i) {
      db->put(key_of(i * 3), value_of(i));
      ref[key_of(i * 3)] = value_of(i);
    }
    for (int i = 0; i < 10000; i += 5) {
      db->del(key_of(i * 3));
      ref.erase(key_of(i * 3));
    }
  }
  auto db = DB::open(dir.file("t.db"), fast_opts());
  EXPECT_EQ(db->stats().recovered_records, 0u);  // clean close checkpointed
  expect_same(*db, ref);
}

TEST(BTree, AtomicBatchApplies) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  db->put("a", "1");
  WriteBatch b;
  b.put("b", "2");
  b.del("a");
  b.put("c", "3");
  db->write(b);
  std::string v;
  EXPECT_FALSE(db->get("a", &v));
  EXPECT_TRUE(db->get("b", &v));
  EXPECT_EQ(v, "2");
  EXPECT_TRUE(db->get("c", &v));
}

TEST(BTree, SecondOpenOfSameFileIsRejected) {
  test::TempDir dir;
  auto db = DB::open(dir.file("t.db"), fast_opts());
  EXPECT_THROW(DB::open(dir.file("t.db"), fast_opts()), IoError);
}
