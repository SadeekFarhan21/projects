#include <gtest/gtest.h>

#include <map>
#include <random>

#include "kvdb/node.h"

using namespace kvdb;

namespace {
alignas(16) char page[kPageSize];
}

TEST(Node, InsertKeepsSortedAndLookupWorks) {
  node::init(page, node::Type::kLeaf, kNoPage);
  for (std::string k : {"m", "c", "x", "a", "q"}) {
    int idx = node::lower_bound(page, k);
    ASSERT_TRUE(node::insert(page, idx, k, "val_" + k));
  }
  ASSERT_EQ(node::count(page), 5);
  std::string why;
  ASSERT_TRUE(node::check(page, &why)) << why;
  EXPECT_EQ(node::key(page, 0), "a");
  EXPECT_EQ(node::key(page, 4), "x");
  EXPECT_EQ(node::value(page, node::lower_bound(page, "q")), "val_q");
  EXPECT_EQ(node::lower_bound(page, "zzz"), 5);
}

TEST(Node, FillsUntilFullThenRefuses) {
  node::init(page, node::Type::kLeaf, kNoPage);
  const std::string v(100, 'v');
  int n = 0;
  while (node::insert(page, n, "k" + std::to_string(1000 + n), v)) ++n;
  // cell = 4 + 5 + 100 = 109, plus 2-byte slot: floor(4080 / 111) = 36.
  EXPECT_EQ(n, 36);
  EXPECT_LE(node::used_bytes(page), node::kUsable);
}

TEST(Node, EraseAndCompactionReclaimSpace) {
  node::init(page, node::Type::kLeaf, kNoPage);
  const std::string v(100, 'v');
  int n = 0;
  while (node::insert(page, n, "k" + std::to_string(1000 + n), v)) ++n;
  // Erase every other cell: space is fragmented, not contiguous.
  for (int i = n - 2; i >= 0; i -= 2) node::erase(page, i);
  const int left = node::count(page);
  // Inserts must succeed by compacting.
  for (int i = 0; i < n - left; ++i) {
    std::string k = "k" + std::to_string(5000 + i);
    ASSERT_TRUE(node::insert(page, node::lower_bound(page, k), k, v)) << i;
  }
  std::string why;
  ASSERT_TRUE(node::check(page, &why)) << why;
}

TEST(Node, UpdateInPlaceAndResize) {
  node::init(page, node::Type::kLeaf, kNoPage);
  ASSERT_TRUE(node::insert(page, 0, "a", "111"));
  ASSERT_TRUE(node::insert(page, 1, "b", "222"));
  ASSERT_TRUE(node::update(page, 0, "999"));
  EXPECT_EQ(node::value(page, 0), "999");
  ASSERT_TRUE(node::update(page, 1, std::string(300, 'z')));
  EXPECT_EQ(node::value(page, 1), std::string(300, 'z'));
  EXPECT_EQ(node::key(page, 1), "b");
  EXPECT_EQ(node::value(page, 0), "999");
}

TEST(Node, RandomOpsMatchMap) {
  std::mt19937 rng(7);
  node::init(page, node::Type::kLeaf, kNoPage);
  std::map<std::string, std::string> ref;
  for (int step = 0; step < 20000; ++step) {
    std::string k = "k" + std::to_string(rng() % 200);
    std::string v(rng() % 40, static_cast<char>('a' + rng() % 26));
    int idx = node::lower_bound(page, k);
    bool exists = idx < node::count(page) && node::key(page, idx) == k;
    if (rng() % 3 == 0) {
      if (exists) {
        node::erase(page, idx);
        ref.erase(k);
      }
    } else if (exists) {
      if (node::update(page, idx, v)) ref[k] = v;
    } else if (node::insert(page, idx, k, v)) {
      ref[k] = v;
    }
  }
  ASSERT_EQ(node::count(page), ref.size());
  int i = 0;
  for (auto& [k, v] : ref) {
    EXPECT_EQ(node::key(page, i), k);
    EXPECT_EQ(node::value(page, i), v);
    ++i;
  }
}

TEST(Node, InternalFindChild) {
  node::init(page, node::Type::kInternal, 10);  // leftmost child = 10
  auto enc = [](uint32_t id) { return std::string(reinterpret_cast<char*>(&id), 4); };
  node::insert(page, 0, "d", enc(11));
  node::insert(page, 1, "h", enc(12));
  EXPECT_EQ(node::find_child(page, "a"), 10u);
  EXPECT_EQ(node::find_child(page, "d"), 11u);  // separator goes right
  EXPECT_EQ(node::find_child(page, "e"), 11u);
  EXPECT_EQ(node::find_child(page, "h"), 12u);
  EXPECT_EQ(node::find_child(page, "z"), 12u);
}
