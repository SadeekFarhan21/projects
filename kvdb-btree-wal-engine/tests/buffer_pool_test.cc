#include <gtest/gtest.h>

#include "kvdb/buffer_pool.h"
#include "test_util.h"

using namespace kvdb;

namespace {
void write_marker(Pager& pager, PageId id) {
  alignas(16) char buf[kPageSize] = {};
  store<uint32_t>(buf, id * 7 + 1);
  pager.write_page(id, buf);
}
}  // namespace

TEST(BufferPool, HitsMissesAndLruEviction) {
  test::TempDir dir;
  Pager pager(dir.file("bp.db"));
  for (PageId i = 0; i < 40; ++i) write_marker(pager, i);
  BufferPool pool(pager, 16);
  for (PageId i = 0; i < 16; ++i) {
    EXPECT_EQ(load<uint32_t>(pool.fetch(i)), i * 7 + 1);
    pool.unpin(i, false);
  }
  EXPECT_EQ(pool.stats().misses, 16u);
  // Touch page 0 so page 1 becomes the LRU victim.
  pool.fetch(0);
  pool.unpin(0, false);
  EXPECT_EQ(pool.stats().hits, 1u);
  pool.fetch(20);
  pool.unpin(20, false);
  EXPECT_EQ(pool.stats().evictions, 1u);
  pool.fetch(0);  // still resident
  pool.unpin(0, false);
  EXPECT_EQ(pool.stats().hits, 2u);
  pool.fetch(1);  // was evicted
  pool.unpin(1, false);
  EXPECT_EQ(pool.stats().misses, 18u);
}

TEST(BufferPool, PinnedAndDirtyPagesAreNeverEvicted) {
  test::TempDir dir;
  Pager pager(dir.file("bp.db"));
  for (PageId i = 0; i < 40; ++i) write_marker(pager, i);
  BufferPool pool(pager, 16);
  for (PageId i = 0; i < 8; ++i) pool.fetch(i);  // pinned
  for (PageId i = 8; i < 16; ++i) {
    pool.fetch(i);
    pool.unpin(i, true);  // dirty
  }
  EXPECT_EQ(pool.dirty_count(), 8u);
  EXPECT_THROW(pool.fetch(30), BufferPoolFull);
  // After a checkpoint marks them clean, eviction can proceed.
  pool.mark_all_clean();
  EXPECT_EQ(pool.dirty_count(), 0u);
  EXPECT_NO_THROW(pool.fetch(30));
  pool.unpin(30, false);
  for (PageId i = 0; i < 8; ++i) pool.unpin(i, false);
}

TEST(BufferPool, DirtyPagesSortedAndCreateZeroes) {
  test::TempDir dir;
  Pager pager(dir.file("bp.db"));
  BufferPool pool(pager, 16);
  for (PageId id : {9u, 3u, 5u}) {
    char* p = pool.create(id);
    EXPECT_EQ(p[100], 0);
    p[0] = static_cast<char>(id);
    pool.unpin(id, true);
  }
  auto d = pool.dirty_pages();
  ASSERT_EQ(d.size(), 3u);
  EXPECT_EQ(d[0].first, 3u);
  EXPECT_EQ(d[2].first, 9u);
  EXPECT_EQ(d[1].second[0], 5);
}

TEST(BufferPool, UnpinErrorsAreDetected) {
  test::TempDir dir;
  Pager pager(dir.file("bp.db"));
  BufferPool pool(pager, 16);
  EXPECT_THROW(pool.unpin(3, false), std::logic_error);
  pool.fetch(3);
  pool.unpin(3, false);
  EXPECT_THROW(pool.unpin(3, false), std::logic_error);
}
