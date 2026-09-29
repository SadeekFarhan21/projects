// Buffer pool: a fixed set of in-memory frames caching pages, with pin counts
// and LRU eviction.
//
// Policy is NO-STEAL: a dirty page is never written back by eviction. Only
// clean, unpinned frames are eviction candidates. Dirty pages leave memory
// through DB::checkpoint(), which writes them crash-atomically via a journal.
// This keeps the on-disk tree equal to the last checkpoint at all times, which
// is what lets the WAL be purely logical (see DESIGN.md).
#pragma once

#include <cstddef>
#include <cstdlib>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kvdb/common.h"
#include "kvdb/pager.h"

namespace kvdb {

class BufferPoolFull : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct BufferPoolStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
};

class BufferPool {
 public:
  BufferPool(Pager& pager, size_t capacity);
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;

  // Pins the page, reading it from disk on a miss. Throws BufferPoolFull when
  // every frame is pinned or dirty.
  char* fetch(PageId id);
  // Pins a frame for a brand-new page without reading disk; contents zeroed
  // and the frame is marked dirty.
  char* create(PageId id);
  // Drops one pin. dirty=true marks the page modified.
  void unpin(PageId id, bool dirty);

  size_t capacity() const { return capacity_; }
  size_t dirty_count() const { return dirty_count_; }
  size_t resident() const { return page_table_.size(); }
  const BufferPoolStats& stats() const { return stats_; }

  // Dirty pages sorted by page id, with pointers to their current contents.
  std::vector<std::pair<PageId, const char*>> dirty_pages() const;
  // Called by checkpoint once dirty pages are durable: they become clean and
  // unpinned ones become evictable again.
  void mark_all_clean();

 private:
  struct Frame {
    PageId page_id = 0;
    int pin_count = 0;
    bool dirty = false;
    bool in_lru = false;
    std::list<size_t>::iterator lru_pos;
  };

  size_t grab_frame();  // free frame or evict LRU victim
  void lru_remove(size_t f);
  void lru_push_mru(size_t f);
  char* data(size_t f) { return buf_.get() + f * kPageSize; }

  Pager& pager_;
  size_t capacity_;
  struct FreeDeleter {
    void operator()(char* p) const { std::free(p); }
  };
  std::unique_ptr<char, FreeDeleter> buf_;  // from aligned_alloc, so free()
  std::vector<Frame> frames_;
  std::vector<size_t> free_;
  std::unordered_map<PageId, size_t> page_table_;
  std::list<size_t> lru_;  // front = most recently used; only clean+unpinned
  size_t dirty_count_ = 0;
  BufferPoolStats stats_;
};

// RAII pin. Unpins on destruction, passing along whether it was modified.
class PageGuard {
 public:
  PageGuard() = default;
  PageGuard(BufferPool* pool, PageId id, char* data)
      : pool_(pool), id_(id), data_(data) {}
  PageGuard(PageGuard&& o) noexcept { *this = std::move(o); }
  PageGuard& operator=(PageGuard&& o) noexcept {
    if (this != &o) {
      release();
      pool_ = o.pool_;
      id_ = o.id_;
      data_ = o.data_;
      dirty_ = o.dirty_;
      o.pool_ = nullptr;
    }
    return *this;
  }
  ~PageGuard() { release(); }

  char* data() { return data_; }
  const char* data() const { return data_; }
  PageId id() const { return id_; }
  void mark_dirty() { dirty_ = true; }
  void release() {
    if (pool_ != nullptr) pool_->unpin(id_, dirty_);
    pool_ = nullptr;
  }

 private:
  BufferPool* pool_ = nullptr;
  PageId id_ = 0;
  char* data_ = nullptr;
  bool dirty_ = false;
};

}  // namespace kvdb
