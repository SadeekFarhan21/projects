#include "kvdb/buffer_pool.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace kvdb {

BufferPool::BufferPool(Pager& pager, size_t capacity)
    : pager_(pager), capacity_(capacity), frames_(capacity) {
  if (capacity < 16) throw std::invalid_argument("buffer pool too small");
  // Page-aligned so frames could later be used with O_DIRECT-style I/O.
  void* mem = std::aligned_alloc(kPageSize, capacity * kPageSize);
  if (mem == nullptr) throw std::bad_alloc();
  buf_.reset(static_cast<char*>(mem));
  free_.reserve(capacity);
  for (size_t i = capacity; i-- > 0;) free_.push_back(i);
  page_table_.reserve(capacity * 2);
}

void BufferPool::lru_remove(size_t f) {
  if (frames_[f].in_lru) {
    lru_.erase(frames_[f].lru_pos);
    frames_[f].in_lru = false;
  }
}

void BufferPool::lru_push_mru(size_t f) {
  lru_.push_front(f);
  frames_[f].lru_pos = lru_.begin();
  frames_[f].in_lru = true;
}

size_t BufferPool::grab_frame() {
  if (!free_.empty()) {
    size_t f = free_.back();
    free_.pop_back();
    return f;
  }
  if (lru_.empty()) {
    throw BufferPoolFull("buffer pool full: all frames pinned or dirty");
  }
  size_t f = lru_.back();  // least recently used clean, unpinned frame
  lru_.pop_back();
  frames_[f].in_lru = false;
  page_table_.erase(frames_[f].page_id);
  ++stats_.evictions;
  return f;
}

char* BufferPool::fetch(PageId id) {
  auto it = page_table_.find(id);
  if (it != page_table_.end()) {
    size_t f = it->second;
    lru_remove(f);  // pinned frames are never in the LRU list
    ++frames_[f].pin_count;
    ++stats_.hits;
    return data(f);
  }
  ++stats_.misses;
  size_t f = grab_frame();
  try {
    pager_.read_page(id, data(f));
  } catch (...) {
    free_.push_back(f);
    throw;
  }
  frames_[f] = Frame{id, 1, false, false, {}};
  page_table_.emplace(id, f);
  return data(f);
}

char* BufferPool::create(PageId id) {
  if (page_table_.count(id) != 0) {
    throw std::logic_error("create() of a page that is already resident");
  }
  size_t f = grab_frame();
  std::memset(data(f), 0, kPageSize);
  frames_[f] = Frame{id, 1, true, false, {}};
  ++dirty_count_;
  page_table_.emplace(id, f);
  return data(f);
}

void BufferPool::unpin(PageId id, bool dirty) {
  auto it = page_table_.find(id);
  if (it == page_table_.end()) throw std::logic_error("unpin of absent page");
  Frame& fr = frames_[it->second];
  if (fr.pin_count <= 0) throw std::logic_error("unpin of unpinned page");
  if (dirty && !fr.dirty) {
    fr.dirty = true;
    ++dirty_count_;
  }
  if (--fr.pin_count == 0 && !fr.dirty) lru_push_mru(it->second);
}

std::vector<std::pair<PageId, const char*>> BufferPool::dirty_pages() const {
  std::vector<std::pair<PageId, const char*>> out;
  out.reserve(dirty_count_);
  for (const auto& [id, f] : page_table_) {
    if (frames_[f].dirty) out.emplace_back(id, buf_.get() + f * kPageSize);
  }
  std::sort(out.begin(), out.end());
  return out;
}

void BufferPool::mark_all_clean() {
  for (const auto& [id, f] : page_table_) {
    Frame& fr = frames_[f];
    if (!fr.dirty) continue;
    fr.dirty = false;
    if (fr.pin_count == 0) lru_push_mru(f);
  }
  dirty_count_ = 0;
}

}  // namespace kvdb
