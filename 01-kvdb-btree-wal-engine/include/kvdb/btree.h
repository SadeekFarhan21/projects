// B+ tree over buffer-pool pages. Single-threaded. Deletes are lazy: they
// remove the cell but never merge or rebalance nodes.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kvdb/buffer_pool.h"
#include "kvdb/node.h"

namespace kvdb {

// Meta page (page 0) layout. Lives in the buffer pool like any other page, so
// root/page-count changes are checkpointed atomically with the tree itself.
struct Meta {
  static constexpr uint64_t kMagic = 0x3130424456564B21ull;  // "!KVVDB01"
  static constexpr uint32_t kVersion = 1;
  static constexpr size_t kMagicOff = 0, kVersionOff = 8, kRootOff = 12,
                          kPageCountOff = 16, kLsnOff = 24, kHeightOff = 32;
};

struct TreeStats {
  uint32_t height = 0;
  uint64_t leaf_pages = 0;
  uint64_t internal_pages = 0;
  uint64_t entries = 0;
  double leaf_fill = 0;  // mean fraction of usable leaf bytes in use
};

class BTree {
 public:
  explicit BTree(BufferPool& pool) : pool_(pool) {}

  // Writes a fresh meta page and an empty root leaf through the pool.
  static void format(BufferPool& pool);

  bool get(std::string_view key, std::string* out);
  void put(std::string_view key, std::string_view value);
  bool del(std::string_view key);
  // Visits keys in [lo, hi) in order; empty hi means unbounded. fn returns
  // false to stop early.
  void scan(std::string_view lo, std::string_view hi,
            const std::function<bool(std::string_view, std::string_view)>& fn);

  uint64_t checkpoint_lsn();
  void set_checkpoint_lsn(uint64_t lsn);
  uint32_t height();

  // Full walk: verifies node invariants, key ordering across nodes and the
  // leaf chain. Returns stats; throws std::logic_error on corruption.
  TreeStats verify();

 private:
  PageGuard pin(PageId id) { return PageGuard(&pool_, id, pool_.fetch(id)); }
  PageId root();
  PageId allocate(PageGuard* out);
  void insert_into_parent(std::vector<PageId>& path, PageId left,
                          std::string sep, PageId right);
  void split_leaf(PageGuard& leaf, std::vector<node::Entry>& ents, int idx,
                  std::vector<PageId>& path);

  BufferPool& pool_;
};

}  // namespace kvdb
