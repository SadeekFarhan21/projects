// Public API: a persistent, single-threaded key-value store.
//
//   auto db = kvdb::DB::open("data.kv", {});
//   db->put("k", "v");
//   std::string v; db->get("k", &v);
//
// Files: <path> (pages), <path>-wal (redo log), <path>-journal (checkpoint
// double-write area, empty except during a checkpoint).
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "kvdb/btree.h"
#include "kvdb/buffer_pool.h"
#include "kvdb/pager.h"
#include "kvdb/wal.h"
#include "kvdb/write_batch.h"

namespace kvdb {

struct Options {
  size_t pool_pages = 16384;                  // 64 MiB of 4 KiB frames
  SyncMode sync = SyncMode::kFsync;           // per-commit WAL durability
  uint64_t wal_checkpoint_bytes = 64ull << 20;  // checkpoint when WAL exceeds
};

struct DbStats {
  uint64_t commits = 0;
  uint64_t checkpoints = 0;
  uint64_t checkpoint_pages = 0;
  // Cumulative wall time per checkpoint phase, microseconds.
  uint64_t ckpt_journal_us = 0;  // journal write + sync
  uint64_t ckpt_data_us = 0;     // in-place page writes + sync
  uint64_t ckpt_tail_us = 0;     // WAL + journal truncation
  uint64_t wal_bytes = 0;
  uint64_t wal_syncs = 0;
  uint64_t recovered_records = 0;  // WAL records redone at the last open
  uint64_t journal_pages_replayed = 0;  // from an interrupted checkpoint
  BufferPoolStats pool;
  uint64_t page_reads = 0;
  uint64_t page_writes = 0;
};

class DB {
 public:
  static std::unique_ptr<DB> open(const std::string& path, const Options& opts = {});
  ~DB();  // checkpoints; a crash instead of a clean close is also safe
  DB(const DB&) = delete;
  DB& operator=(const DB&) = delete;

  void put(std::string_view key, std::string_view value);
  bool del(std::string_view key);  // true if the key existed
  bool get(std::string_view key, std::string* value);
  // Atomic: after a crash either every op in the batch is visible or none.
  void write(const WriteBatch& batch);
  // Keys in [lo, hi); empty hi = unbounded. fn returns false to stop.
  void scan(std::string_view lo, std::string_view hi,
            const std::function<bool(std::string_view, std::string_view)>& fn);

  // Writes all dirty pages crash-atomically and empties the WAL.
  void checkpoint() { checkpoint_impl(true); }

  DbStats stats() const;
  TreeStats verify() { return tree_->verify(); }
  uint32_t height() { return tree_->height(); }

 private:
  DB(const std::string& path, const Options& opts);
  void recover();
  void recover_journal();
  void apply(std::string_view rep);
  void maybe_checkpoint(bool between_batches);
  void checkpoint_impl(bool truncate_wal);

  std::string path_;
  Options opts_;
  std::unique_ptr<Pager> pager_;
  std::unique_ptr<BufferPool> pool_;
  std::unique_ptr<BTree> tree_;
  std::unique_ptr<Wal> wal_;
  int journal_fd_ = -1;
  uint64_t next_lsn_ = 1;
  uint64_t applied_lsn_ = 0;  // every batch <= this is fully in the tree
  DbStats stats_;
  WriteBatch scratch_;  // reused by put/del
};

}  // namespace kvdb
