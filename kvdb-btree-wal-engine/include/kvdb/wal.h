// Write-ahead log. Logical redo records, one per committed WriteBatch.
//
// Record framing (little-endian):
//   u32 crc32   covers everything after this field
//   u32 len     payload length
//   u64 lsn
//   payload     WriteBatch encoding (see write_batch.h)
//
// A torn or corrupt tail, or the zero-filled preallocated tail, fails the
// length/CRC check and is truncated on replay.
#pragma once

#include <functional>
#include <string>
#include <string_view>

#include "kvdb/common.h"

namespace kvdb {

class Wal {
 public:
  explicit Wal(const std::string& path);
  ~Wal();
  Wal(const Wal&) = delete;
  Wal& operator=(const Wal&) = delete;

  // Calls fn for every intact record in order, truncates anything after the
  // last intact record, and returns the highest LSN seen (0 if none).
  uint64_t replay(const std::function<void(uint64_t, std::string_view)>& fn);

  // Appends one record with a single write() and then syncs per mode. After
  // this returns the record survives a process crash (and an OS crash too,
  // unless mode is kNone).
  void append(uint64_t lsn, std::string_view payload, SyncMode mode);

  // Empties the log (after a checkpoint made its contents redundant).
  void reset(SyncMode mode);

  // Logical size (end of the last record), not the preallocated file size.
  uint64_t size() const { return size_; }
  uint64_t bytes_appended() const { return bytes_appended_; }
  uint64_t syncs() const { return syncs_; }

  static constexpr size_t kHeaderSize = 16;
  // The file is grown in chunks with ftruncate so most appends land inside
  // the file. Measured on APFS: about 1.7x cheaper without sync and about
  // 1.1x with fsync (results/bench_walgrow.csv, DEVLOG.md).
  static constexpr uint64_t kPreallocBytes = 4ull << 20;

 private:
  int fd_ = -1;
  std::string path_;
  uint64_t size_ = 0;       // logical end of log
  uint64_t file_size_ = 0;  // physical size, >= size_ (zero-filled tail)
  uint64_t bytes_appended_ = 0;
  uint64_t syncs_ = 0;
  std::string buf_;  // reused scratch for framing
};

}  // namespace kvdb
