// A log segment: one append-only data file plus a sparse offset index.
//
//   <dir>/<base offset, 20 digits>.log    records in the format of record.h
//   <dir>/<base offset, 20 digits>.index  8-byte entries {u32 offset - base, u32 file position}
//
// An index entry is added whenever at least `index_interval` bytes were written
// since the previous entry, so the index stays tiny (one entry per ~4 KiB) and a
// lookup costs one binary search plus a short forward scan of the data file.
//
// Thread safety: the owning PartitionLog serialises all mutations and index
// lookups under its mutex. read_from() only touches the file descriptor and the
// arguments it is given, so it may run without the lock on bytes below a size
// snapshot taken under the lock (those bytes are immutable).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace mk {

enum class FlushPolicy {
  None,       // rely on the OS page cache (survives a process crash, not a power cut)
  Fsync,      // fsync() after every append; on macOS this does NOT flush the drive cache
  FullFsync,  // fcntl(F_FULLFSYNC) after every append; true durability on macOS
};

struct IndexEntry {
  uint32_t rel_offset;
  uint32_t position;
};

struct RecoveryStats {
  uint64_t records_scanned = 0;
  uint64_t bytes_truncated = 0;
};

class Segment {
 public:
  static std::shared_ptr<Segment> create(const std::filesystem::path& dir, uint64_t base_offset,
                                         uint32_t index_interval);
  // Opens an existing segment. If `recover` is true the data file is scanned,
  // every record is CRC-checked, a torn or corrupt tail is truncated and the index
  // is rebuilt. Otherwise the index file is trusted (sealed segments) and
  // `next_offset` must be supplied by the caller.
  static std::shared_ptr<Segment> open(const std::filesystem::path& dir, uint64_t base_offset,
                                       uint32_t index_interval, bool recover, RecoveryStats* stats);

  ~Segment();
  Segment(const Segment&) = delete;
  Segment& operator=(const Segment&) = delete;

  uint64_t base_offset() const { return base_; }
  uint64_t next_offset() const { return next_offset_; }
  void set_next_offset(uint64_t v) { next_offset_ = v; }
  uint64_t size() const { return size_; }
  size_t index_entries() const { return index_.size(); }

  // Appends `n` bytes holding `count` records whose offsets start at first_offset.
  void append(const uint8_t* data, size_t n, uint64_t first_offset, uint32_t count);

  // Largest indexed file position whose offset is <= `offset` (0 if none).
  uint64_t floor_position(uint64_t offset) const;

  // Returns whole records with offset >= `offset`, starting the scan at
  // `start_pos` and never reading at or beyond `limit`. Returns at most
  // `max_bytes`, except that the first record is always returned whole so a
  // consumer can never get stuck behind a record larger than its fetch size.
  std::vector<uint8_t> read_from(uint64_t start_pos, uint64_t offset, uint32_t max_bytes,
                                 uint64_t limit) const;

  void flush(FlushPolicy policy);
  void remove_files();  // unlinks both files; open fds stay valid for readers

  static std::string file_stem(uint64_t base_offset);

 private:
  Segment(std::filesystem::path log_path, std::filesystem::path index_path, uint64_t base,
          uint32_t index_interval);
  void recover(RecoveryStats* stats);
  void load_index();

  std::filesystem::path log_path_;
  std::filesystem::path index_path_;
  int log_fd_ = -1;
  int index_fd_ = -1;
  uint64_t base_;
  uint32_t index_interval_;
  uint64_t size_ = 0;
  uint64_t next_offset_;
  uint64_t bytes_since_index_ = 0;
  std::vector<IndexEntry> index_;
};

}  // namespace mk
