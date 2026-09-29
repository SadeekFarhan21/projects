// Pager: fixed-size page I/O over a single file. It knows nothing about page
// contents; the buffer pool is its only caller on the hot path.
#pragma once

#include <string>

#include "kvdb/common.h"

namespace kvdb {

class Pager {
 public:
  explicit Pager(const std::string& path);
  ~Pager();
  Pager(const Pager&) = delete;
  Pager& operator=(const Pager&) = delete;

  // Reads page id into out (kPageSize bytes). Pages past EOF read as zeros.
  void read_page(PageId id, char* out);
  void write_page(PageId id, const char* data);
  void sync(SyncMode mode);

  // Number of whole pages currently in the file.
  uint32_t file_pages() const;
  uint64_t reads() const { return reads_; }
  uint64_t writes() const { return writes_; }

 private:
  int fd_ = -1;
  std::string path_;
  uint64_t reads_ = 0;
  uint64_t writes_ = 0;
};

}  // namespace kvdb
