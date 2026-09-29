#include "kvdb/pager.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kvdb {

Pager::Pager(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) throw_errno("open " + path);
  // One process per database file. The lock dies with the process, which is
  // exactly what the kill -9 crash tests rely on.
  if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd_);
    throw_errno("database is locked by another process: " + path);
  }
}

Pager::~Pager() {
  if (fd_ >= 0) ::close(fd_);
}

void Pager::read_page(PageId id, char* out) {
  const off_t off = static_cast<off_t>(id) * kPageSize;
  size_t done = 0;
  while (done < kPageSize) {
    ssize_t n = ::pread(fd_, out + done, kPageSize - done, off + done);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw_errno("pread " + path_);
    }
    if (n == 0) {  // past EOF: treat the remainder as zeros
      std::memset(out + done, 0, kPageSize - done);
      break;
    }
    done += static_cast<size_t>(n);
  }
  ++reads_;
}

void Pager::write_page(PageId id, const char* data) {
  const off_t off = static_cast<off_t>(id) * kPageSize;
  size_t done = 0;
  while (done < kPageSize) {
    ssize_t n = ::pwrite(fd_, data + done, kPageSize - done, off + done);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw_errno("pwrite " + path_);
    }
    done += static_cast<size_t>(n);
  }
  ++writes_;
}

void Pager::sync(SyncMode mode) { sync_fd(fd_, mode); }

uint32_t Pager::file_pages() const {
  struct stat st {};
  if (::fstat(fd_, &st) != 0) throw_errno("fstat " + path_);
  return static_cast<uint32_t>(st.st_size / static_cast<off_t>(kPageSize));
}

}  // namespace kvdb
