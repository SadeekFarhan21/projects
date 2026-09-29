#include "kvdb/wal.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kvdb {

namespace {
void write_all(int fd, const char* p, size_t n, off_t off, const std::string& path) {
  size_t done = 0;
  while (done < n) {
    ssize_t w = ::pwrite(fd, p + done, n - done, off + static_cast<off_t>(done));
    if (w < 0) {
      if (errno == EINTR) continue;
      throw_errno("pwrite " + path);
    }
    done += static_cast<size_t>(w);
  }
}
}  // namespace

Wal::Wal(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) throw_errno("open " + path);
  struct stat st {};
  if (::fstat(fd_, &st) != 0) throw_errno("fstat " + path);
  size_ = file_size_ = static_cast<uint64_t>(st.st_size);
}

Wal::~Wal() {
  if (fd_ >= 0) ::close(fd_);
}

uint64_t Wal::replay(const std::function<void(uint64_t, std::string_view)>& fn) {
  // The log is bounded by the checkpoint threshold, so reading it whole is fine.
  std::string data(size_, '\0');
  size_t got = 0;
  while (got < data.size()) {
    ssize_t r = ::pread(fd_, data.data() + got, data.size() - got, static_cast<off_t>(got));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw_errno("pread " + path_);
    }
    if (r == 0) break;
    got += static_cast<size_t>(r);
  }
  data.resize(got);

  uint64_t last_lsn = 0;
  size_t off = 0;
  while (off + kHeaderSize <= data.size()) {
    const uint32_t crc = load<uint32_t>(data.data() + off);
    const uint32_t len = load<uint32_t>(data.data() + off + 4);
    if (len > data.size() - off - kHeaderSize) break;  // torn tail
    if (crc32(data.data() + off + 4, 12 + len) != crc) break;  // corrupt
    const uint64_t lsn = load<uint64_t>(data.data() + off + 8);
    if (lsn <= last_lsn) break;  // LSNs strictly increase; stale bytes
    fn(lsn, std::string_view(data.data() + off + kHeaderSize, len));
    last_lsn = lsn;
    off += kHeaderSize + len;
  }
  if (off != size_) {
    // Drop the invalid tail so new records are not appended after garbage.
    if (::ftruncate(fd_, static_cast<off_t>(off)) != 0) throw_errno("ftruncate " + path_);
    sync_fd(fd_, SyncMode::kFsync);
    size_ = file_size_ = off;
  }
  return last_lsn;
}

void Wal::append(uint64_t lsn, std::string_view payload, SyncMode mode) {
  buf_.resize(kHeaderSize + payload.size());
  char* p = buf_.data();
  store<uint32_t>(p + 4, static_cast<uint32_t>(payload.size()));
  store<uint64_t>(p + 8, lsn);
  std::memcpy(p + kHeaderSize, payload.data(), payload.size());
  store<uint32_t>(p, crc32(p + 4, 12 + payload.size()));
  if (size_ + buf_.size() > file_size_) {
    const uint64_t want = (size_ + buf_.size() + kPreallocBytes - 1) / kPreallocBytes * kPreallocBytes;
    if (::ftruncate(fd_, static_cast<off_t>(want)) != 0) throw_errno("ftruncate " + path_);
    file_size_ = want;
  }
  write_all(fd_, p, buf_.size(), static_cast<off_t>(size_), path_);
  size_ += buf_.size();
  bytes_appended_ += buf_.size();
  if (mode != SyncMode::kNone) {
    sync_fd(fd_, mode);
    ++syncs_;
  }
}

void Wal::reset(SyncMode mode) {
  if (::ftruncate(fd_, 0) != 0) throw_errno("ftruncate " + path_);
  sync_fd(fd_, mode);
  size_ = file_size_ = 0;
}

}  // namespace kvdb
