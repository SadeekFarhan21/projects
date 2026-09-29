#include "minikafka/segment.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include "minikafka/bytes.h"
#include "minikafka/crc32c.h"
#include "minikafka/record.h"

namespace mk {
namespace fs = std::filesystem;

namespace {

[[noreturn]] void throw_errno(const std::string& what) {
  throw std::system_error(errno, std::generic_category(), what);
}

void pwrite_all(int fd, const uint8_t* p, size_t n, uint64_t off) {
  while (n > 0) {
    ssize_t w = ::pwrite(fd, p, n, static_cast<off_t>(off));
    if (w < 0) {
      if (errno == EINTR) continue;
      throw_errno("pwrite");
    }
    p += w;
    n -= static_cast<size_t>(w);
    off += static_cast<uint64_t>(w);
  }
}

void pread_all(int fd, uint8_t* p, size_t n, uint64_t off) {
  while (n > 0) {
    ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw_errno("pread");
    }
    if (r == 0) throw std::runtime_error("unexpected EOF in segment");
    p += r;
    n -= static_cast<size_t>(r);
    off += static_cast<uint64_t>(r);
  }
}

uint64_t file_size(int fd) {
  struct stat st {};
  if (::fstat(fd, &st) != 0) throw_errno("fstat");
  return static_cast<uint64_t>(st.st_size);
}

// Buffered random access over [0, limit) of a file. at(pos, n) returns a pointer
// to n bytes at pos, re-reading a window starting at pos when needed. This keeps
// a fetch to one or two pread calls instead of one per record header.
class ChunkReader {
 public:
  ChunkReader(int fd, uint64_t limit, size_t chunk) : fd_(fd), limit_(limit), chunk_(chunk) {}

  const uint8_t* at(uint64_t pos, size_t n) {
    if (pos < buf_pos_ || pos + n > buf_pos_ + buf_.size()) {
      size_t want = std::max(n, chunk_);
      want = static_cast<size_t>(std::min<uint64_t>(want, limit_ - pos));
      if (want < n) throw std::runtime_error("read past segment limit");
      buf_.resize(want);
      pread_all(fd_, buf_.data(), want, pos);
      buf_pos_ = pos;
    }
    return buf_.data() + (pos - buf_pos_);
  }

 private:
  int fd_;
  uint64_t limit_;
  size_t chunk_;
  std::vector<uint8_t> buf_;
  uint64_t buf_pos_ = 0;
};

}  // namespace

std::string Segment::file_stem(uint64_t base_offset) {
  char name[32];
  std::snprintf(name, sizeof(name), "%020llu", static_cast<unsigned long long>(base_offset));
  return name;
}

Segment::Segment(fs::path log_path, fs::path index_path, uint64_t base, uint32_t index_interval)
    : log_path_(std::move(log_path)),
      index_path_(std::move(index_path)),
      base_(base),
      index_interval_(index_interval),
      next_offset_(base) {}

Segment::~Segment() {
  if (log_fd_ >= 0) ::close(log_fd_);
  if (index_fd_ >= 0) ::close(index_fd_);
}

std::shared_ptr<Segment> Segment::create(const fs::path& dir, uint64_t base_offset,
                                         uint32_t index_interval) {
  const std::string stem = file_stem(base_offset);
  std::shared_ptr<Segment> s(
      new Segment(dir / (stem + ".log"), dir / (stem + ".index"), base_offset, index_interval));
  s->log_fd_ = ::open(s->log_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (s->log_fd_ < 0) throw_errno("open " + s->log_path_.string());
  s->index_fd_ = ::open(s->index_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (s->index_fd_ < 0) throw_errno("open " + s->index_path_.string());
  return s;
}

std::shared_ptr<Segment> Segment::open(const fs::path& dir, uint64_t base_offset,
                                       uint32_t index_interval, bool recover,
                                       RecoveryStats* stats) {
  const std::string stem = file_stem(base_offset);
  std::shared_ptr<Segment> s(
      new Segment(dir / (stem + ".log"), dir / (stem + ".index"), base_offset, index_interval));
  s->log_fd_ = ::open(s->log_path_.c_str(), O_RDWR | O_CLOEXEC);
  if (s->log_fd_ < 0) throw_errno("open " + s->log_path_.string());
  s->index_fd_ = ::open(s->index_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (s->index_fd_ < 0) throw_errno("open " + s->index_path_.string());
  s->size_ = file_size(s->log_fd_);
  if (recover) {
    s->recover(stats);
  } else {
    s->load_index();
  }
  return s;
}

void Segment::load_index() {
  const uint64_t isz = file_size(index_fd_);
  const size_t n = static_cast<size_t>(isz / sizeof(IndexEntry));
  index_.resize(n);
  if (n > 0) pread_all(index_fd_, reinterpret_cast<uint8_t*>(index_.data()), n * sizeof(IndexEntry), 0);
  // Sanity check: strictly increasing and inside the data file. A bad index is
  // not fatal, it only costs a rebuild.
  bool ok = isz % sizeof(IndexEntry) == 0;
  for (size_t i = 0; ok && i < n; ++i) {
    if (index_[i].position >= size_) ok = false;
    if (i > 0 && (index_[i].rel_offset <= index_[i - 1].rel_offset ||
                  index_[i].position <= index_[i - 1].position))
      ok = false;
  }
  if (!ok) recover(nullptr);
}

void Segment::recover(RecoveryStats* stats) {
  // Scan every record from the start, stopping at the first one that is
  // truncated, fails its CRC, or has a non-contiguous offset. Everything from
  // there on is a torn write from a crash and is cut off.
  index_.clear();
  bytes_since_index_ = 0;
  const uint64_t limit = size_;
  ChunkReader r(log_fd_, limit, 1 << 20);
  uint64_t pos = 0;
  uint64_t expect = base_;
  uint64_t scanned = 0;
  while (limit - pos >= kRecordHeaderSize) {
    const uint8_t* h = r.at(pos, kRecordHeaderSize);
    const uint64_t off = load_u64(h);
    const uint32_t len = load_u32(h + 8);
    const uint32_t crc = load_u32(h + 12);
    if (off != expect || len < kPayloadFixedSize || len > kMaxPayloadSize ||
        len > limit - pos - kRecordHeaderSize)
      break;
    const uint8_t* body = r.at(pos + kRecordHeaderSize, len);
    if (crc32c(body, len) != crc) break;
    if (bytes_since_index_ >= index_interval_) {
      index_.push_back({static_cast<uint32_t>(off - base_), static_cast<uint32_t>(pos)});
      bytes_since_index_ = 0;
    }
    const uint64_t rec = kRecordHeaderSize + len;
    bytes_since_index_ += rec;
    pos += rec;
    ++expect;
    ++scanned;
  }
  if (pos != size_) {
    if (::ftruncate(log_fd_, static_cast<off_t>(pos)) != 0) throw_errno("ftruncate log");
  }
  if (stats) {
    stats->records_scanned += scanned;
    stats->bytes_truncated += size_ - pos;
  }
  size_ = pos;
  next_offset_ = expect;
  // Rewrite the index file from the rebuilt entries.
  if (::ftruncate(index_fd_, 0) != 0) throw_errno("ftruncate index");
  if (!index_.empty())
    pwrite_all(index_fd_, reinterpret_cast<const uint8_t*>(index_.data()),
               index_.size() * sizeof(IndexEntry), 0);
}

void Segment::append(const uint8_t* data, size_t n, uint64_t first_offset, uint32_t count) {
  // Index entries are placed at record boundaries inside the batch, not only at
  // the batch start. With 1 MiB batches, indexing only batch starts left up to
  // 1 MiB to scan on every fetch that starts mid-batch (see DEVLOG, Problems).
  const size_t old_entries = index_.size();
  size_t pos = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (bytes_since_index_ >= index_interval_) {
      index_.push_back({static_cast<uint32_t>(first_offset + i - base_), static_cast<uint32_t>(size_ + pos)});
      bytes_since_index_ = 0;
    }
    const size_t rec = kRecordHeaderSize + load_u32(data + pos + 8);
    bytes_since_index_ += rec;
    pos += rec;
  }
  pwrite_all(log_fd_, data, n, size_);
  // New index entries go out in one write, after the data they point at.
  if (index_.size() > old_entries)
    pwrite_all(index_fd_, reinterpret_cast<const uint8_t*>(index_.data() + old_entries),
               (index_.size() - old_entries) * sizeof(IndexEntry), old_entries * sizeof(IndexEntry));
  size_ += n;
  next_offset_ = first_offset + count;
}

uint64_t Segment::floor_position(uint64_t offset) const {
  if (offset <= base_ || index_.empty()) return 0;
  const uint64_t rel = offset - base_;
  // First entry with rel_offset > rel, then step back one.
  auto it = std::upper_bound(index_.begin(), index_.end(), rel,
                             [](uint64_t v, const IndexEntry& e) { return v < e.rel_offset; });
  if (it == index_.begin()) return 0;
  return std::prev(it)->position;
}

std::vector<uint8_t> Segment::read_from(uint64_t start_pos, uint64_t offset, uint32_t max_bytes,
                                        uint64_t limit) const {
  std::vector<uint8_t> out;
  if (start_pos >= limit) return out;
  ChunkReader r(log_fd_, limit, std::max<size_t>(64 << 10, size_t{max_bytes} + index_interval_ + 4096));
  // Phase 1: skip forward from the index position to the first record >= offset.
  uint64_t pos = start_pos;
  while (limit - pos >= kRecordHeaderSize) {
    const uint8_t* h = r.at(pos, kRecordHeaderSize);
    if (load_u64(h) >= offset) break;
    pos += kRecordHeaderSize + load_u32(h + 8);
  }
  // Phase 2: take whole records until max_bytes (the first one always fits).
  const uint64_t start = pos;
  while (limit - pos >= kRecordHeaderSize) {
    const uint8_t* h = r.at(pos, kRecordHeaderSize);
    const uint64_t rec = kRecordHeaderSize + load_u32(h + 8);
    if (rec > limit - pos) break;
    if (pos > start && pos - start + rec > max_bytes) break;
    pos += rec;
  }
  if (pos > start) {
    const uint8_t* p = r.at(start, static_cast<size_t>(pos - start));
    out.assign(p, p + (pos - start));
  }
  return out;
}

void Segment::flush(FlushPolicy policy) {
  switch (policy) {
    case FlushPolicy::None:
      return;
    case FlushPolicy::Fsync:
      if (::fsync(log_fd_) != 0) throw_errno("fsync");
      return;
    case FlushPolicy::FullFsync:
#ifdef F_FULLFSYNC
      if (::fcntl(log_fd_, F_FULLFSYNC) != 0) throw_errno("F_FULLFSYNC");
#else
      if (::fsync(log_fd_) != 0) throw_errno("fsync");
#endif
      return;
  }
}

void Segment::remove_files() {
  std::error_code ec;
  fs::remove(log_path_, ec);
  fs::remove(index_path_, ec);
}

}  // namespace mk
