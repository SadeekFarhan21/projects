#include "minikafka/partition_log.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "minikafka/bytes.h"
#include "minikafka/record.h"

namespace mk {
namespace fs = std::filesystem;

PartitionLog::PartitionLog(fs::path dir, LogConfig cfg) : dir_(std::move(dir)), cfg_(cfg) {
  if (cfg_.segment_bytes >= (4ull << 30)) throw std::invalid_argument("segment_bytes must be < 4 GiB");
  fs::create_directories(dir_);

  std::vector<uint64_t> bases;
  for (const auto& e : fs::directory_iterator(dir_)) {
    if (e.path().extension() != ".log") continue;
    try {
      bases.push_back(std::stoull(e.path().stem().string()));
    } catch (...) {
      // not one of ours
    }
  }
  std::sort(bases.begin(), bases.end());

  // Sealed segments trust their index; only the last (active) segment can have
  // a torn tail, so only it is fully scanned and CRC-checked on startup.
  for (size_t i = 0; i < bases.size(); ++i) {
    const bool last = i + 1 == bases.size();
    auto seg = Segment::open(dir_, bases[i], cfg_.index_interval_bytes, last, &recovery_);
    if (!last) seg->set_next_offset(bases[i + 1]);
    total_bytes_ += seg->size();
    segments_.push_back(std::move(seg));
  }
  if (segments_.empty()) segments_.push_back(Segment::create(dir_, 0, cfg_.index_interval_bytes));
}

void PartitionLog::roll() {
  const uint64_t base = segments_.back()->next_offset();
  // The outgoing segment is flushed with the configured policy before the new
  // one is created, so a sealed segment is never left half-written.
  segments_.back()->flush(cfg_.flush);
  segments_.push_back(Segment::create(dir_, base, cfg_.index_interval_bytes));
}

void PartitionLog::enforce_retention() {
  if (cfg_.retention_bytes < 0) return;
  const auto limit = static_cast<uint64_t>(cfg_.retention_bytes);
  // Delete the oldest segment only if what remains is still >= retention_bytes,
  // so the log never drops below the retention target, and never delete the
  // active segment.
  while (segments_.size() > 1 && total_bytes_ - segments_.front()->size() >= limit) {
    total_bytes_ -= segments_.front()->size();
    segments_.front()->remove_files();
    segments_.erase(segments_.begin());
  }
}

uint64_t PartitionLog::append(uint8_t* batch, size_t n, uint32_t count) {
  std::lock_guard lk(mu_);
  Segment* active = segments_.back().get();
  if (active->size() > 0 && active->size() + n > cfg_.segment_bytes) {
    roll();
    active = segments_.back().get();
  }
  const uint64_t base = active->next_offset();
  // Patch offsets into the record headers in place.
  size_t pos = 0;
  for (uint32_t i = 0; i < count; ++i) {
    store_u64(batch + pos, base + i);
    pos += kRecordHeaderSize + load_u32(batch + pos + 8);
  }
  if (pos != n) throw std::logic_error("batch length does not match record count");
  active->append(batch, n, base, count);
  total_bytes_ += n;
  active->flush(cfg_.flush);
  enforce_retention();
  return base;
}

ReadResult PartitionLog::read(uint64_t offset, uint32_t max_bytes) const {
  ReadResult res;
  std::shared_ptr<Segment> seg;
  uint64_t start_pos = 0, limit = 0;
  {
    std::lock_guard lk(mu_);
    res.log_start = segments_.front()->base_offset();
    res.log_end = segments_.back()->next_offset();
    if (offset < res.log_start || offset > res.log_end) {
      res.error = ErrorCode::OffsetOutOfRange;
      return res;
    }
    if (offset == res.log_end) return res;  // caught up
    // Last segment whose base offset is <= offset.
    auto it = std::upper_bound(segments_.begin(), segments_.end(), offset,
                               [](uint64_t v, const std::shared_ptr<Segment>& s) {
                                 return v < s->base_offset();
                               });
    seg = *std::prev(it);
    start_pos = seg->floor_position(offset);
    limit = seg->size();  // snapshot: bytes below this are immutable
  }
  // The file read happens outside the lock so fetches never block produces.
  // The shared_ptr keeps the fd open even if retention deletes the segment now.
  res.data = seg->read_from(start_pos, offset, max_bytes, limit);
  return res;
}

uint64_t PartitionLog::log_end() const {
  std::lock_guard lk(mu_);
  return segments_.back()->next_offset();
}

uint64_t PartitionLog::log_start() const {
  std::lock_guard lk(mu_);
  return segments_.front()->base_offset();
}

uint64_t PartitionLog::size_bytes() const {
  std::lock_guard lk(mu_);
  return total_bytes_;
}

size_t PartitionLog::segment_count() const {
  std::lock_guard lk(mu_);
  return segments_.size();
}

}  // namespace mk
