// One partition: an ordered list of segments forming a single append-only log.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include "minikafka/protocol.h"
#include "minikafka/segment.h"

namespace mk {

struct LogConfig {
  uint64_t segment_bytes = 64ull << 20;   // roll to a new segment past this size (must be < 4 GiB)
  uint32_t index_interval_bytes = 4096;   // bytes between sparse index entries
  int64_t retention_bytes = -1;           // -1 keeps everything
  FlushPolicy flush = FlushPolicy::None;
};

struct ReadResult {
  ErrorCode error = ErrorCode::None;
  std::vector<uint8_t> data;  // whole records, byte-identical to the segment file
  uint64_t log_end = 0;       // next offset to be assigned (the "high watermark" with one replica)
  uint64_t log_start = 0;     // first offset still retained
};

class PartitionLog {
 public:
  PartitionLog(std::filesystem::path dir, LogConfig cfg);

  // Assigns offsets to the `count` records in `batch` (patching the offset
  // fields in place), writes them to the active segment and returns the first
  // offset. The batch must already be validated.
  uint64_t append(uint8_t* batch, size_t n, uint32_t count);

  ReadResult read(uint64_t offset, uint32_t max_bytes) const;

  uint64_t log_end() const;
  uint64_t log_start() const;
  uint64_t size_bytes() const;
  size_t segment_count() const;
  const RecoveryStats& recovery_stats() const { return recovery_; }

 private:
  void roll();
  void enforce_retention();

  std::filesystem::path dir_;
  LogConfig cfg_;
  mutable std::mutex mu_;
  std::vector<std::shared_ptr<Segment>> segments_;  // sorted by base offset, last is active
  uint64_t total_bytes_ = 0;
  RecoveryStats recovery_;
};

}  // namespace mk
