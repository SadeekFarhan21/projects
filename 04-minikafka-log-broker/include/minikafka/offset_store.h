// Committed consumer offsets, persisted in an internal log.
//
// Every commit is appended as a record to the partition log
// `<data_dir>/__consumer_offsets-0` with key "group\0topic\0partition" and an
// 8-byte value holding the offset. The latest record per key wins. On startup
// the log is replayed from the beginning to rebuild the in-memory map. This is
// the same idea as Kafka's __consumer_offsets topic, minus compaction (so the
// log grows until retention or compaction, which is on the roadmap).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "minikafka/partition_log.h"

namespace mk {

class OffsetStore {
 public:
  OffsetStore(const std::filesystem::path& data_dir, LogConfig cfg);

  void commit(const std::string& group, const std::string& topic, uint32_t partition,
              uint64_t offset);
  std::optional<uint64_t> fetch(const std::string& group, const std::string& topic,
                                uint32_t partition) const;
  size_t replayed_records() const { return replayed_; }

  static constexpr const char* kTopicName = "__consumer_offsets";

 private:
  static std::string make_key(const std::string& group, const std::string& topic, uint32_t partition);

  std::unique_ptr<PartitionLog> log_;
  mutable std::mutex mu_;
  std::map<std::string, uint64_t> offsets_;
  size_t replayed_ = 0;
};

}  // namespace mk
