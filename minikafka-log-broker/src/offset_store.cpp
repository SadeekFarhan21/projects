#include "minikafka/offset_store.h"

#include <chrono>
#include <cstring>
#include <vector>

#include "minikafka/record.h"

namespace mk {

OffsetStore::OffsetStore(const std::filesystem::path& data_dir, LogConfig cfg) {
  // Retention would silently forget offsets without compaction, so it is off
  // for the internal log regardless of the broker-wide setting.
  cfg.retention_bytes = -1;
  log_ = std::make_unique<PartitionLog>(data_dir / (std::string(kTopicName) + "-0"), cfg);

  uint64_t pos = log_->log_start();
  const uint64_t end = log_->log_end();
  while (pos < end) {
    ReadResult r = log_->read(pos, 1 << 20);
    if (r.error != ErrorCode::None || r.data.empty()) break;
    for (const auto& rec : parse_records(r.data.data(), r.data.size())) {
      if (rec.value.size() == 8) {
        uint64_t off;
        std::memcpy(&off, rec.value.data(), 8);
        offsets_[std::string(rec.key)] = off;
      }
      pos = rec.offset + 1;
      ++replayed_;
    }
  }
}

std::string OffsetStore::make_key(const std::string& group, const std::string& topic,
                                  uint32_t partition) {
  std::string k = group;
  k.push_back('\0');
  k += topic;
  k.push_back('\0');
  k += std::to_string(partition);
  return k;
}

void OffsetStore::commit(const std::string& group, const std::string& topic, uint32_t partition,
                         uint64_t offset) {
  const std::string key = make_key(group, topic, partition);
  std::vector<uint8_t> batch;
  const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  append_record(batch, now, key,
                std::string_view(reinterpret_cast<const char*>(&offset), sizeof(offset)));
  // Holding mu_ across the append keeps the log order and the map order the
  // same, so replay reproduces exactly the state readers observed.
  std::lock_guard lk(mu_);
  log_->append(batch.data(), batch.size(), 1);
  offsets_[key] = offset;
}

std::optional<uint64_t> OffsetStore::fetch(const std::string& group, const std::string& topic,
                                           uint32_t partition) const {
  std::lock_guard lk(mu_);
  auto it = offsets_.find(make_key(group, topic, partition));
  if (it == offsets_.end()) return std::nullopt;
  return it->second;
}

}  // namespace mk
