#include "minikafka/producer.h"

#include <chrono>

#include "minikafka/record.h"

namespace mk {

namespace {
uint32_t fnv1a(std::string_view s) {
  uint32_t h = 2166136261u;
  for (unsigned char c : s) {
    h ^= c;
    h *= 16777619u;
  }
  return h;
}
}  // namespace

Producer::Producer(Client& client, ProducerConfig cfg) : client_(client), cfg_(cfg) {}

Producer::~Producer() {
  try {
    flush();
  } catch (...) {
    // Destructors must not throw; callers that care call flush() themselves.
  }
}

uint32_t Producer::pick_partition(const std::string& topic, std::string_view key) {
  auto it = partitions_.find(topic);
  if (it == partitions_.end()) {
    const int32_t n = client_.partition_count(topic);
    if (n <= 0) throw BrokerError(ErrorCode::UnknownTopic, "producer metadata for " + topic);
    it = partitions_.emplace(topic, static_cast<uint32_t>(n)).first;
  }
  return key.empty() ? rr_++ % it->second : fnv1a(key) % it->second;
}

void Producer::send(const std::string& topic, std::string_view key, std::string_view value,
                    int32_t partition, int64_t timestamp_ms) {
  const uint32_t p = partition >= 0 ? static_cast<uint32_t>(partition) : pick_partition(topic, key);
  if (timestamp_ms < 0)
    timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  Key k{topic, p};
  Pending& pend = pending_[k];
  append_record(pend.buf, timestamp_ms, key, value);
  ++pend.count;
  if (pend.count >= cfg_.batch_records || pend.buf.size() >= cfg_.batch_bytes) send_batch(k, pend);
}

void Producer::send_batch(const Key& k, Pending& p) {
  if (p.count == 0) return;
  p.last_base = client_.produce(k.first, k.second, p.buf, p.count);
  ++requests_;
  p.buf.clear();
  p.count = 0;
}

void Producer::flush() {
  for (auto& [k, p] : pending_) send_batch(k, p);
}

uint64_t Producer::last_base_offset(const std::string& topic, uint32_t partition) const {
  auto it = pending_.find({topic, partition});
  return it == pending_.end() ? 0 : it->second.last_base;
}

}  // namespace mk
