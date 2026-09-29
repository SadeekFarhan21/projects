// Batching producer. Records are accumulated per partition and sent as one
// Produce request when a batch reaches batch_records or batch_bytes, or on flush().
// Requests are synchronous (one in flight per connection), which preserves
// per-partition order trivially; pipelining is a later optimisation.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "minikafka/client.h"

namespace mk {

struct ProducerConfig {
  uint32_t batch_records = 100;
  uint32_t batch_bytes = 1 << 20;
};

class Producer {
 public:
  Producer(Client& client, ProducerConfig cfg = {});
  ~Producer();

  // partition < 0 picks one: hash(key) for non-empty keys, round-robin otherwise.
  // timestamp_ms < 0 uses the wall clock.
  void send(const std::string& topic, std::string_view key, std::string_view value,
            int32_t partition = -1, int64_t timestamp_ms = -1);
  void flush();

  uint64_t requests_sent() const { return requests_; }
  // Base offset returned by the most recent produce request for a partition.
  uint64_t last_base_offset(const std::string& topic, uint32_t partition) const;

 private:
  struct Pending {
    std::vector<uint8_t> buf;
    uint32_t count = 0;
    uint64_t last_base = 0;
  };
  using Key = std::pair<std::string, uint32_t>;

  uint32_t pick_partition(const std::string& topic, std::string_view key);
  void send_batch(const Key& k, Pending& p);

  Client& client_;
  ProducerConfig cfg_;
  std::map<Key, Pending> pending_;
  std::map<std::string, uint32_t> partitions_;
  uint32_t rr_ = 0;
  uint64_t requests_ = 0;
};

}  // namespace mk
