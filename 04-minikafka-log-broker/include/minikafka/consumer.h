// Group consumer. Joins a consumer group, fetches from its assigned partitions,
// and commits positions to the broker's offset store.
//
// Position semantics follow Kafka: a committed offset is the offset of the next
// record to read. A partition with no committed offset starts at the log start
// (auto.offset.reset=earliest).
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "minikafka/client.h"
#include "minikafka/record.h"

namespace mk {

struct ConsumerConfig {
  std::string group;
  uint32_t session_timeout_ms = 10000;
  uint32_t heartbeat_interval_ms = 1000;
  uint32_t max_bytes_per_partition = 1 << 20;
};

class Consumer {
 public:
  Consumer(Client& client, ConsumerConfig cfg);
  ~Consumer();

  void subscribe(const std::vector<std::string>& topics);
  // Returns the next records from the assigned partitions, waiting up to
  // timeout_ms on the broker (long poll) if nothing is available.
  std::vector<Record> poll(uint32_t timeout_ms);
  // Commits the current position of every assigned partition. Returns false if
  // the broker fenced the commit because the group rebalanced.
  bool commit_sync();
  void close();  // leaves the group

  const std::vector<TopicPartition>& assignment() const { return assignment_; }
  const std::string& member_id() const { return member_id_; }
  uint32_t generation() const { return generation_; }
  uint64_t position(const TopicPartition& tp) const;
  uint64_t rebalances() const { return rebalances_; }

 private:
  void join();
  void maybe_heartbeat();

  Client& client_;
  ConsumerConfig cfg_;
  std::vector<std::string> topics_;
  std::string member_id_;
  uint32_t generation_ = 0;
  bool joined_ = false;
  std::vector<TopicPartition> assignment_;
  std::map<TopicPartition, uint64_t> positions_;
  std::chrono::steady_clock::time_point last_heartbeat_;
  uint64_t rebalances_ = 0;
};

}  // namespace mk
