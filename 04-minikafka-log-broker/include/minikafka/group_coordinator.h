// Consumer group membership and partition assignment.
//
// This is a simplified, broker-side version of Kafka's group protocol:
//   * JoinGroup adds (or refreshes) a member. Any membership change bumps the
//     group generation and recomputes the assignment on the broker with a range
//     assignor, so there is no separate SyncGroup round or client-side leader.
//   * Heartbeat keeps a member alive and tells it when the generation moved on
//     (RebalanceInProgress), at which point the client simply joins again.
//   * Members that miss their session timeout are expired lazily on the next
//     coordinator call.
//   * Commits carrying a member id must carry the current generation, so a
//     consumer that has been rebalanced away cannot overwrite the new owner's
//     progress. Commits with an empty member id are accepted as-is.
// There is no join barrier: during a rebalance two members can briefly fetch the
// same partition. Commits are fenced by generation, so the result is duplicate
// delivery (at-least-once), never lost offsets.
#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "minikafka/protocol.h"

namespace mk {

struct TopicPartition {
  std::string topic;
  uint32_t partition = 0;
  auto operator<=>(const TopicPartition&) const = default;
};

struct JoinResult {
  ErrorCode error = ErrorCode::None;
  std::string member_id;
  uint32_t generation = 0;
  std::vector<TopicPartition> assignment;
};

class GroupCoordinator {
 public:
  using Clock = std::chrono::steady_clock;
  // Returns the partition count of a topic, or -1 if it does not exist.
  using PartitionsOf = std::function<int32_t(const std::string&)>;

  explicit GroupCoordinator(PartitionsOf partitions_of);

  JoinResult join(const std::string& group, const std::string& member_id,
                  const std::vector<std::string>& topics, uint32_t session_timeout_ms);
  ErrorCode heartbeat(const std::string& group, const std::string& member_id, uint32_t generation);
  ErrorCode leave(const std::string& group, const std::string& member_id);
  ErrorCode check_commit(const std::string& group, const std::string& member_id, uint32_t generation);

  uint32_t generation(const std::string& group);

 private:
  struct Member {
    std::vector<std::string> topics;
    uint32_t session_timeout_ms = 10000;
    Clock::time_point last_seen;
  };
  struct Group {
    uint32_t generation = 0;
    std::map<std::string, Member> members;  // ordered, so assignment is deterministic
    std::map<std::string, std::vector<TopicPartition>> assignment;
  };

  void expire(Group& g, Clock::time_point now);
  void rebalance(Group& g);

  PartitionsOf partitions_of_;
  std::mutex mu_;
  std::map<std::string, Group> groups_;
  uint64_t next_member_ = 1;
};

}  // namespace mk
