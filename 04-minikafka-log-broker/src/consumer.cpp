#include "minikafka/consumer.h"

#include <thread>

namespace mk {

Consumer::Consumer(Client& client, ConsumerConfig cfg) : client_(client), cfg_(std::move(cfg)) {}

Consumer::~Consumer() {
  try {
    close();
  } catch (...) {
  }
}

void Consumer::subscribe(const std::vector<std::string>& topics) {
  topics_ = topics;
  joined_ = false;
}

void Consumer::join() {
  JoinResult jr = client_.join_group(cfg_.group, member_id_, topics_, cfg_.session_timeout_ms);
  if (jr.error != ErrorCode::None) throw BrokerError(jr.error, "join_group");
  member_id_ = jr.member_id;
  generation_ = jr.generation;
  ++rebalances_;

  // Keep positions for partitions we still own (we have been fetching them);
  // load committed offsets for newly assigned ones.
  std::map<TopicPartition, uint64_t> next;
  for (const auto& tp : jr.assignment) {
    auto it = positions_.find(tp);
    if (it != positions_.end()) {
      next[tp] = it->second;
      continue;
    }
    auto committed = client_.committed_offset(cfg_.group, tp.topic, tp.partition);
    next[tp] = committed ? *committed : client_.list_offsets(tp.topic, tp.partition).first;
  }
  positions_.swap(next);
  assignment_ = std::move(jr.assignment);
  joined_ = true;
  last_heartbeat_ = std::chrono::steady_clock::now();
}

void Consumer::maybe_heartbeat() {
  const auto now = std::chrono::steady_clock::now();
  if (now - last_heartbeat_ < std::chrono::milliseconds(cfg_.heartbeat_interval_ms)) return;
  last_heartbeat_ = now;
  ErrorCode e = client_.heartbeat(cfg_.group, member_id_, generation_);
  if (e == ErrorCode::RebalanceInProgress) {
    join();
  } else if (e == ErrorCode::UnknownMember) {
    member_id_.clear();  // we were expired; join as a new member
    positions_.clear();
    join();
  } else if (e != ErrorCode::None) {
    throw BrokerError(e, "heartbeat");
  }
}

std::vector<Record> Consumer::poll(uint32_t timeout_ms) {
  if (!joined_) join();
  maybe_heartbeat();
  std::vector<Record> out;
  if (assignment_.empty()) {
    // Nothing assigned (more members than partitions); wait out the timeout.
    std::this_thread::sleep_for(std::chrono::milliseconds(std::min<uint32_t>(timeout_ms, 100)));
    return out;
  }
  std::vector<FetchRequest> reqs;
  for (const auto& tp : assignment_)
    reqs.push_back({tp.topic, tp.partition, positions_[tp], cfg_.max_bytes_per_partition});
  // Keep the long poll shorter than the heartbeat interval so a slow topic does
  // not starve heartbeats.
  const uint32_t wait = std::min(timeout_ms, cfg_.heartbeat_interval_ms);
  for (auto& f : client_.fetch(reqs, wait)) {
    TopicPartition tp{f.topic, f.partition};
    if (f.error == ErrorCode::OffsetOutOfRange) {
      // Retention deleted what we wanted, or the position is past the end.
      positions_[tp] = positions_[tp] < f.log_start ? f.log_start : f.log_end;
      continue;
    }
    if (f.error != ErrorCode::None) throw BrokerError(f.error, "fetch " + f.topic);
    for (auto& r : decode_records(f.records.data(), f.records.size())) {
      if (r.offset < positions_[tp]) continue;
      positions_[tp] = r.offset + 1;
      out.push_back(std::move(r));
    }
  }
  return out;
}

bool Consumer::commit_sync() {
  if (!joined_) return true;
  for (const auto& tp : assignment_) {
    ErrorCode e = client_.commit_offset(cfg_.group, member_id_, generation_, tp.topic, tp.partition,
                                        positions_[tp]);
    if (e == ErrorCode::IllegalGeneration || e == ErrorCode::UnknownMember) return false;
    if (e != ErrorCode::None) throw BrokerError(e, "commit");
  }
  return true;
}

uint64_t Consumer::position(const TopicPartition& tp) const {
  auto it = positions_.find(tp);
  return it == positions_.end() ? 0 : it->second;
}

void Consumer::close() {
  if (joined_ && !member_id_.empty()) client_.leave_group(cfg_.group, member_id_);
  joined_ = false;
  member_id_.clear();
  assignment_.clear();
  positions_.clear();
}

}  // namespace mk
