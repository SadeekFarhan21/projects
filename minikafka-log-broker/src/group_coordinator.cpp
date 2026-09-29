#include "minikafka/group_coordinator.h"

#include <algorithm>
#include <set>

namespace mk {

GroupCoordinator::GroupCoordinator(PartitionsOf partitions_of)
    : partitions_of_(std::move(partitions_of)) {}

void GroupCoordinator::expire(Group& g, Clock::time_point now) {
  bool changed = false;
  for (auto it = g.members.begin(); it != g.members.end();) {
    if (now - it->second.last_seen > std::chrono::milliseconds(it->second.session_timeout_ms)) {
      it = g.members.erase(it);
      changed = true;
    } else {
      ++it;
    }
  }
  if (changed) rebalance(g);
}

void GroupCoordinator::rebalance(Group& g) {
  ++g.generation;
  g.assignment.clear();
  for (const auto& [id, _] : g.members) g.assignment[id];

  // Range assignor, per topic: sort the subscribed members, then give each a
  // contiguous block of partitions, with the first (n % members) getting one extra.
  std::set<std::string> topics;
  for (const auto& [_, m] : g.members) topics.insert(m.topics.begin(), m.topics.end());
  for (const auto& topic : topics) {
    const int32_t n = partitions_of_(topic);
    if (n <= 0) continue;
    std::vector<std::string> subs;
    for (const auto& [id, m] : g.members)
      if (std::find(m.topics.begin(), m.topics.end(), topic) != m.topics.end()) subs.push_back(id);
    const uint32_t per = static_cast<uint32_t>(n) / subs.size();
    const uint32_t extra = static_cast<uint32_t>(n) % subs.size();
    uint32_t p = 0;
    for (size_t i = 0; i < subs.size(); ++i) {
      const uint32_t take = per + (i < extra ? 1 : 0);
      for (uint32_t k = 0; k < take; ++k) g.assignment[subs[i]].push_back({topic, p++});
    }
  }
}

JoinResult GroupCoordinator::join(const std::string& group, const std::string& member_id,
                                  const std::vector<std::string>& topics,
                                  uint32_t session_timeout_ms) {
  std::lock_guard lk(mu_);
  Group& g = groups_[group];
  const auto now = Clock::now();
  expire(g, now);

  JoinResult res;
  res.member_id = member_id.empty() ? "member-" + std::to_string(next_member_++) : member_id;
  std::vector<std::string> sorted = topics;
  std::sort(sorted.begin(), sorted.end());

  auto it = g.members.find(res.member_id);
  const bool is_new = it == g.members.end();
  if (is_new || it->second.topics != sorted) {
    g.members[res.member_id] = Member{sorted, session_timeout_ms, now};
    rebalance(g);
  } else {
    it->second.last_seen = now;
    it->second.session_timeout_ms = session_timeout_ms;
  }
  res.generation = g.generation;
  res.assignment = g.assignment[res.member_id];
  return res;
}

ErrorCode GroupCoordinator::heartbeat(const std::string& group, const std::string& member_id,
                                      uint32_t generation) {
  std::lock_guard lk(mu_);
  auto git = groups_.find(group);
  if (git == groups_.end()) return ErrorCode::UnknownMember;
  Group& g = git->second;
  expire(g, Clock::now());
  auto it = g.members.find(member_id);
  if (it == g.members.end()) return ErrorCode::UnknownMember;
  it->second.last_seen = Clock::now();
  if (generation != g.generation) return ErrorCode::RebalanceInProgress;
  return ErrorCode::None;
}

ErrorCode GroupCoordinator::leave(const std::string& group, const std::string& member_id) {
  std::lock_guard lk(mu_);
  auto git = groups_.find(group);
  if (git == groups_.end()) return ErrorCode::UnknownMember;
  if (git->second.members.erase(member_id) == 0) return ErrorCode::UnknownMember;
  rebalance(git->second);
  return ErrorCode::None;
}

ErrorCode GroupCoordinator::check_commit(const std::string& group, const std::string& member_id,
                                         uint32_t generation) {
  if (member_id.empty()) return ErrorCode::None;  // standalone consumer, no fencing
  std::lock_guard lk(mu_);
  auto git = groups_.find(group);
  if (git == groups_.end()) return ErrorCode::UnknownMember;
  Group& g = git->second;
  expire(g, Clock::now());
  if (!g.members.count(member_id)) return ErrorCode::UnknownMember;
  if (generation != g.generation) return ErrorCode::IllegalGeneration;
  return ErrorCode::None;
}

uint32_t GroupCoordinator::generation(const std::string& group) {
  std::lock_guard lk(mu_);
  return groups_[group].generation;
}

}  // namespace mk
