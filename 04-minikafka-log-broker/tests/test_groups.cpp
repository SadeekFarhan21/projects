#include <gtest/gtest.h>

#include <set>
#include <thread>

#include "minikafka/group_coordinator.h"

using namespace mk;

namespace {
GroupCoordinator make() {
  return GroupCoordinator([](const std::string& t) -> int32_t {
    if (t == "a") return 6;
    if (t == "b") return 3;
    return -1;
  });
}
std::set<TopicPartition> as_set(const std::vector<TopicPartition>& v) { return {v.begin(), v.end()}; }
}  // namespace

TEST(GroupCoordinator, SingleMemberGetsEverything) {
  auto gc = make();
  auto j = gc.join("g", "", {"a", "b"}, 10000);
  EXPECT_EQ(j.error, ErrorCode::None);
  EXPECT_FALSE(j.member_id.empty());
  EXPECT_EQ(j.generation, 1u);
  EXPECT_EQ(j.assignment.size(), 9u);
}

TEST(GroupCoordinator, AssignmentIsDisjointAndComplete) {
  auto gc = make();
  auto j1 = gc.join("g", "", {"a"}, 10000);
  auto j2 = gc.join("g", "", {"a"}, 10000);
  auto j3 = gc.join("g", "", {"a"}, 10000);
  EXPECT_EQ(j3.generation, 3u);
  // Earlier members learn about the rebalance through heartbeat and rejoin.
  EXPECT_EQ(gc.heartbeat("g", j1.member_id, j1.generation), ErrorCode::RebalanceInProgress);
  j1 = gc.join("g", j1.member_id, {"a"}, 10000);
  j2 = gc.join("g", j2.member_id, {"a"}, 10000);
  EXPECT_EQ(j1.generation, 3u);
  EXPECT_EQ(j2.generation, 3u);
  std::set<TopicPartition> all;
  for (auto* j : {&j1, &j2, &j3}) {
    EXPECT_EQ(j->assignment.size(), 2u);
    for (auto& tp : j->assignment) EXPECT_TRUE(all.insert(tp).second) << "partition assigned twice";
  }
  EXPECT_EQ(all.size(), 6u);
  EXPECT_EQ(gc.heartbeat("g", j1.member_id, 3), ErrorCode::None);
}

TEST(GroupCoordinator, LeaveTriggersRebalanceAndFencesOldGeneration) {
  auto gc = make();
  auto j1 = gc.join("g", "", {"a"}, 10000);
  auto j2 = gc.join("g", "", {"a"}, 10000);
  j1 = gc.join("g", j1.member_id, {"a"}, 10000);
  EXPECT_EQ(gc.check_commit("g", j1.member_id, j1.generation), ErrorCode::None);
  EXPECT_EQ(gc.leave("g", j2.member_id), ErrorCode::None);
  // j1's generation is now stale: its commits are fenced until it rejoins.
  EXPECT_EQ(gc.check_commit("g", j1.member_id, j1.generation), ErrorCode::IllegalGeneration);
  auto again = gc.join("g", j1.member_id, {"a"}, 10000);
  EXPECT_EQ(as_set(again.assignment).size(), 6u);
  EXPECT_EQ(gc.check_commit("g", j1.member_id, again.generation), ErrorCode::None);
  EXPECT_EQ(gc.check_commit("g", j2.member_id, again.generation), ErrorCode::UnknownMember);
  EXPECT_EQ(gc.check_commit("g", "", 0), ErrorCode::None);  // standalone commits are unfenced
}

TEST(GroupCoordinator, ExpiredMembersAreRemoved) {
  auto gc = make();
  auto j1 = gc.join("g", "", {"a"}, 50);
  auto j2 = gc.join("g", "", {"a"}, 10000);
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  EXPECT_EQ(gc.heartbeat("g", j2.member_id, j2.generation), ErrorCode::RebalanceInProgress);
  auto again = gc.join("g", j2.member_id, {"a"}, 10000);
  EXPECT_EQ(again.assignment.size(), 6u);
  EXPECT_EQ(gc.heartbeat("g", j1.member_id, j1.generation), ErrorCode::UnknownMember);
}
