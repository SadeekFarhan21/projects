// End-to-end tests over real TCP connections to an in-process broker.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <set>
#include <thread>

#include "minikafka/broker.h"
#include "minikafka/client.h"
#include "minikafka/consumer.h"
#include "minikafka/producer.h"
#include "minikafka/record.h"
#include "test_util.h"

using namespace mk;
using namespace std::chrono_literals;

namespace {

BrokerConfig config_for(const test::TempDir& dir) {
  BrokerConfig c;
  c.data_dir = dir.path();
  c.port = 0;
  c.log.segment_bytes = 64 * 1024;  // small, so tests cross segment boundaries
  c.log.index_interval_bytes = 1024;
  return c;
}

// Reads a partition from `from` to the current end.
std::vector<Record> drain(Client& c, const std::string& topic, uint32_t p, uint64_t from = 0) {
  std::vector<Record> out;
  uint64_t pos = from;
  for (;;) {
    auto resp = c.fetch({{topic, p, pos, 64 * 1024}}, 0);
    EXPECT_EQ(resp.at(0).error, ErrorCode::None);
    auto recs = decode_records(resp[0].records.data(), resp[0].records.size());
    if (recs.empty()) break;
    for (auto& r : recs) {
      pos = r.offset + 1;
      out.push_back(std::move(r));
    }
  }
  return out;
}

}  // namespace

TEST(Broker, CreateTopicAndMetadata) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  EXPECT_EQ(c.create_topic("orders", 4), ErrorCode::None);
  EXPECT_EQ(c.create_topic("orders", 4), ErrorCode::TopicExists);
  EXPECT_EQ(c.create_topic("__internal", 1), ErrorCode::InvalidRequest);
  EXPECT_EQ(c.create_topic("bad/name", 1), ErrorCode::InvalidRequest);
  EXPECT_EQ(c.partition_count("orders"), 4);
  EXPECT_EQ(c.partition_count("missing"), -1);
  auto resp = c.fetch({{"missing", 0, 0, 1024}}, 0);
  EXPECT_EQ(resp[0].error, ErrorCode::UnknownTopic);
}

TEST(Broker, RejectsCorruptProduce) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  c.create_topic("t", 1);
  std::vector<uint8_t> batch;
  append_record(batch, 0, "k", "v");
  batch.back() ^= 0x5A;  // break the CRC
  try {
    c.produce("t", 0, batch, 1);
    FAIL() << "expected CorruptMessage";
  } catch (const BrokerError& e) {
    EXPECT_EQ(e.code, ErrorCode::CorruptMessage);
  }
  EXPECT_EQ(c.list_offsets("t", 0).second, 0u);  // nothing was written
}

TEST(Broker, OrderingWithinPartitionSingleProducer) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  c.create_topic("t", 1);
  {
    Producer p(c, {.batch_records = 37});
    for (int i = 0; i < 10000; ++i) p.send("t", "", "m" + std::to_string(i), 0);
  }
  auto recs = drain(c, "t", 0);
  ASSERT_EQ(recs.size(), 10000u);
  for (int i = 0; i < 10000; ++i) {
    ASSERT_EQ(recs[i].offset, static_cast<uint64_t>(i));
    ASSERT_EQ(recs[i].value, "m" + std::to_string(i));
  }
}

TEST(Broker, OrderingWithinPartitionConcurrentProducers) {
  // Several producers write to one partition at once. Offsets must be dense
  // (no gaps, no duplicates) and each producer's messages must appear in the
  // order it sent them.
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  {
    Client c("127.0.0.1", b.port());
    c.create_topic("t", 1);
  }
  constexpr int kProducers = 6, kPer = 3000;
  std::vector<std::thread> threads;
  for (int id = 0; id < kProducers; ++id) {
    threads.emplace_back([&, id] {
      Client c("127.0.0.1", b.port());
      Producer p(c, {.batch_records = static_cast<uint32_t>(5 + id * 7)});
      for (int i = 0; i < kPer; ++i) p.send("t", "p" + std::to_string(id), std::to_string(i), 0);
      p.flush();
    });
  }
  for (auto& t : threads) t.join();

  Client c("127.0.0.1", b.port());
  auto recs = drain(c, "t", 0);
  ASSERT_EQ(recs.size(), static_cast<size_t>(kProducers * kPer));
  std::map<std::string, int> last;
  for (size_t i = 0; i < recs.size(); ++i) {
    ASSERT_EQ(recs[i].offset, i);
    int v = std::stoi(recs[i].value);
    auto it = last.find(recs[i].key);
    int prev = it == last.end() ? -1 : it->second;
    ASSERT_EQ(v, prev + 1) << "producer " << recs[i].key << " out of order";
    last[recs[i].key] = v;
  }
}

TEST(Broker, KeyedMessagesStickToOnePartition) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  c.create_topic("t", 4);
  {
    Producer p(c);
    for (int i = 0; i < 400; ++i) p.send("t", "user-" + std::to_string(i % 10), std::to_string(i));
  }
  std::map<std::string, std::set<uint32_t>> where;
  size_t total = 0;
  for (uint32_t part = 0; part < 4; ++part) {
    for (auto& r : drain(c, "t", part)) {
      where[r.key].insert(part);
      ++total;
    }
  }
  EXPECT_EQ(total, 400u);
  for (auto& [k, parts] : where) EXPECT_EQ(parts.size(), 1u) << k;
}

TEST(Broker, DurableAcrossRestart) {
  test::TempDir dir;
  std::vector<std::string> sent;
  {
    Broker b(config_for(dir));
    b.start();
    Client c("127.0.0.1", b.port());
    c.create_topic("t", 3);
    Producer p(c, {.batch_records = 50});
    for (int i = 0; i < 6000; ++i) {
      sent.push_back("payload-" + std::to_string(i));
      p.send("t", "", sent.back(), i % 3);
    }
    p.flush();
    b.stop();
  }
  // A new broker over the same directory must see every acknowledged record.
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  ASSERT_EQ(c.partition_count("t"), 3);
  for (uint32_t part = 0; part < 3; ++part) {
    auto recs = drain(c, "t", part);
    ASSERT_EQ(recs.size(), 2000u);
    for (size_t i = 0; i < recs.size(); ++i) {
      ASSERT_EQ(recs[i].offset, i);
      ASSERT_EQ(recs[i].value, sent[i * 3 + part]);
    }
  }
  // Appends continue at the recovered end offset.
  std::vector<uint8_t> batch;
  append_record(batch, 0, "", "after-restart");
  EXPECT_EQ(c.produce("t", 0, batch, 1), 2000u);
}

TEST(Broker, LongPollFetchWakesOnProduce) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  c.create_topic("t", 1);

  std::thread producer([&] {
    std::this_thread::sleep_for(100ms);
    Client pc("127.0.0.1", b.port());
    std::vector<uint8_t> batch;
    append_record(batch, 0, "", "late");
    pc.produce("t", 0, batch, 1);
  });
  const auto t0 = std::chrono::steady_clock::now();
  auto resp = c.fetch({{"t", 0, 0, 1024}}, 5000);
  const auto waited = std::chrono::steady_clock::now() - t0;
  producer.join();
  auto recs = decode_records(resp[0].records.data(), resp[0].records.size());
  ASSERT_EQ(recs.size(), 1u);
  EXPECT_EQ(recs[0].value, "late");
  EXPECT_GE(waited, 90ms);
  EXPECT_LT(waited, 2000ms) << "fetch should return as soon as data arrives, not at the timeout";

  // With no data it returns empty after roughly max_wait.
  const auto t1 = std::chrono::steady_clock::now();
  auto empty = c.fetch({{"t", 0, 1, 1024}}, 150);
  EXPECT_TRUE(empty[0].records.empty());
  EXPECT_GE(std::chrono::steady_clock::now() - t1, 140ms);
}

TEST(Broker, RetentionThroughBroker) {
  test::TempDir dir;
  BrokerConfig cfg = config_for(dir);
  cfg.log.retention_bytes = 256 * 1024;
  Broker b(cfg);
  b.start();
  Client c("127.0.0.1", b.port());
  c.create_topic("t", 1);
  {
    Producer p(c, {.batch_records = 100});
    for (int i = 0; i < 20000; ++i) p.send("t", "", std::string(100, 'x'), 0);
  }
  auto [start, end] = c.list_offsets("t", 0);
  EXPECT_GT(start, 0u);
  EXPECT_EQ(end, 20000u);
  EXPECT_EQ(c.fetch({{"t", 0, 0, 1024}}, 0)[0].error, ErrorCode::OffsetOutOfRange);

  // A group consumer with no committed offset starts at the retained log start.
  Consumer cons(c, {.group = "g"});
  cons.subscribe({"t"});
  auto recs = cons.poll(100);
  ASSERT_FALSE(recs.empty());
  EXPECT_EQ(recs.front().offset, start);
}

TEST(Broker, OffsetCommitAndResume) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  {
    Client c("127.0.0.1", b.port());
    c.create_topic("t", 2);
    Producer p(c);
    for (int i = 0; i < 1000; ++i) p.send("t", "", std::to_string(i), i % 2);
  }

  std::map<uint32_t, std::vector<uint64_t>> seen;
  // First consumer reads a few polls, commits, and closes.
  uint64_t first_count = 0;
  {
    Client c("127.0.0.1", b.port());
    Consumer cons(c, {.group = "g", .max_bytes_per_partition = 2048});
    cons.subscribe({"t"});
    for (int k = 0; k < 3; ++k)
      for (auto& r : cons.poll(100)) {
        seen[std::stoi(r.value) % 2].push_back(r.offset);
        ++first_count;
      }
    ASSERT_GT(first_count, 0u);
    ASSERT_LT(first_count, 1000u);
    ASSERT_TRUE(cons.commit_sync());
    cons.close();
  }
  // A second consumer in the same group resumes exactly after the commit.
  {
    Client c("127.0.0.1", b.port());
    Consumer cons(c, {.group = "g"});
    cons.subscribe({"t"});
    uint64_t got = 0;
    for (int k = 0; k < 50 && first_count + got < 1000; ++k)
      for (auto& r : cons.poll(100)) {
        seen[std::stoi(r.value) % 2].push_back(r.offset);
        ++got;
      }
    EXPECT_EQ(first_count + got, 1000u);
  }
  // Every offset of every partition seen exactly once, in order.
  for (uint32_t part = 0; part < 2; ++part) {
    ASSERT_EQ(seen[part].size(), 500u);
    for (size_t i = 0; i < 500; ++i) ASSERT_EQ(seen[part][i], i);
  }
}

TEST(Broker, CommittedOffsetsSurviveRestart) {
  test::TempDir dir;
  {
    Broker b(config_for(dir));
    b.start();
    Client c("127.0.0.1", b.port());
    c.create_topic("t", 1);
    Producer p(c);
    for (int i = 0; i < 100; ++i) p.send("t", "", std::to_string(i), 0);
    p.flush();
    Consumer cons(c, {.group = "g", .max_bytes_per_partition = 1024});
    cons.subscribe({"t"});
    auto recs = cons.poll(100);
    ASSERT_FALSE(recs.empty());
    ASSERT_LT(recs.size(), 100u);
    ASSERT_TRUE(cons.commit_sync());
    EXPECT_EQ(c.committed_offset("g", "t", 0), recs.back().offset + 1);
  }
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  auto committed = c.committed_offset("g", "t", 0);
  ASSERT_TRUE(committed.has_value());
  Consumer cons(c, {.group = "g"});
  cons.subscribe({"t"});
  auto recs = cons.poll(100);
  ASSERT_FALSE(recs.empty());
  EXPECT_EQ(recs.front().offset, *committed);
}

TEST(Broker, GroupRebalanceSplitsPartitions) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client admin("127.0.0.1", b.port());
  admin.create_topic("t", 4);

  Client c1("127.0.0.1", b.port()), c2("127.0.0.1", b.port());
  Consumer a(c1, {.group = "g", .heartbeat_interval_ms = 20});
  Consumer bb(c2, {.group = "g", .heartbeat_interval_ms = 20});
  a.subscribe({"t"});
  bb.subscribe({"t"});
  a.poll(10);
  EXPECT_EQ(a.assignment().size(), 4u);
  bb.poll(10);
  // `a` notices the rebalance on its next heartbeat and rejoins.
  std::this_thread::sleep_for(30ms);
  a.poll(10);
  EXPECT_EQ(a.generation(), bb.generation());
  EXPECT_EQ(a.assignment().size(), 2u);
  EXPECT_EQ(bb.assignment().size(), 2u);
  std::set<TopicPartition> all(a.assignment().begin(), a.assignment().end());
  all.insert(bb.assignment().begin(), bb.assignment().end());
  EXPECT_EQ(all.size(), 4u);

  // After `bb` leaves, `a` gets everything back.
  bb.close();
  std::this_thread::sleep_for(30ms);
  a.poll(10);
  EXPECT_EQ(a.assignment().size(), 4u);
}

TEST(Broker, StaleGenerationCommitIsFenced) {
  test::TempDir dir;
  Broker b(config_for(dir));
  b.start();
  Client c("127.0.0.1", b.port());
  c.create_topic("t", 2);
  auto j1 = c.join_group("g", "", {"t"}, 10000);
  auto j2 = c.join_group("g", "", {"t"}, 10000);
  EXPECT_EQ(c.commit_offset("g", j1.member_id, j1.generation, "t", 0, 5), ErrorCode::IllegalGeneration);
  EXPECT_EQ(c.commit_offset("g", j2.member_id, j2.generation, "t", 0, 5), ErrorCode::None);
  EXPECT_EQ(c.committed_offset("g", "t", 0), 5u);
}
