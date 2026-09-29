// The broker: owns topics, the offset store and the group coordinator, and
// serves the binary protocol over TCP with one thread per connection.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "minikafka/bytes.h"
#include "minikafka/group_coordinator.h"
#include "minikafka/offset_store.h"
#include "minikafka/partition_log.h"
#include "minikafka/protocol.h"

namespace mk {

struct BrokerConfig {
  std::filesystem::path data_dir = "data";
  std::string host = "127.0.0.1";
  uint16_t port = 0;  // 0 picks a free port; read it back with Broker::port()
  LogConfig log;
};

class Broker {
 public:
  explicit Broker(BrokerConfig cfg);
  ~Broker();
  Broker(const Broker&) = delete;
  Broker& operator=(const Broker&) = delete;

  // Loads topics from disk (running recovery), replays committed offsets and
  // starts accepting connections.
  void start();
  void stop();
  uint16_t port() const { return port_; }

  // Handles one request frame (without the length prefix) and returns the full
  // response frame (with its prefix). Public so tests can drive it directly.
  std::vector<uint8_t> handle(std::vector<uint8_t>& request);

  int32_t partition_count(const std::string& topic) const;
  PartitionLog* partition(const std::string& topic, uint32_t p) const;

 private:
  struct Topic {
    std::vector<std::unique_ptr<PartitionLog>> partitions;
  };

  void load_topics();
  ErrorCode create_topic(const std::string& name, uint32_t partitions);
  void accept_loop();
  void serve(int fd, uint64_t conn_id);
  void reap_finished_connections();

  ErrorCode handle_create_topic(Reader& r, Writer& w);
  ErrorCode handle_metadata(Reader& r, Writer& w);
  ErrorCode handle_produce(Reader& r, Writer& w, std::vector<uint8_t>& req);
  ErrorCode handle_fetch(Reader& r, Writer& w);
  ErrorCode handle_list_offsets(Reader& r, Writer& w);
  ErrorCode handle_offset_commit(Reader& r, Writer& w);
  ErrorCode handle_offset_fetch(Reader& r, Writer& w);
  ErrorCode handle_join(Reader& r, Writer& w);
  ErrorCode handle_heartbeat(Reader& r, Writer& w);
  ErrorCode handle_leave(Reader& r, Writer& w);

  BrokerConfig cfg_;
  uint16_t port_ = 0;

  mutable std::shared_mutex topics_mu_;
  std::map<std::string, Topic> topics_;
  std::unique_ptr<OffsetStore> offsets_;
  GroupCoordinator groups_;

  // Long-poll support: fetches with no data wait here until a produce bumps the epoch.
  std::mutex data_mu_;
  std::condition_variable data_cv_;
  uint64_t data_epoch_ = 0;
  std::atomic<int> fetch_waiters_{0};

  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};
  int listen_fd_ = -1;
  int wake_pipe_[2] = {-1, -1};
  std::thread accept_thread_;
  std::mutex conn_mu_;
  std::unordered_set<int> conn_fds_;
  std::map<uint64_t, std::thread> conn_threads_;
  std::vector<uint64_t> finished_conns_;  // exited threads waiting to be joined
  uint64_t next_conn_id_ = 0;
};

}  // namespace mk
