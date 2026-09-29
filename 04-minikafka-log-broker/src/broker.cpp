#include "minikafka/broker.h"

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <set>

#include "minikafka/net.h"
#include "minikafka/record.h"

namespace mk {
namespace fs = std::filesystem;

namespace {

bool valid_topic_name(const std::string& name) {
  if (name.empty() || name.size() > 200 || name.rfind("__", 0) == 0) return false;
  for (char c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-')) return false;
  }
  return true;
}

// "<topic>-<partition>" -> (topic, partition). Splits on the last '-', so topic
// names may themselves contain dashes.
bool parse_partition_dir(const std::string& dir, std::string* topic, uint32_t* partition) {
  auto dash = dir.rfind('-');
  if (dash == std::string::npos || dash == 0 || dash + 1 == dir.size()) return false;
  for (size_t i = dash + 1; i < dir.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(dir[i]))) return false;
  *topic = dir.substr(0, dash);
  *partition = static_cast<uint32_t>(std::stoul(dir.substr(dash + 1)));
  return true;
}

}  // namespace

Broker::Broker(BrokerConfig cfg)
    : cfg_(std::move(cfg)),
      groups_([this](const std::string& t) { return partition_count(t); }) {}

Broker::~Broker() { stop(); }

void Broker::load_topics() {
  fs::create_directories(cfg_.data_dir);
  std::map<std::string, uint32_t> counts;
  for (const auto& e : fs::directory_iterator(cfg_.data_dir)) {
    if (!e.is_directory()) continue;
    std::string topic;
    uint32_t p;
    if (!parse_partition_dir(e.path().filename().string(), &topic, &p)) continue;
    if (!valid_topic_name(topic)) continue;  // skips internal "__" topics
    counts[topic] = std::max(counts[topic], p + 1);
  }
  std::unique_lock lk(topics_mu_);
  for (const auto& [name, n] : counts) {
    Topic t;
    for (uint32_t p = 0; p < n; ++p)
      t.partitions.push_back(std::make_unique<PartitionLog>(
          cfg_.data_dir / (name + "-" + std::to_string(p)), cfg_.log));
    topics_.emplace(name, std::move(t));
  }
}

void Broker::start() {
  if (running_) return;
  load_topics();
  offsets_ = std::make_unique<OffsetStore>(cfg_.data_dir, cfg_.log);
  listen_fd_ = listen_tcp(cfg_.host, cfg_.port, &port_);
  if (::pipe(wake_pipe_) != 0) throw std::runtime_error("pipe failed");
  stopping_ = false;
  running_ = true;
  accept_thread_ = std::thread([this] { accept_loop(); });
}

void Broker::stop() {
  if (!running_.exchange(false)) return;
  stopping_ = true;
  char b = 1;
  (void)!::write(wake_pipe_[1], &b, 1);
  if (accept_thread_.joinable()) accept_thread_.join();
  {
    std::lock_guard lk(conn_mu_);
    for (int fd : conn_fds_) ::shutdown(fd, SHUT_RDWR);  // wakes threads blocked in recv
  }
  {
    std::lock_guard lk(data_mu_);
    ++data_epoch_;
  }
  data_cv_.notify_all();  // wakes long-polling fetches
  std::map<uint64_t, std::thread> threads;
  {
    std::lock_guard lk(conn_mu_);
    threads.swap(conn_threads_);
    finished_conns_.clear();
  }
  for (auto& [_, t] : threads) t.join();
  ::close(listen_fd_);
  ::close(wake_pipe_[0]);
  ::close(wake_pipe_[1]);
  listen_fd_ = -1;
}

void Broker::reap_finished_connections() {
  std::vector<std::thread> done;
  {
    std::lock_guard lk(conn_mu_);
    for (uint64_t id : finished_conns_) {
      auto it = conn_threads_.find(id);
      if (it != conn_threads_.end()) {
        done.push_back(std::move(it->second));
        conn_threads_.erase(it);
      }
    }
    finished_conns_.clear();
  }
  for (auto& t : done) t.join();
}

void Broker::accept_loop() {
  // poll() on the listen socket plus a self-pipe: on macOS, closing or
  // shutting down a listening socket does not reliably wake a blocked accept().
  while (!stopping_) {
    pollfd pfds[2] = {{listen_fd_, POLLIN, 0}, {wake_pipe_[0], POLLIN, 0}};
    int rc = ::poll(pfds, 2, 1000);
    reap_finished_connections();
    if (rc < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (pfds[1].revents) break;
    if (!(pfds[0].revents & POLLIN)) continue;
    int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) continue;
    set_socket_options(fd);
    std::lock_guard lk(conn_mu_);
    const uint64_t id = next_conn_id_++;
    conn_fds_.insert(fd);
    conn_threads_.emplace(id, std::thread([this, fd, id] { serve(fd, id); }));
  }
}

void Broker::serve(int fd, uint64_t conn_id) {
  std::vector<uint8_t> req;
  try {
    while (!stopping_) {
      if (!read_frame(fd, req, kMaxFrameSize)) break;
      std::vector<uint8_t> resp = handle(req);
      write_full(fd, resp.data(), resp.size());
    }
  } catch (const std::exception& e) {
    if (!stopping_) std::fprintf(stderr, "[broker] connection error: %s\n", e.what());
  }
  std::lock_guard lk(conn_mu_);
  conn_fds_.erase(fd);
  ::close(fd);
  finished_conns_.push_back(conn_id);
}

int32_t Broker::partition_count(const std::string& topic) const {
  std::shared_lock lk(topics_mu_);
  auto it = topics_.find(topic);
  return it == topics_.end() ? -1 : static_cast<int32_t>(it->second.partitions.size());
}

PartitionLog* Broker::partition(const std::string& topic, uint32_t p) const {
  std::shared_lock lk(topics_mu_);
  auto it = topics_.find(topic);
  if (it == topics_.end() || p >= it->second.partitions.size()) return nullptr;
  // Topics are never deleted in v0, so the pointer stays valid after unlock.
  return it->second.partitions[p].get();
}

ErrorCode Broker::create_topic(const std::string& name, uint32_t partitions) {
  if (!valid_topic_name(name) || partitions == 0 || partitions > 10000) return ErrorCode::InvalidRequest;
  std::unique_lock lk(topics_mu_);
  if (topics_.count(name)) return ErrorCode::TopicExists;
  Topic t;
  for (uint32_t p = 0; p < partitions; ++p)
    t.partitions.push_back(std::make_unique<PartitionLog>(
        cfg_.data_dir / (name + "-" + std::to_string(p)), cfg_.log));
  topics_.emplace(name, std::move(t));
  return ErrorCode::None;
}

std::vector<uint8_t> Broker::handle(std::vector<uint8_t>& req) {
  Reader r(req.data(), req.size());
  const auto api = static_cast<ApiKey>(r.u16());
  const uint32_t corr = r.u32();

  Writer w;
  w.u32(0);     // frame length, patched below
  w.u32(corr);
  w.u16(0);     // error code, patched below
  const size_t header = w.buf.size();

  ErrorCode err;
  try {
    switch (api) {
      case ApiKey::CreateTopic: err = handle_create_topic(r, w); break;
      case ApiKey::Metadata: err = handle_metadata(r, w); break;
      case ApiKey::Produce: err = handle_produce(r, w, req); break;
      case ApiKey::Fetch: err = handle_fetch(r, w); break;
      case ApiKey::ListOffsets: err = handle_list_offsets(r, w); break;
      case ApiKey::OffsetCommit: err = handle_offset_commit(r, w); break;
      case ApiKey::OffsetFetch: err = handle_offset_fetch(r, w); break;
      case ApiKey::JoinGroup: err = handle_join(r, w); break;
      case ApiKey::Heartbeat: err = handle_heartbeat(r, w); break;
      case ApiKey::LeaveGroup: err = handle_leave(r, w); break;
      default: err = ErrorCode::UnknownApi; break;
    }
  } catch (const DecodeError&) {
    err = ErrorCode::InvalidRequest;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[broker] internal error: %s\n", e.what());
    err = ErrorCode::InternalError;
  }
  if (err != ErrorCode::None) w.buf.resize(header);  // error responses carry no body
  store_u32(w.buf.data(), static_cast<uint32_t>(w.buf.size() - 4));
  const auto e16 = static_cast<uint16_t>(err);
  std::memcpy(w.buf.data() + 8, &e16, 2);
  return std::move(w.buf);
}

ErrorCode Broker::handle_create_topic(Reader& r, Writer&) {
  std::string name = r.str();
  uint32_t n = r.u32();
  return create_topic(name, n);
}

ErrorCode Broker::handle_metadata(Reader& r, Writer& w) {
  int32_t n = partition_count(r.str());
  if (n < 0) return ErrorCode::UnknownTopic;
  w.u32(static_cast<uint32_t>(n));
  return ErrorCode::None;
}

ErrorCode Broker::handle_produce(Reader& r, Writer& w, std::vector<uint8_t>& req) {
  std::string topic = r.str();
  uint32_t p = r.u32();
  uint32_t count = r.u32();
  uint32_t len = r.u32();
  std::string_view blob = r.view(len);
  PartitionLog* log = partition(topic, p);
  if (!log) return ErrorCode::UnknownTopic;
  // The records are validated (lengths and CRCs) before they touch the log, so
  // a corrupt or malicious batch can never make the on-disk log unreadable.
  uint8_t* data = req.data() + (reinterpret_cast<const uint8_t*>(blob.data()) - req.data());
  std::string why;
  const int64_t n = validate_batch(data, len, &why);
  if (n < 0 || static_cast<uint32_t>(n) != count || count == 0) return ErrorCode::CorruptMessage;
  const uint64_t base = log->append(data, len, count);
  if (fetch_waiters_.load(std::memory_order_acquire) > 0) {
    {
      std::lock_guard lk(data_mu_);
      ++data_epoch_;
    }
    data_cv_.notify_all();
  }
  w.u64(base);
  return ErrorCode::None;
}

ErrorCode Broker::handle_fetch(Reader& r, Writer& w) {
  struct Req {
    std::string topic;
    uint32_t partition;
    uint64_t offset;
    uint32_t max_bytes;
  };
  const uint32_t max_wait_ms = r.u32();
  const uint32_t n = r.u32();
  if (n > 100000) return ErrorCode::InvalidRequest;
  std::vector<Req> reqs(n);
  for (auto& q : reqs) {
    q.topic = r.str();
    q.partition = r.u32();
    q.offset = r.u64();
    q.max_bytes = r.u32();
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_wait_ms);
  std::vector<ReadResult> results(n);
  // Register as a waiter before the first read. A producer checks the waiter
  // count after its append; the partition mutex orders our read against that
  // append, so either our read sees the data or the producer sees us waiting.
  struct WaiterGuard {
    std::atomic<int>* c;
    ~WaiterGuard() {
      if (c) c->fetch_sub(1, std::memory_order_acq_rel);
    }
  } guard{max_wait_ms > 0 ? &fetch_waiters_ : nullptr};
  if (guard.c) guard.c->fetch_add(1, std::memory_order_acq_rel);
  for (;;) {
    // Snapshot the epoch before reading so a produce that lands between the
    // read and the wait is not missed.
    uint64_t epoch;
    {
      std::lock_guard lk(data_mu_);
      epoch = data_epoch_;
    }
    bool any = false;
    for (uint32_t i = 0; i < n; ++i) {
      PartitionLog* log = partition(reqs[i].topic, reqs[i].partition);
      if (!log) {
        results[i] = ReadResult{};
        results[i].error = ErrorCode::UnknownTopic;
        any = true;
        continue;
      }
      results[i] = log->read(reqs[i].offset, reqs[i].max_bytes);
      if (!results[i].data.empty() || results[i].error != ErrorCode::None) any = true;
    }
    if (any || max_wait_ms == 0 || stopping_ || std::chrono::steady_clock::now() >= deadline) break;
    std::unique_lock lk(data_mu_);
    data_cv_.wait_until(lk, deadline, [&] { return data_epoch_ != epoch || stopping_.load(); });
  }

  w.u32(n);
  for (uint32_t i = 0; i < n; ++i) {
    w.str(reqs[i].topic);
    w.u32(reqs[i].partition);
    w.u16(static_cast<uint16_t>(results[i].error));
    w.u64(results[i].log_end);
    w.u64(results[i].log_start);
    w.blob(results[i].data.data(), results[i].data.size());
  }
  return ErrorCode::None;
}

ErrorCode Broker::handle_list_offsets(Reader& r, Writer& w) {
  std::string topic = r.str();
  uint32_t p = r.u32();
  PartitionLog* log = partition(topic, p);
  if (!log) return ErrorCode::UnknownTopic;
  w.u64(log->log_start());
  w.u64(log->log_end());
  return ErrorCode::None;
}

ErrorCode Broker::handle_offset_commit(Reader& r, Writer&) {
  std::string group = r.str();
  std::string member = r.str();
  uint32_t generation = r.u32();
  std::string topic = r.str();
  uint32_t p = r.u32();
  uint64_t offset = r.u64();
  if (group.empty()) return ErrorCode::InvalidRequest;
  if (!partition(topic, p)) return ErrorCode::UnknownTopic;
  ErrorCode e = groups_.check_commit(group, member, generation);
  if (e != ErrorCode::None) return e;
  offsets_->commit(group, topic, p, offset);
  return ErrorCode::None;
}

ErrorCode Broker::handle_offset_fetch(Reader& r, Writer& w) {
  std::string group = r.str();
  std::string topic = r.str();
  uint32_t p = r.u32();
  auto off = offsets_->fetch(group, topic, p);
  w.u8(off ? 1 : 0);
  w.u64(off.value_or(0));
  return ErrorCode::None;
}

ErrorCode Broker::handle_join(Reader& r, Writer& w) {
  std::string group = r.str();
  std::string member = r.str();
  uint32_t session_ms = r.u32();
  uint32_t n = r.u32();
  if (n > 10000 || group.empty()) return ErrorCode::InvalidRequest;
  std::vector<std::string> topics(n);
  for (auto& t : topics) t = r.str();
  JoinResult jr = groups_.join(group, member, topics, session_ms);
  if (jr.error != ErrorCode::None) return jr.error;
  w.str(jr.member_id);
  w.u32(jr.generation);
  w.u32(static_cast<uint32_t>(jr.assignment.size()));
  for (const auto& tp : jr.assignment) {
    w.str(tp.topic);
    w.u32(tp.partition);
  }
  return ErrorCode::None;
}

ErrorCode Broker::handle_heartbeat(Reader& r, Writer&) {
  std::string group = r.str();
  std::string member = r.str();
  uint32_t generation = r.u32();
  return groups_.heartbeat(group, member, generation);
}

ErrorCode Broker::handle_leave(Reader& r, Writer&) {
  std::string group = r.str();
  std::string member = r.str();
  return groups_.leave(group, member);
}

}  // namespace mk
