#include "minikafka/client.h"

#include <unistd.h>

#include "minikafka/net.h"

namespace mk {

Client::Client(const std::string& host, uint16_t port) : fd_(connect_tcp(host, port)) {}

Client::~Client() {
  if (fd_ >= 0) ::close(fd_);
}

ErrorCode Client::call(ApiKey api, const Writer& body, Reader** out, const uint8_t* tail,
                       size_t tail_len) {
  const uint32_t corr = next_corr_++;
  const uint32_t len = static_cast<uint32_t>(2 + 4 + body.buf.size() + tail_len);
  // Header and body are sent with one write so small requests are one segment.
  req_.resize(4 + 2 + 4);
  store_u32(req_.data(), len);
  const auto a = static_cast<uint16_t>(api);
  std::memcpy(req_.data() + 4, &a, 2);
  store_u32(req_.data() + 6, corr);
  req_.insert(req_.end(), body.buf.begin(), body.buf.end());
  write_full(fd_, req_.data(), req_.size());
  // Large payloads (produce batches) are sent straight from the caller's buffer
  // instead of being copied into the request buffer first.
  if (tail_len > 0) write_full(fd_, tail, tail_len);

  if (!read_frame(fd_, resp_, kMaxFrameSize)) throw std::runtime_error("broker closed connection");
  reader_.emplace(resp_.data(), resp_.size());
  if (reader_->u32() != corr) throw std::runtime_error("correlation id mismatch");
  const auto err = static_cast<ErrorCode>(reader_->u16());
  if (out) *out = &*reader_;
  return err;
}

ErrorCode Client::create_topic(const std::string& topic, uint32_t partitions) {
  Writer w;
  w.str(topic);
  w.u32(partitions);
  return call(ApiKey::CreateTopic, w, nullptr);
}

int32_t Client::partition_count(const std::string& topic) {
  Writer w;
  w.str(topic);
  Reader* r;
  if (call(ApiKey::Metadata, w, &r) != ErrorCode::None) return -1;
  return static_cast<int32_t>(r->u32());
}

uint64_t Client::produce(const std::string& topic, uint32_t partition,
                         const std::vector<uint8_t>& batch, uint32_t count) {
  Writer w;
  w.str(topic);
  w.u32(partition);
  w.u32(count);
  w.u32(static_cast<uint32_t>(batch.size()));
  Reader* r;
  ErrorCode e = call(ApiKey::Produce, w, &r, batch.data(), batch.size());
  if (e != ErrorCode::None) throw BrokerError(e, "produce");
  return r->u64();
}

std::vector<FetchResponse> Client::fetch(const std::vector<FetchRequest>& reqs, uint32_t max_wait_ms) {
  Writer w;
  w.u32(max_wait_ms);
  w.u32(static_cast<uint32_t>(reqs.size()));
  for (const auto& q : reqs) {
    w.str(q.topic);
    w.u32(q.partition);
    w.u64(q.offset);
    w.u32(q.max_bytes);
  }
  Reader* r;
  ErrorCode e = call(ApiKey::Fetch, w, &r);
  if (e != ErrorCode::None) throw BrokerError(e, "fetch");
  std::vector<FetchResponse> out(r->u32());
  for (auto& f : out) {
    f.topic = r->str();
    f.partition = r->u32();
    f.error = static_cast<ErrorCode>(r->u16());
    f.log_end = r->u64();
    f.log_start = r->u64();
    const uint32_t n = r->u32();
    auto v = r->view(n);
    f.records.assign(v.begin(), v.end());
  }
  return out;
}

std::pair<uint64_t, uint64_t> Client::list_offsets(const std::string& topic, uint32_t partition) {
  Writer w;
  w.str(topic);
  w.u32(partition);
  Reader* r;
  ErrorCode e = call(ApiKey::ListOffsets, w, &r);
  if (e != ErrorCode::None) throw BrokerError(e, "list_offsets");
  uint64_t start = r->u64();
  uint64_t end = r->u64();
  return {start, end};
}

ErrorCode Client::commit_offset(const std::string& group, const std::string& member,
                                uint32_t generation, const std::string& topic, uint32_t partition,
                                uint64_t offset) {
  Writer w;
  w.str(group);
  w.str(member);
  w.u32(generation);
  w.str(topic);
  w.u32(partition);
  w.u64(offset);
  return call(ApiKey::OffsetCommit, w, nullptr);
}

std::optional<uint64_t> Client::committed_offset(const std::string& group, const std::string& topic,
                                                 uint32_t partition) {
  Writer w;
  w.str(group);
  w.str(topic);
  w.u32(partition);
  Reader* r;
  ErrorCode e = call(ApiKey::OffsetFetch, w, &r);
  if (e != ErrorCode::None) throw BrokerError(e, "offset_fetch");
  const bool found = r->u8() != 0;
  const uint64_t off = r->u64();
  if (!found) return std::nullopt;
  return off;
}

JoinResult Client::join_group(const std::string& group, const std::string& member,
                              const std::vector<std::string>& topics, uint32_t session_timeout_ms) {
  Writer w;
  w.str(group);
  w.str(member);
  w.u32(session_timeout_ms);
  w.u32(static_cast<uint32_t>(topics.size()));
  for (const auto& t : topics) w.str(t);
  Reader* r;
  JoinResult jr;
  jr.error = call(ApiKey::JoinGroup, w, &r);
  if (jr.error != ErrorCode::None) return jr;
  jr.member_id = r->str();
  jr.generation = r->u32();
  jr.assignment.resize(r->u32());
  for (auto& tp : jr.assignment) {
    tp.topic = r->str();
    tp.partition = r->u32();
  }
  return jr;
}

ErrorCode Client::heartbeat(const std::string& group, const std::string& member, uint32_t generation) {
  Writer w;
  w.str(group);
  w.str(member);
  w.u32(generation);
  return call(ApiKey::Heartbeat, w, nullptr);
}

ErrorCode Client::leave_group(const std::string& group, const std::string& member) {
  Writer w;
  w.str(group);
  w.str(member);
  return call(ApiKey::LeaveGroup, w, nullptr);
}

}  // namespace mk
