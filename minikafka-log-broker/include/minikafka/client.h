// Low-level client: one blocking TCP connection, one request in flight.
// Producer and Consumer (producer.h, consumer.h) are built on top of it.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "minikafka/bytes.h"
#include "minikafka/group_coordinator.h"
#include "minikafka/protocol.h"

namespace mk {

struct BrokerError : std::runtime_error {
  ErrorCode code;
  BrokerError(ErrorCode c, const std::string& what)
      : std::runtime_error(what + ": " + error_name(c)), code(c) {}
};

struct FetchRequest {
  std::string topic;
  uint32_t partition = 0;
  uint64_t offset = 0;
  uint32_t max_bytes = 1 << 20;
};

struct FetchResponse {
  std::string topic;
  uint32_t partition = 0;
  ErrorCode error = ErrorCode::None;
  uint64_t log_end = 0;
  uint64_t log_start = 0;
  std::vector<uint8_t> records;  // raw record bytes; parse with parse_records()
};

class Client {
 public:
  Client(const std::string& host, uint16_t port);
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  ErrorCode create_topic(const std::string& topic, uint32_t partitions);
  int32_t partition_count(const std::string& topic);  // -1 if unknown

  // Sends one pre-encoded batch (see append_record) and returns its base offset.
  uint64_t produce(const std::string& topic, uint32_t partition, const std::vector<uint8_t>& batch,
                   uint32_t count);
  std::vector<FetchResponse> fetch(const std::vector<FetchRequest>& reqs, uint32_t max_wait_ms);
  std::pair<uint64_t, uint64_t> list_offsets(const std::string& topic, uint32_t partition);

  ErrorCode commit_offset(const std::string& group, const std::string& member, uint32_t generation,
                          const std::string& topic, uint32_t partition, uint64_t offset);
  std::optional<uint64_t> committed_offset(const std::string& group, const std::string& topic,
                                           uint32_t partition);
  JoinResult join_group(const std::string& group, const std::string& member,
                        const std::vector<std::string>& topics, uint32_t session_timeout_ms);
  ErrorCode heartbeat(const std::string& group, const std::string& member, uint32_t generation);
  ErrorCode leave_group(const std::string& group, const std::string& member);

 private:
  // Sends a request and returns the response error; the body stays in resp_.
  // `tail` bytes are appended to the body on the wire without being copied.
  ErrorCode call(ApiKey api, const Writer& body, Reader** out, const uint8_t* tail = nullptr,
                 size_t tail_len = 0);

  int fd_ = -1;
  uint32_t next_corr_ = 1;
  std::vector<uint8_t> req_;
  std::vector<uint8_t> resp_;
  std::optional<Reader> reader_;
};

}  // namespace mk
