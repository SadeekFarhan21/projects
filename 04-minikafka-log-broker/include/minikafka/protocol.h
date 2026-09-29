// Wire protocol. Every message is a length-prefixed frame over TCP:
//
//   request:  u32 frame_len | u16 api_key | u32 correlation_id | body...
//   response: u32 frame_len | u32 correlation_id | u16 error_code | body...
//
// frame_len counts the bytes after itself. Integers are little-endian, strings
// are u32 length + bytes. Bodies per API are documented next to each enum value
// and in DESIGN.md.
#pragma once

#include <cstdint>
#include <string>

namespace mk {

enum class ApiKey : uint16_t {
  CreateTopic = 1,   // str topic, u32 partitions                       -> (none)
  Metadata = 2,      // str topic                                       -> u32 partitions
  Produce = 3,       // str topic, u32 partition, u32 count, blob recs  -> u64 base_offset
  Fetch = 4,         // u32 max_wait_ms, u32 n, n x (str topic, u32 partition, u64 offset, u32 max_bytes)
                     //   -> u32 n, n x (str topic, u32 partition, u16 err, u64 end, u64 start, blob recs)
  ListOffsets = 5,   // str topic, u32 partition                        -> u64 log_start, u64 log_end
  OffsetCommit = 6,  // str group, str member, u32 generation, str topic, u32 partition, u64 offset
  OffsetFetch = 7,   // str group, str topic, u32 partition             -> u8 found, u64 offset
  JoinGroup = 8,     // str group, str member, u32 session_ms, u32 n, n x str topic
                     //   -> str member, u32 generation, u32 n, n x (str topic, u32 partition)
  Heartbeat = 9,     // str group, str member, u32 generation
  LeaveGroup = 10,   // str group, str member
};

enum class ErrorCode : uint16_t {
  None = 0,
  UnknownTopic = 1,
  OffsetOutOfRange = 2,
  CorruptMessage = 3,
  InvalidRequest = 4,
  TopicExists = 5,
  UnknownMember = 6,
  IllegalGeneration = 7,
  RebalanceInProgress = 8,
  MessageTooLarge = 9,
  InternalError = 10,
  UnknownApi = 11,
};

const char* error_name(ErrorCode e);

constexpr uint32_t kMaxFrameSize = 64u << 20;  // 64 MiB

}  // namespace mk
