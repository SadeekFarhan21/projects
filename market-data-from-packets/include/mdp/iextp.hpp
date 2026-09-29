// IEX Transport Protocol v1 (IEX-TP) segment header and message block framing.
#pragma once

#include <cstdint>

#include "mdp/bytes.hpp"

namespace mdp {

constexpr uint32_t kSegmentHeaderSize = 40;
constexpr uint16_t kProtoTops15 = 0x8002;
constexpr uint16_t kProtoTops16 = 0x8003;
constexpr uint16_t kProtoDeep10 = 0x8004;
constexpr uint16_t kProtoDeepPlus = 0x8005;

struct SegmentHeader {
  uint8_t version = 0;
  uint16_t protocol_id = 0;
  uint32_t channel_id = 0;
  uint32_t session_id = 0;
  uint16_t payload_length = 0;
  uint16_t message_count = 0;
  int64_t stream_offset = 0;
  int64_t first_seq = 0;
  int64_t send_time = 0;  // ns since epoch
  bool is_heartbeat() const { return message_count == 0 && payload_length == 0; }
};

enum class SegStatus : uint8_t { Ok = 0, TooShort, BadVersion, LengthMismatch };

inline const char* to_string(SegStatus s) {
  switch (s) {
    case SegStatus::Ok: return "ok";
    case SegStatus::TooShort: return "too_short";
    case SegStatus::BadVersion: return "bad_version";
    case SegStatus::LengthMismatch: return "length_mismatch";
  }
  return "?";
}

inline SegStatus parse_segment(const uint8_t* p, uint32_t len, SegmentHeader& h, const uint8_t*& payload) {
  if (len < kSegmentHeaderSize) return SegStatus::TooShort;
  h.version = p[0];
  if (h.version != 1) return SegStatus::BadVersion;
  h.protocol_id = load_le<uint16_t>(p + 2);
  h.channel_id = load_le<uint32_t>(p + 4);
  h.session_id = load_le<uint32_t>(p + 8);
  h.payload_length = load_le<uint16_t>(p + 12);
  h.message_count = load_le<uint16_t>(p + 14);
  h.stream_offset = load_le<int64_t>(p + 16);
  h.first_seq = load_le<int64_t>(p + 24);
  h.send_time = load_le<int64_t>(p + 32);
  if (static_cast<uint32_t>(h.payload_length) + kSegmentHeaderSize != len) return SegStatus::LengthMismatch;
  payload = p + kSegmentHeaderSize;
  return SegStatus::Ok;
}

// Walks the Message Blocks of a segment payload. Each block is a u16 length
// followed by that many bytes of message data.
class MessageIterator {
 public:
  MessageIterator(const uint8_t* payload, uint16_t payload_len, uint16_t count)
      : p_(payload), end_(payload + payload_len), remaining_(count) {}

  // Returns false when all `count` messages are consumed or on a framing error
  // (check framing_error()).
  bool next(const uint8_t*& msg, uint16_t& len) {
    if (remaining_ == 0) return false;
    if (end_ - p_ < 2) {
      error_ = true;
      return false;
    }
    len = load_le<uint16_t>(p_);
    if (end_ - p_ - 2 < len) {
      error_ = true;
      return false;
    }
    msg = p_ + 2;
    p_ += 2 + len;
    --remaining_;
    return true;
  }
  bool framing_error() const { return error_; }
  // After consuming all messages, the payload should be fully used.
  bool trailing_bytes() const { return remaining_ == 0 && p_ != end_; }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
  uint16_t remaining_;
  bool error_ = false;
};

}  // namespace mdp
