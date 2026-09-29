// Record format shared by the produce request, the on-disk log and the fetch response.
//
// One record on the wire and on disk:
//
//   offset       u64   assigned by the broker (producers send 0)
//   payload_len  u32   number of bytes after the crc field
//   crc          u32   CRC-32C over the payload
//   payload:
//     timestamp  i64   milliseconds since epoch (set by the producer)
//     key_len    u32
//     key        key_len bytes
//     value      payload_len - 12 - key_len bytes
//
// Because the three places use the same bytes, the broker can write a produce
// batch to disk after patching offsets in place, and serve a fetch by copying
// a byte range out of a segment file without re-encoding anything.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mk {

constexpr size_t kRecordHeaderSize = 16;        // offset + payload_len + crc
constexpr size_t kPayloadFixedSize = 12;        // timestamp + key_len
constexpr uint32_t kMaxPayloadSize = 16u << 20; // 16 MiB per record

struct Record {
  uint64_t offset = 0;
  int64_t timestamp_ms = 0;
  std::string key;
  std::string value;
};

struct RecordView {
  uint64_t offset;
  int64_t timestamp_ms;
  std::string_view key;
  std::string_view value;
};

// Appends one encoded record (offset 0) to `out`.
void append_record(std::vector<uint8_t>& out, int64_t timestamp_ms, std::string_view key,
                   std::string_view value);

// Checks that [p, p+n) is a sequence of complete, well-formed records with valid
// CRCs. Returns the record count, or -1 and fills `error`.
int64_t validate_batch(const uint8_t* p, size_t n, std::string* error);

// Parses records from a fetch response buffer. Stops at the first incomplete
// record (never happens for well-formed responses). Throws on CRC mismatch.
std::vector<RecordView> parse_records(const uint8_t* p, size_t n);
std::vector<Record> decode_records(const uint8_t* p, size_t n);

}  // namespace mk
