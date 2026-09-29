#include "minikafka/record.h"

#include <stdexcept>

#include "minikafka/bytes.h"
#include "minikafka/crc32c.h"

namespace mk {

void append_record(std::vector<uint8_t>& out, int64_t timestamp_ms, std::string_view key,
                   std::string_view value) {
  const size_t payload = kPayloadFixedSize + key.size() + value.size();
  if (payload > kMaxPayloadSize) throw std::invalid_argument("record too large");
  const size_t start = out.size();
  out.resize(start + kRecordHeaderSize + payload);
  uint8_t* h = out.data() + start;
  store_u64(h, 0);
  store_u32(h + 8, static_cast<uint32_t>(payload));
  uint8_t* body = h + kRecordHeaderSize;
  std::memcpy(body, &timestamp_ms, 8);
  store_u32(body + 8, static_cast<uint32_t>(key.size()));
  if (!key.empty()) std::memcpy(body + 12, key.data(), key.size());
  if (!value.empty()) std::memcpy(body + 12 + key.size(), value.data(), value.size());
  store_u32(h + 12, crc32c(body, payload));
}

int64_t validate_batch(const uint8_t* p, size_t n, std::string* error) {
  size_t pos = 0;
  int64_t count = 0;
  while (pos < n) {
    if (n - pos < kRecordHeaderSize) {
      if (error) *error = "truncated record header";
      return -1;
    }
    const uint32_t len = load_u32(p + pos + 8);
    if (len < kPayloadFixedSize || len > kMaxPayloadSize || len > n - pos - kRecordHeaderSize) {
      if (error) *error = "bad payload length";
      return -1;
    }
    const uint8_t* body = p + pos + kRecordHeaderSize;
    if (load_u32(body + 8) > len - kPayloadFixedSize) {
      if (error) *error = "bad key length";
      return -1;
    }
    if (crc32c(body, len) != load_u32(p + pos + 12)) {
      if (error) *error = "crc mismatch";
      return -1;
    }
    pos += kRecordHeaderSize + len;
    ++count;
  }
  return count;
}

std::vector<RecordView> parse_records(const uint8_t* p, size_t n) {
  std::vector<RecordView> out;
  size_t pos = 0;
  while (n - pos >= kRecordHeaderSize) {
    const uint32_t len = load_u32(p + pos + 8);
    if (len < kPayloadFixedSize || len > n - pos - kRecordHeaderSize) break;
    const uint8_t* body = p + pos + kRecordHeaderSize;
    if (crc32c(body, len) != load_u32(p + pos + 12)) throw std::runtime_error("crc mismatch in fetched record");
    const uint32_t klen = load_u32(body + 8);
    if (klen > len - kPayloadFixedSize) throw std::runtime_error("bad key length in fetched record");
    RecordView r;
    r.offset = load_u64(p + pos);
    std::memcpy(&r.timestamp_ms, body, 8);
    r.key = std::string_view(reinterpret_cast<const char*>(body + 12), klen);
    r.value = std::string_view(reinterpret_cast<const char*>(body + 12 + klen), len - kPayloadFixedSize - klen);
    out.push_back(r);
    pos += kRecordHeaderSize + len;
  }
  return out;
}

std::vector<Record> decode_records(const uint8_t* p, size_t n) {
  std::vector<Record> out;
  for (const auto& v : parse_records(p, n)) {
    out.push_back(Record{v.offset, v.timestamp_ms, std::string(v.key), std::string(v.value)});
  }
  return out;
}

}  // namespace mk
