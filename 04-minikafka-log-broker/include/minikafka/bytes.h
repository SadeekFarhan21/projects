// Little-endian binary encoding helpers used by the wire protocol and the log format.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mk {

// Both the on-disk format and the wire format are little-endian. Every target we
// care about (arm64, x86-64) is little-endian, so encoding is a plain memcpy.
static_assert(std::endian::native == std::endian::little, "minikafka assumes a little-endian host");

inline void store_u32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
inline void store_u64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }
inline uint32_t load_u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t load_u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

class Writer {
 public:
  std::vector<uint8_t> buf;

  void put(const void* p, size_t n) {
    auto* c = static_cast<const uint8_t*>(p);
    buf.insert(buf.end(), c, c + n);
  }
  void u8(uint8_t v) { buf.push_back(v); }
  void u16(uint16_t v) { put(&v, 2); }
  void u32(uint32_t v) { put(&v, 4); }
  void u64(uint64_t v) { put(&v, 8); }
  void i64(int64_t v) { put(&v, 8); }
  void str(std::string_view s) {
    u32(static_cast<uint32_t>(s.size()));
    put(s.data(), s.size());
  }
  // Length-prefixed opaque bytes.
  void blob(const uint8_t* p, size_t n) {
    u32(static_cast<uint32_t>(n));
    put(p, n);
  }
};

struct DecodeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Bounds-checked reader over a byte range. Throws DecodeError on truncation so a
// malformed request can never read past its buffer.
class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}

  uint8_t u8() { return get<uint8_t>(); }
  uint16_t u16() { return get<uint16_t>(); }
  uint32_t u32() { return get<uint32_t>(); }
  uint64_t u64() { return get<uint64_t>(); }
  int64_t i64() { return get<int64_t>(); }
  std::string str() {
    uint32_t n = u32();
    auto v = view(n);
    return std::string(v);
  }
  std::string_view view(size_t n) {
    need(n);
    std::string_view v(reinterpret_cast<const char*>(p_ + pos_), n);
    pos_ += n;
    return v;
  }
  size_t remaining() const { return n_ - pos_; }

 private:
  template <class T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  void need(size_t k) const {
    if (k > n_ - pos_) throw DecodeError("truncated message");
  }

  const uint8_t* p_;
  size_t n_;
  size_t pos_ = 0;
};

}  // namespace mk
