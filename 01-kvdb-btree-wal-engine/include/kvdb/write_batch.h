// WriteBatch: an ordered list of puts/deletes applied atomically. Its byte
// encoding is also the WAL payload, so logging a batch is a single copy.
//
//   u32 count, then per op: u8 kind, u16 klen, u16 vlen, key, value
#pragma once

#include <string>
#include <string_view>

#include "kvdb/common.h"

namespace kvdb {

class WriteBatch {
 public:
  enum Kind : uint8_t { kPut = 1, kDel = 2 };

  WriteBatch() { clear(); }
  void put(std::string_view k, std::string_view v) { add(kPut, k, v); }
  void del(std::string_view k) { add(kDel, k, {}); }
  void clear() {
    rep_.assign(4, '\0');
    count_ = 0;
  }
  uint32_t count() const { return count_; }
  std::string_view rep() const { return rep_; }

  // Decodes an encoding (from a batch or a WAL record). fn(kind, key, value).
  // Returns false if the encoding is malformed.
  template <typename Fn>
  static bool iterate(std::string_view rep, Fn&& fn) {
    if (rep.size() < 4) return false;
    const uint32_t n = load<uint32_t>(rep.data());
    size_t off = 4;
    for (uint32_t i = 0; i < n; ++i) {
      if (off + 5 > rep.size()) return false;
      const auto kind = static_cast<Kind>(rep[off]);
      const uint16_t kl = load<uint16_t>(rep.data() + off + 1);
      const uint16_t vl = load<uint16_t>(rep.data() + off + 3);
      off += 5;
      if (off + kl + vl > rep.size()) return false;
      if (kind != kPut && kind != kDel) return false;
      fn(kind, rep.substr(off, kl), rep.substr(off + kl, vl));
      off += kl + vl;
    }
    return off == rep.size();
  }

 private:
  void add(Kind kind, std::string_view k, std::string_view v) {
    if (k.empty() || k.size() > kMaxKeySize) throw std::invalid_argument("key size must be 1..128 bytes");
    if (v.size() > kMaxValueSize) throw std::invalid_argument("value larger than 512 bytes");
    char h[5];
    h[0] = static_cast<char>(kind);
    store<uint16_t>(h + 1, static_cast<uint16_t>(k.size()));
    store<uint16_t>(h + 3, static_cast<uint16_t>(v.size()));
    rep_.append(h, 5);
    rep_.append(k);
    rep_.append(v);
    store<uint32_t>(rep_.data(), ++count_);
  }

  std::string rep_;
  uint32_t count_ = 0;
};

}  // namespace kvdb
