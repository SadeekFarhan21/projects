// Test-only encoder: the inverse of the decoders, used to build hand-made
// packets and randomized round trips. Not linked into the tools.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "mdp/bytes.hpp"
#include "mdp/iex_messages.hpp"
#include "mdp/iextp.hpp"

namespace mdp::test {

using Bytes = std::vector<uint8_t>;

class Buf {
 public:
  Buf& u8(uint8_t v) {
    b.push_back(v);
    return *this;
  }
  template <class T>
  Buf& le(T v) {
    size_t n = b.size();
    b.resize(n + sizeof(T));
    store_le<T>(b.data() + n, v);
    return *this;
  }
  template <class T>
  Buf& be(T v) {
    size_t n = b.size();
    b.resize(n + sizeof(T));
    store_be<T>(b.data() + n, v);
    return *this;
  }
  template <class T>
  Buf& ord(T v, bool swap) {
    return swap ? be(v) : le(v);
  }
  Buf& bytes(const uint8_t* p, size_t n) {
    b.insert(b.end(), p, p + n);
    return *this;
  }
  Buf& bytes(const Bytes& v) { return bytes(v.data(), v.size()); }
  Buf& pad4() {
    while (b.size() % 4) b.push_back(0);
    return *this;
  }
  Bytes b;
};

inline Bytes hex(const std::string& s) {
  Bytes out;
  int hi = -1;
  for (char c : s) {
    int v;
    if (c >= '0' && c <= '9') v = c - '0';
    else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
    else continue;
    if (hi < 0) hi = v;
    else {
      out.push_back(static_cast<uint8_t>(hi << 4 | v));
      hi = -1;
    }
  }
  return out;
}

inline Bytes encode(const SystemEvent& e) { return Buf().u8('S').u8(e.event).le(e.ts).b; }
inline Bytes encode(const SecurityDirectory& d) {
  return Buf().u8('D').u8(d.flags).le(d.ts).le(d.symbol).le(d.round_lot).le(d.adjusted_poc_price).u8(d.luld_tier).b;
}
inline Bytes encode(const TradingStatus& s) {
  return Buf().u8('H').u8(s.status).le(s.ts).le(s.symbol).le(s.reason).b;
}
inline Bytes encode(const SecurityStatus& s) {
  Buf b;
  b.u8(s.type).u8(s.status).le(s.ts).le(s.symbol);
  if (s.type == msg::ShortSalePriceTest) b.u8(s.detail);
  return b.b;
}
inline Bytes encode(const QuoteUpdate& q) {
  return Buf()
      .u8('Q')
      .u8(q.flags)
      .le(q.ts)
      .le(q.symbol)
      .le(q.bid_size)
      .le(q.bid_price)
      .le(q.ask_price)
      .le(q.ask_size)
      .b;
}
inline Bytes encode(const Trade& t) {
  return Buf().u8(t.type).u8(t.flags).le(t.ts).le(t.symbol).le(t.size).le(t.price).le(t.trade_id).b;
}
inline Bytes encode(const OfficialPrice& p) {
  return Buf().u8('X').u8(p.price_type).le(p.ts).le(p.symbol).le(p.price).b;
}
inline Bytes encode(const PriceLevelUpdate& u) {
  return Buf().u8(u.side).u8(u.flags).le(u.ts).le(u.symbol).le(u.size).le(u.price).b;
}
inline Bytes encode(const AuctionInfo& a) {
  return Buf()
      .u8('A')
      .u8(a.auction_type)
      .le(a.ts)
      .le(a.symbol)
      .le(a.paired_shares)
      .le(a.reference_price)
      .le(a.indicative_clearing_price)
      .le(a.imbalance_shares)
      .u8(a.imbalance_side)
      .u8(a.extension_number)
      .le(a.scheduled_auction_time)
      .le(a.auction_book_clearing_price)
      .le(a.collar_reference_price)
      .le(a.lower_auction_collar)
      .le(a.upper_auction_collar)
      .b;
}

struct SegmentSpec {
  uint16_t protocol_id = kProtoDeep10;
  uint32_t channel_id = 1;
  uint32_t session_id = 0x42870000;
  int64_t stream_offset = 0;
  int64_t first_seq = 1;
  int64_t send_time = 0;
};

inline Bytes build_segment(const SegmentSpec& s, const std::vector<Bytes>& msgs) {
  Buf payload;
  for (const auto& m : msgs) payload.le<uint16_t>(static_cast<uint16_t>(m.size())).bytes(m);
  Buf b;
  b.u8(1).u8(0).le(s.protocol_id).le(s.channel_id).le(s.session_id);
  b.le<uint16_t>(static_cast<uint16_t>(payload.b.size())).le<uint16_t>(static_cast<uint16_t>(msgs.size()));
  b.le(s.stream_offset).le(s.first_seq).le(s.send_time);
  b.bytes(payload.b);
  return b.b;
}

struct FrameSpec {
  uint32_t src_ip = 0x17e29b83;  // 23.226.155.131
  uint32_t dst_ip = 0xe9d71504;  // 233.215.21.4
  uint16_t src_port = 10378;
  uint16_t dst_port = 10378;
  int vlan_tags = 0;
  uint16_t frag = 0x4000;  // DF
  uint8_t ip_proto = 17;
};

inline Bytes udp_ipv4(const Bytes& payload, const FrameSpec& f) {
  Buf b;
  uint16_t total = static_cast<uint16_t>(20 + 8 + payload.size());
  b.u8(0x45).u8(0).be<uint16_t>(total).be<uint16_t>(0x1234).be<uint16_t>(f.frag).u8(64).u8(f.ip_proto);
  b.be<uint16_t>(0).be<uint32_t>(f.src_ip).be<uint32_t>(f.dst_ip);
  b.be<uint16_t>(f.src_port).be<uint16_t>(f.dst_port).be<uint16_t>(static_cast<uint16_t>(8 + payload.size()));
  b.be<uint16_t>(0).bytes(payload);
  return b.b;
}

inline Bytes ethernet(const Bytes& ip, const FrameSpec& f) {
  Buf b;
  const uint8_t dst[6] = {0x01, 0x00, 0x5e, 0x57, 0x15, 0x04};
  const uint8_t src[6] = {0xa0, 0x88, 0xc2, 0xac, 0xb9, 0x05};
  b.bytes(dst, 6).bytes(src, 6);
  for (int i = 0; i < f.vlan_tags; ++i) b.be<uint16_t>(i == 0 && f.vlan_tags == 2 ? 0x88a8 : 0x8100).be<uint16_t>(100 + i);
  b.be<uint16_t>(0x0800).bytes(ip);
  while (b.b.size() < 60) b.u8(0);  // Ethernet minimum frame padding
  return b.b;
}

inline Bytes frame(const Bytes& segment, const FrameSpec& f = {}) { return ethernet(udp_ipv4(segment, f), f); }

struct TimedFrame {
  int64_t ts_ns;
  Bytes data;
};

inline Bytes write_pcap(const std::vector<TimedFrame>& frames, bool nano, bool swap, uint32_t link = 1) {
  Buf b;
  uint32_t magic = nano ? 0xa1b23c4d : 0xa1b2c3d4;
  b.ord<uint32_t>(magic, swap).ord<uint16_t>(2, swap).ord<uint16_t>(4, swap).ord<int32_t>(0, swap);
  b.ord<uint32_t>(0, swap).ord<uint32_t>(262144, swap).ord<uint32_t>(link, swap);
  for (const auto& f : frames) {
    int64_t sec = f.ts_ns / 1'000'000'000, frac = f.ts_ns % 1'000'000'000;
    if (!nano) frac /= 1000;
    b.ord<uint32_t>(static_cast<uint32_t>(sec), swap).ord<uint32_t>(static_cast<uint32_t>(frac), swap);
    b.ord<uint32_t>(static_cast<uint32_t>(f.data.size()), swap)
        .ord<uint32_t>(static_cast<uint32_t>(f.data.size()), swap);
    b.bytes(f.data);
  }
  return b.b;
}

struct PcapNgOptions {
  bool swap = false;
  int tsresol = 6;  // decimal exponent; negative means binary 2^-(-tsresol)
  bool extra_blocks = true;  // insert a Name Resolution Block and a custom block to be skipped
  int sections = 1;
  bool use_spb = false;
};

inline void ng_block(Buf& out, uint32_t type, const Bytes& body, bool swap) {
  Buf blk;
  uint32_t len = static_cast<uint32_t>(12 + ((body.size() + 3) & ~size_t{3}));
  blk.ord<uint32_t>(type, swap).ord<uint32_t>(len, swap).bytes(body).pad4().ord<uint32_t>(len, swap);
  out.bytes(blk.b);
}

inline Bytes write_pcapng(const std::vector<TimedFrame>& frames, const PcapNgOptions& o) {
  Buf out;
  const bool s = o.swap;
  size_t per_section = (frames.size() + o.sections - 1) / std::max(1, o.sections);
  size_t idx = 0;
  for (int sec = 0; sec < o.sections; ++sec) {
    // SHB with a comment option
    Buf shb;
    shb.ord<uint32_t>(0x1a2b3c4d, s).ord<uint16_t>(1, s).ord<uint16_t>(0, s).ord<int64_t>(-1, s);
    const char* comment = "mdp test";
    shb.ord<uint16_t>(1, s).ord<uint16_t>(8, s).bytes(reinterpret_cast<const uint8_t*>(comment), 8);
    shb.ord<uint16_t>(0, s).ord<uint16_t>(0, s);
    ng_block(out, 0x0a0d0d0a, shb.b, s);
    // IDB with if_tsresol
    Buf idb;
    idb.ord<uint16_t>(1, s).ord<uint16_t>(0, s).ord<uint32_t>(0, s);
    uint8_t res = o.tsresol >= 0 ? static_cast<uint8_t>(o.tsresol) : static_cast<uint8_t>(0x80 | -o.tsresol);
    idb.ord<uint16_t>(9, s).ord<uint16_t>(1, s).u8(res).u8(0).u8(0).u8(0);
    idb.ord<uint16_t>(0, s).ord<uint16_t>(0, s);
    ng_block(out, 1, idb.b, s);
    if (o.extra_blocks) {
      Bytes nrb(16, 0);
      ng_block(out, 4, nrb, s);
      Bytes custom(12, 0xab);
      ng_block(out, 0x00000BAD, custom, s);
    }
    for (size_t k = 0; k < per_section && idx < frames.size(); ++k, ++idx) {
      const auto& f = frames[idx];
      if (o.use_spb) {
        Buf spb;
        spb.ord<uint32_t>(static_cast<uint32_t>(f.data.size()), s).bytes(f.data);
        ng_block(out, 3, spb.b, s);
        continue;
      }
      uint64_t ticks;
      if (o.tsresol >= 0) {
        int64_t div = 1;
        for (int i = 0; i < 9 - o.tsresol; ++i) div *= 10;
        ticks = static_cast<uint64_t>(f.ts_ns / div);
      } else {
        ticks = static_cast<uint64_t>((static_cast<__int128>(f.ts_ns) << (-o.tsresol)) / 1'000'000'000);
      }
      Buf epb;
      epb.ord<uint32_t>(0, s)
          .ord<uint32_t>(static_cast<uint32_t>(ticks >> 32), s)
          .ord<uint32_t>(static_cast<uint32_t>(ticks), s);
      epb.ord<uint32_t>(static_cast<uint32_t>(f.data.size()), s)
          .ord<uint32_t>(static_cast<uint32_t>(f.data.size()), s);
      epb.bytes(f.data).pad4();
      ng_block(out, 6, epb.b, s);
    }
  }
  return out.b;
}

}  // namespace mdp::test
