// Link layer, IPv4 and UDP parsing. Header-only because it sits on the hot path.
#pragma once

#include <cstdint>

#include "mdp/bytes.hpp"
#include "mdp/pcap.hpp"

namespace mdp {

enum class NetStatus : uint8_t {
  Ok = 0,
  UnsupportedLink,
  NotIpv4,
  NotUdp,
  Fragment,
  Truncated,
  BadHeader,
};

inline const char* to_string(NetStatus s) {
  switch (s) {
    case NetStatus::Ok: return "ok";
    case NetStatus::UnsupportedLink: return "unsupported_link";
    case NetStatus::NotIpv4: return "not_ipv4";
    case NetStatus::NotUdp: return "not_udp";
    case NetStatus::Fragment: return "fragment";
    case NetStatus::Truncated: return "truncated";
    case NetStatus::BadHeader: return "bad_header";
  }
  return "?";
}

struct UdpDatagram {
  uint32_t src_ip = 0;  // host order
  uint32_t dst_ip = 0;
  uint16_t src_port = 0;
  uint16_t dst_port = 0;
  const uint8_t* payload = nullptr;
  uint32_t len = 0;
};

// Parses an IPv4 packet starting at p. `avail` is the number of captured bytes.
inline NetStatus parse_ipv4_udp(const uint8_t* p, uint32_t avail, UdpDatagram& out) {
  if (avail < 20) return NetStatus::Truncated;
  if ((p[0] >> 4) != 4) return NetStatus::NotIpv4;
  uint32_t ihl = (p[0] & 0x0f) * 4u;
  if (ihl < 20) return NetStatus::BadHeader;
  uint32_t total = load_be<uint16_t>(p + 2);
  if (total < ihl) return NetStatus::BadHeader;
  if (total > avail) return NetStatus::Truncated;
  uint16_t frag = load_be<uint16_t>(p + 6);
  if ((frag & 0x2000) || (frag & 0x1fff)) return NetStatus::Fragment;  // MF set or nonzero offset
  if (p[9] != 17) return NetStatus::NotUdp;
  const uint8_t* u = p + ihl;
  uint32_t ip_payload = total - ihl;
  if (ip_payload < 8) return NetStatus::Truncated;
  uint32_t ulen = load_be<uint16_t>(u + 4);
  if (ulen < 8 || ulen > ip_payload) return NetStatus::BadHeader;
  out.src_ip = load_be<uint32_t>(p + 12);
  out.dst_ip = load_be<uint32_t>(p + 16);
  out.src_port = load_be<uint16_t>(u);
  out.dst_port = load_be<uint16_t>(u + 2);
  out.payload = u + 8;
  out.len = ulen - 8;
  return NetStatus::Ok;
}

// Strips the link layer (Ethernet with up to two VLAN tags, raw IP, Linux
// cooked v1 and v2) and parses IPv4 and UDP.
inline NetStatus parse_udp(uint16_t link_type, const uint8_t* p, uint32_t caplen, UdpDatagram& out) {
  uint16_t ethertype = 0;
  uint32_t off = 0;
  switch (link_type) {
    case kLinkEthernet: {
      if (caplen < 14) return NetStatus::Truncated;
      ethertype = load_be<uint16_t>(p + 12);
      off = 14;
      for (int tags = 0; tags < 2 && (ethertype == 0x8100 || ethertype == 0x88a8); ++tags) {
        if (caplen < off + 4) return NetStatus::Truncated;
        ethertype = load_be<uint16_t>(p + off + 2);
        off += 4;
      }
      break;
    }
    case kLinkRaw:
    case kLinkIpv4:
      return parse_ipv4_udp(p, caplen, out);
    case kLinkLinuxSll:
      if (caplen < 16) return NetStatus::Truncated;
      ethertype = load_be<uint16_t>(p + 14);
      off = 16;
      break;
    case kLinkLinuxSll2:
      if (caplen < 20) return NetStatus::Truncated;
      ethertype = load_be<uint16_t>(p);
      off = 20;
      break;
    default:
      return NetStatus::UnsupportedLink;
  }
  if (ethertype != 0x0800) return NetStatus::NotIpv4;
  return parse_ipv4_udp(p + off, caplen - off, out);
}

}  // namespace mdp
