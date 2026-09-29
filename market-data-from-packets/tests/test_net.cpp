#include <gtest/gtest.h>

#include "mdp/net.hpp"
#include "support/encoder.hpp"

using namespace mdp;
using namespace mdp::test;

namespace {
Bytes payload() { return Bytes{1, 2, 3, 4, 5, 6, 7}; }
}  // namespace

TEST(Net, EthernetIpv4Udp) {
  FrameSpec f;
  Bytes fr = frame(payload(), f);
  EXPECT_EQ(fr.size(), 60u);  // padded to Ethernet minimum; IP total length must win
  UdpDatagram d;
  ASSERT_EQ(parse_udp(kLinkEthernet, fr.data(), static_cast<uint32_t>(fr.size()), d), NetStatus::Ok);
  EXPECT_EQ(Bytes(d.payload, d.payload + d.len), payload());
  EXPECT_EQ(d.dst_port, 10378);
  EXPECT_EQ(d.dst_ip, 0xe9d71504u);
  EXPECT_EQ(d.src_ip, 0x17e29b83u);
}

TEST(Net, VlanTags) {
  for (int tags : {1, 2}) {
    FrameSpec f;
    f.vlan_tags = tags;
    Bytes fr = frame(payload(), f);
    UdpDatagram d;
    ASSERT_EQ(parse_udp(kLinkEthernet, fr.data(), static_cast<uint32_t>(fr.size()), d), NetStatus::Ok) << tags;
    EXPECT_EQ(d.len, 7u);
  }
}

TEST(Net, RawAndCookedLinks) {
  FrameSpec f;
  Bytes ip = udp_ipv4(payload(), f);
  UdpDatagram d;
  EXPECT_EQ(parse_udp(kLinkRaw, ip.data(), static_cast<uint32_t>(ip.size()), d), NetStatus::Ok);
  EXPECT_EQ(parse_udp(kLinkIpv4, ip.data(), static_cast<uint32_t>(ip.size()), d), NetStatus::Ok);
  Buf sll;
  sll.be<uint16_t>(0).be<uint16_t>(1).be<uint16_t>(6);
  for (int i = 0; i < 8; ++i) sll.u8(0);
  sll.be<uint16_t>(0x0800).bytes(ip);
  EXPECT_EQ(parse_udp(kLinkLinuxSll, sll.b.data(), static_cast<uint32_t>(sll.b.size()), d), NetStatus::Ok);
  EXPECT_EQ(d.len, 7u);
  Buf sll2;
  sll2.be<uint16_t>(0x0800).be<uint16_t>(0);
  for (int i = 0; i < 16; ++i) sll2.u8(0);
  sll2.bytes(ip);
  EXPECT_EQ(parse_udp(kLinkLinuxSll2, sll2.b.data(), static_cast<uint32_t>(sll2.b.size()), d), NetStatus::Ok);
  EXPECT_EQ(parse_udp(999, ip.data(), static_cast<uint32_t>(ip.size()), d), NetStatus::UnsupportedLink);
}

TEST(Net, RejectsFragmentsNonUdpAndTruncation) {
  UdpDatagram d;
  FrameSpec frag;
  frag.frag = 0x2000;  // more fragments
  Bytes a = frame(payload(), frag);
  EXPECT_EQ(parse_udp(kLinkEthernet, a.data(), static_cast<uint32_t>(a.size()), d), NetStatus::Fragment);
  FrameSpec off;
  off.frag = 0x0010;  // nonzero fragment offset
  Bytes b = frame(payload(), off);
  EXPECT_EQ(parse_udp(kLinkEthernet, b.data(), static_cast<uint32_t>(b.size()), d), NetStatus::Fragment);
  FrameSpec tcp;
  tcp.ip_proto = 6;
  Bytes c = frame(payload(), tcp);
  EXPECT_EQ(parse_udp(kLinkEthernet, c.data(), static_cast<uint32_t>(c.size()), d), NetStatus::NotUdp);
  Bytes big(200, 0xaa);
  Bytes e = frame(big, FrameSpec{});
  EXPECT_EQ(parse_udp(kLinkEthernet, e.data(), 100, d), NetStatus::Truncated);
  EXPECT_EQ(parse_udp(kLinkEthernet, e.data(), 10, d), NetStatus::Truncated);
  Bytes arp = e;
  arp[12] = 0x08;
  arp[13] = 0x06;
  EXPECT_EQ(parse_udp(kLinkEthernet, arp.data(), static_cast<uint32_t>(arp.size()), d), NetStatus::NotIpv4);
  // UDP length field larger than the IP payload.
  Bytes f = frame(payload(), FrameSpec{});
  f[14 + 20 + 4] = 0x01;
  EXPECT_EQ(parse_udp(kLinkEthernet, f.data(), static_cast<uint32_t>(f.size()), d), NetStatus::BadHeader);
}
