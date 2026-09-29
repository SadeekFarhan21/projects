#include <gtest/gtest.h>
#include <unistd.h>
#include <zlib.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "mdp/pcap.hpp"
#include "mdp/pcap_writer.hpp"
#include "support/encoder.hpp"

using namespace mdp;
using namespace mdp::test;

namespace {

std::vector<TimedFrame> sample_frames(int n, int64_t t0 = 1789729540032514123) {
  std::vector<TimedFrame> v;
  for (int i = 0; i < n; ++i) {
    Bytes payload(static_cast<size_t>(10 + i * 7), static_cast<uint8_t>(i));
    v.push_back({t0 + i * 1'000'003, payload});
  }
  return v;
}

std::vector<Packet> read_all(const Bytes& file, std::vector<Bytes>* datas = nullptr, CaptureReader** out = nullptr) {
  static std::unique_ptr<CaptureReader> keep;
  keep = std::make_unique<CaptureReader>(std::make_unique<MemorySource>(file.data(), file.size()));
  std::vector<Packet> out_pk;
  Packet p;
  while (keep->next(p)) {
    out_pk.push_back(p);
    if (datas) datas->emplace_back(p.data, p.data + p.caplen);
  }
  if (out) *out = keep.get();
  return out_pk;
}

std::string tmp_path(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("mdp_test_" + std::to_string(::getpid()) + "_" + name)).string();
}

}  // namespace

class PcapClassic : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(PcapClassic, ReadsAllVariants) {
  auto [nano, swap] = GetParam();
  auto frames = sample_frames(5);
  Bytes file = write_pcap(frames, nano, swap);
  std::vector<Bytes> datas;
  CaptureReader* r = nullptr;
  auto pk = read_all(file, &datas, &r);
  EXPECT_EQ(r->format(), CaptureFormat::Pcap);
  ASSERT_EQ(pk.size(), frames.size());
  for (size_t i = 0; i < pk.size(); ++i) {
    int64_t expect = nano ? frames[i].ts_ns : frames[i].ts_ns / 1000 * 1000;
    EXPECT_EQ(pk[i].ts_ns, expect);
    EXPECT_EQ(datas[i], frames[i].data);
    EXPECT_EQ(pk[i].link_type, kLinkEthernet);
  }
  EXPECT_FALSE(r->truncated());
}

INSTANTIATE_TEST_SUITE_P(AllMagics, PcapClassic,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

class PcapNg : public ::testing::TestWithParam<std::tuple<bool, int, int>> {};

TEST_P(PcapNg, ReadsSectionsResolutionsAndByteOrders) {
  auto [swap, tsresol, sections] = GetParam();
  auto frames = sample_frames(7);
  PcapNgOptions o;
  o.swap = swap;
  o.tsresol = tsresol;
  o.sections = sections;
  Bytes file = write_pcapng(frames, o);
  std::vector<Bytes> datas;
  CaptureReader* r = nullptr;
  auto pk = read_all(file, &datas, &r);
  EXPECT_EQ(r->format(), CaptureFormat::PcapNg);
  ASSERT_EQ(pk.size(), frames.size());
  for (size_t i = 0; i < pk.size(); ++i) {
    int64_t expect;
    if (tsresol >= 0) {
      int64_t unit = 1;
      for (int k = 0; k < 9 - tsresol; ++k) unit *= 10;
      expect = frames[i].ts_ns / unit * unit;
      EXPECT_EQ(pk[i].ts_ns, expect);
    } else {
      // binary resolution: within one tick
      double tick = 1e9 / static_cast<double>(int64_t{1} << -tsresol);
      EXPECT_NEAR(static_cast<double>(pk[i].ts_ns), static_cast<double>(frames[i].ts_ns), tick + 1);
    }
    EXPECT_EQ(datas[i], frames[i].data);
  }
  EXPECT_EQ(r->stats().sections, static_cast<uint64_t>(sections));
  EXPECT_EQ(r->stats().blocks_skipped, static_cast<uint64_t>(2 * sections));
}

INSTANTIATE_TEST_SUITE_P(Variants, PcapNg,
                         ::testing::Combine(::testing::Bool(), ::testing::Values(6, 9, 3, -20),
                                            ::testing::Values(1, 3)));

TEST(PcapNgBlocks, SimplePacketBlock) {
  auto frames = sample_frames(3);
  PcapNgOptions o;
  o.use_spb = true;
  Bytes file = write_pcapng(frames, o);
  std::vector<Bytes> datas;
  auto pk = read_all(file, &datas);
  ASSERT_EQ(pk.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(datas[i], frames[i].data);
    EXPECT_EQ(pk[i].ts_ns, 0);
  }
}

TEST(Capture, DetectsTruncatedTail) {
  Bytes file = write_pcap(sample_frames(4), true, false);
  file.resize(file.size() - 3);
  CaptureReader* r = nullptr;
  auto pk = read_all(file, nullptr, &r);
  EXPECT_EQ(pk.size(), 3u);
  EXPECT_TRUE(r->truncated());

  PcapNgOptions o;
  Bytes ng = write_pcapng(sample_frames(4), o);
  ng.resize(ng.size() - 5);
  pk = read_all(ng, nullptr, &r);
  EXPECT_EQ(pk.size(), 3u);
  EXPECT_TRUE(r->truncated());
}

TEST(Capture, RejectsUnknownMagicAndCorruptBlocks) {
  Bytes junk = {0xde, 0xad, 0xbe, 0xef, 0, 0, 0, 0};
  EXPECT_THROW(CaptureReader(std::make_unique<MemorySource>(junk.data(), junk.size())), CaptureError);

  PcapNgOptions o;
  o.extra_blocks = false;
  Bytes ng = write_pcapng(sample_frames(2), o);
  // Corrupt the trailing length of the last block.
  ng[ng.size() - 1] ^= 0xff;
  CaptureReader r(std::make_unique<MemorySource>(ng.data(), ng.size()));
  Packet p;
  EXPECT_TRUE(r.next(p));
  EXPECT_THROW(r.next(p), CaptureError);
}

TEST(Capture, ReadsGzipFileAndPlainFile) {
  auto frames = sample_frames(20);
  Bytes file = write_pcapng(frames, PcapNgOptions{});
  std::string gz = tmp_path("cap.pcapng.gz");
  std::string plain = tmp_path("cap.pcapng");
  {
    gzFile g = gzopen(gz.c_str(), "wb");
    ASSERT_NE(g, nullptr);
    gzwrite(g, file.data(), static_cast<unsigned>(file.size()));
    gzclose(g);
    std::ofstream f(plain, std::ios::binary);
    f.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
  }
  for (const auto& path : {gz, plain}) {
    CaptureReader r(open_source(path));
    Packet p;
    size_t n = 0;
    while (r.next(p)) {
      EXPECT_EQ(Bytes(p.data, p.data + p.caplen), frames[n].data);
      ++n;
    }
    EXPECT_EQ(n, frames.size());
    EXPECT_FALSE(r.truncated());
  }
  // Truncated gzip: drop the last third of the compressed file.
  {
    std::ifstream f(gz, std::ios::binary);
    std::string all((std::istreambuf_iterator<char>(f)), {});
    std::ofstream t(gz, std::ios::binary | std::ios::trunc);
    t.write(all.data(), static_cast<std::streamsize>(all.size() * 2 / 3));
  }
  CaptureReader r(open_source(gz));
  Packet p;
  size_t n = 0;
  while (r.next(p)) ++n;
  EXPECT_LT(n, frames.size());
  EXPECT_TRUE(r.truncated());
  std::remove(gz.c_str());
  std::remove(plain.c_str());
}

TEST(PcapWriter, RoundTripsThroughReader) {
  auto frames = sample_frames(6);
  std::string path = tmp_path("w.pcap");
  {
    PcapWriter w(path, kLinkEthernet, true);
    for (const auto& f : frames) w.write(f.ts_ns, f.data.data(), static_cast<uint32_t>(f.data.size()));
  }
  CaptureReader r(open_source(path));
  Packet p;
  size_t n = 0;
  while (r.next(p)) {
    EXPECT_EQ(p.ts_ns, frames[n].ts_ns);
    ++n;
  }
  EXPECT_EQ(n, frames.size());
  std::remove(path.c_str());
}
