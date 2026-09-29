// Minimal classic pcap writer (nanosecond or microsecond magic). Used by the
// slicing tool to cut small fixtures out of full-day captures, and by tests.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "mdp/bytes.hpp"

namespace mdp {

class PcapWriter {
 public:
  PcapWriter(const std::string& path, uint16_t link_type, bool nanosecond = true) : nano_(nanosecond) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot open " + path);
    uint8_t h[24];
    store_le<uint32_t>(h, nano_ ? 0xa1b23c4d : 0xa1b2c3d4);
    store_le<uint16_t>(h + 4, 2);
    store_le<uint16_t>(h + 6, 4);
    store_le<int32_t>(h + 8, 0);
    store_le<uint32_t>(h + 12, 0);
    store_le<uint32_t>(h + 16, 262144);
    store_le<uint32_t>(h + 20, link_type);
    std::fwrite(h, 1, 24, f_);
  }
  ~PcapWriter() {
    if (f_) std::fclose(f_);
  }
  PcapWriter(const PcapWriter&) = delete;
  PcapWriter& operator=(const PcapWriter&) = delete;

  void write(int64_t ts_ns, const uint8_t* data, uint32_t len, uint32_t origlen = 0) {
    uint8_t h[16];
    int64_t sec = ts_ns / 1'000'000'000;
    int64_t frac = ts_ns % 1'000'000'000;
    if (!nano_) frac /= 1000;
    store_le<uint32_t>(h, static_cast<uint32_t>(sec));
    store_le<uint32_t>(h + 4, static_cast<uint32_t>(frac));
    store_le<uint32_t>(h + 8, len);
    store_le<uint32_t>(h + 12, origlen ? origlen : len);
    std::fwrite(h, 1, 16, f_);
    std::fwrite(data, 1, len, f_);
    ++packets_;
  }
  uint64_t packets() const { return packets_; }

 private:
  std::FILE* f_ = nullptr;
  bool nano_;
  uint64_t packets_ = 0;
};

}  // namespace mdp
