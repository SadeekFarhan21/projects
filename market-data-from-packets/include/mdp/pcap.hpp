// Capture file reading without libpcap.
//
// Supports classic pcap (microsecond and nanosecond magics, either byte order)
// and pcapng (SHB, IDB, EPB, SPB, obsolete PB; if_tsresol and if_tsoffset;
// multiple sections; either byte order). The format is detected from the first
// four bytes. Input can be a plain file (memory mapped), a gzip file, or stdin.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdp {

class CaptureError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A cursor over a byte stream that can hand out contiguous views.
class ByteSource {
 public:
  virtual ~ByteSource() = default;
  // Returns a pointer to at least n contiguous bytes at the cursor, or nullptr
  // if fewer than n bytes remain. The pointer stays valid until the next call.
  virtual const uint8_t* ensure(size_t n) = 0;
  virtual void consume(size_t n) = 0;
  // Bytes consumed so far (uncompressed).
  virtual uint64_t position() const = 0;
  // True if the underlying stream ended abnormally (for example a truncated gzip).
  virtual bool truncated() const { return false; }
};

// Non-owning view over memory. Used by tests and the benchmark.
class MemorySource final : public ByteSource {
 public:
  MemorySource(const uint8_t* data, size_t len) : data_(data), len_(len) {}
  const uint8_t* ensure(size_t n) override { return pos_ + n <= len_ ? data_ + pos_ : nullptr; }
  void consume(size_t n) override { pos_ += n; }
  uint64_t position() const override { return pos_; }

 private:
  const uint8_t* data_;
  size_t len_;
  size_t pos_ = 0;
};

// Opens a path. "-" means stdin. gzip is detected by magic (1f 8b) and
// decompressed on the fly with zlib; plain files are memory mapped.
std::unique_ptr<ByteSource> open_source(const std::string& path);

enum class CaptureFormat { Pcap, PcapNg };

// Link-layer types we understand (LINKTYPE_* values from tcpdump.org).
enum LinkType : uint16_t {
  kLinkEthernet = 1,
  kLinkRaw = 101,
  kLinkLinuxSll = 113,
  kLinkIpv4 = 228,
  kLinkLinuxSll2 = 276,
};

struct Packet {
  int64_t ts_ns = 0;         // capture timestamp, ns since epoch (0 if the block has none)
  const uint8_t* data = nullptr;
  uint32_t caplen = 0;
  uint32_t origlen = 0;
  uint16_t link_type = 0;
  uint32_t interface_id = 0;
};

struct CaptureStats {
  uint64_t packets = 0;
  uint64_t blocks_skipped = 0;  // pcapng blocks we do not need (NRB, ISB, custom, ...)
  uint64_t sections = 0;
};

class CaptureReader {
 public:
  explicit CaptureReader(std::unique_ptr<ByteSource> src);
  CaptureFormat format() const { return format_; }
  // Fills `out` and returns true, or returns false at end of input. The packet
  // data pointer is valid until the next call. Throws CaptureError on corruption.
  bool next(Packet& out);
  const CaptureStats& stats() const { return stats_; }
  uint64_t bytes_consumed() const { return src_->position(); }
  bool truncated() const { return truncated_ || src_->truncated(); }

 private:
  struct Interface {
    uint16_t link_type = 0;
    uint32_t snaplen = 0;
    // timestamp units: ts_ns = ts * mul / div + offset_ns
    int64_t mul = 1000;
    int64_t div = 1;
    int64_t offset_ns = 0;
  };

  bool next_pcap(Packet& out);
  bool next_pcapng(Packet& out);
  void read_shb();
  void parse_idb(const uint8_t* body, uint32_t body_len);
  int64_t to_ns(const Interface& itf, uint64_t ts) const;

  std::unique_ptr<ByteSource> src_;
  CaptureFormat format_{};
  bool swap_ = false;
  bool truncated_ = false;
  // classic pcap
  uint16_t pcap_link_ = 0;
  int64_t pcap_frac_mul_ = 1000;  // 1000 for usec, 1 for nsec
  // pcapng
  std::vector<Interface> ifaces_;
  CaptureStats stats_;
};

}  // namespace mdp
