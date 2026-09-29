#include "mdp/pcap.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <cerrno>
#include <cstring>

#include "mdp/bytes.hpp"

namespace mdp {

namespace {

constexpr uint32_t kPcapMagicUsec = 0xa1b2c3d4;
constexpr uint32_t kPcapMagicNsec = 0xa1b23c4d;
constexpr uint32_t kPcapNgShb = 0x0a0d0d0a;
constexpr uint32_t kPcapNgByteOrder = 0x1a2b3c4d;
constexpr uint32_t kPcapNgIdb = 1;
constexpr uint32_t kPcapNgObsoletePb = 2;
constexpr uint32_t kPcapNgSpb = 3;
constexpr uint32_t kPcapNgEpb = 6;
constexpr uint32_t kMaxRecord = 16u << 20;  // 16 MiB; far above any real snaplen

// Owns an mmap of a whole file.
class MmapSource final : public ByteSource {
 public:
  explicit MmapSource(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw CaptureError("open " + path + ": " + std::strerror(errno));
    struct stat st {};
    if (::fstat(fd_, &st) != 0) throw CaptureError("stat " + path);
    len_ = static_cast<size_t>(st.st_size);
    if (len_ > 0) {
      void* p = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (p == MAP_FAILED) throw CaptureError("mmap " + path + ": " + std::strerror(errno));
      ::madvise(p, len_, MADV_SEQUENTIAL);
      data_ = static_cast<const uint8_t*>(p);
    }
  }
  ~MmapSource() override {
    if (data_) ::munmap(const_cast<uint8_t*>(data_), len_);
    if (fd_ >= 0) ::close(fd_);
  }
  const uint8_t* ensure(size_t n) override { return pos_ + n <= len_ ? data_ + pos_ : nullptr; }
  void consume(size_t n) override { pos_ += n; }
  uint64_t position() const override { return pos_; }

 private:
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  size_t len_ = 0;
  size_t pos_ = 0;
};

// Buffered reader over zlib's gz* API. gzread is transparent for non-gzip
// input, so this also serves stdin whatever its content.
class GzSource final : public ByteSource {
 public:
  explicit GzSource(const std::string& path) {
    gz_ = path == "-" ? gzdopen(::dup(0), "rb") : gzopen(path.c_str(), "rb");
    if (!gz_) throw CaptureError("gzopen " + path);
    gzbuffer(gz_, 1 << 20);
    buf_.resize(8 << 20);
  }
  ~GzSource() override {
    if (gz_) gzclose(gz_);
  }
  const uint8_t* ensure(size_t n) override {
    if (end_ - begin_ >= n) return buf_.data() + begin_;
    if (eof_) return nullptr;
    if (begin_ > 0) {
      std::memmove(buf_.data(), buf_.data() + begin_, end_ - begin_);
      end_ -= begin_;
      begin_ = 0;
    }
    if (n > buf_.size()) buf_.resize(std::max(n, buf_.size() * 2));
    while (end_ < n && !eof_) {
      size_t want = buf_.size() - end_;
      if (want > (1u << 30)) want = 1u << 30;
      int got = gzread(gz_, buf_.data() + end_, static_cast<unsigned>(want));
      if (got > 0) {
        end_ += static_cast<size_t>(got);
        continue;
      }
      int errnum = 0;
      const char* msg = gzerror(gz_, &errnum);
      if (got < 0 || (errnum != Z_OK && errnum != Z_STREAM_END)) {
        // Z_BUF_ERROR means the compressed stream ended mid-member: a
        // truncated download. Treat as end of data but remember it.
        if (errnum == Z_BUF_ERROR || errnum == Z_DATA_ERROR) {
          truncated_ = true;
        } else {
          throw CaptureError(std::string("gzread: ") + (msg ? msg : "unknown"));
        }
      }
      eof_ = true;
    }
    return end_ - begin_ >= n ? buf_.data() + begin_ : nullptr;
  }
  void consume(size_t n) override {
    begin_ += n;
    pos_ += n;
  }
  uint64_t position() const override { return pos_; }
  bool truncated() const override { return truncated_; }

 private:
  gzFile gz_ = nullptr;
  std::vector<uint8_t> buf_;
  size_t begin_ = 0;
  size_t end_ = 0;
  uint64_t pos_ = 0;
  bool eof_ = false;
  bool truncated_ = false;
};

bool looks_gzip(const std::string& path) {
  if (path == "-") return true;
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw CaptureError("open " + path + ": " + std::strerror(errno));
  uint8_t m[2] = {0, 0};
  ssize_t n = ::read(fd, m, 2);
  ::close(fd);
  return n == 2 && m[0] == 0x1f && m[1] == 0x8b;
}

int64_t pow10(int e) {
  int64_t v = 1;
  while (e-- > 0) v *= 10;
  return v;
}

}  // namespace

std::unique_ptr<ByteSource> open_source(const std::string& path) {
  if (looks_gzip(path)) return std::make_unique<GzSource>(path);
  return std::make_unique<MmapSource>(path);
}

CaptureReader::CaptureReader(std::unique_ptr<ByteSource> src) : src_(std::move(src)) {
  const uint8_t* p = src_->ensure(4);
  if (!p) throw CaptureError("capture shorter than 4 bytes");
  uint32_t magic = load_le<uint32_t>(p);
  if (magic == kPcapNgShb) {
    format_ = CaptureFormat::PcapNg;
    read_shb();
    return;
  }
  format_ = CaptureFormat::Pcap;
  if (magic == kPcapMagicUsec) {
    swap_ = false;
    pcap_frac_mul_ = 1000;
  } else if (magic == bswap(kPcapMagicUsec)) {
    swap_ = true;
    pcap_frac_mul_ = 1000;
  } else if (magic == kPcapMagicNsec) {
    swap_ = false;
    pcap_frac_mul_ = 1;
  } else if (magic == bswap(kPcapMagicNsec)) {
    swap_ = true;
    pcap_frac_mul_ = 1;
  } else {
    char buf[64];
    std::snprintf(buf, sizeof buf, "unknown capture magic 0x%08x", magic);
    throw CaptureError(buf);
  }
  p = src_->ensure(24);
  if (!p) throw CaptureError("truncated pcap global header");
  pcap_link_ = static_cast<uint16_t>(load_ord<uint32_t>(p + 20, swap_) & 0xffff);
  src_->consume(24);
  stats_.sections = 1;
}

bool CaptureReader::next(Packet& out) {
  return format_ == CaptureFormat::Pcap ? next_pcap(out) : next_pcapng(out);
}

bool CaptureReader::next_pcap(Packet& out) {
  const uint8_t* h = src_->ensure(16);
  if (!h) {
    if (src_->ensure(1)) truncated_ = true;
    return false;
  }
  uint32_t sec = load_ord<uint32_t>(h, swap_);
  uint32_t frac = load_ord<uint32_t>(h + 4, swap_);
  uint32_t incl = load_ord<uint32_t>(h + 8, swap_);
  uint32_t orig = load_ord<uint32_t>(h + 12, swap_);
  if (incl > kMaxRecord) throw CaptureError("pcap record length too large: " + std::to_string(incl));
  const uint8_t* rec = src_->ensure(16 + static_cast<size_t>(incl));
  if (!rec) {
    truncated_ = true;
    return false;
  }
  out.ts_ns = static_cast<int64_t>(sec) * 1'000'000'000 + static_cast<int64_t>(frac) * pcap_frac_mul_;
  out.data = rec + 16;
  out.caplen = incl;
  out.origlen = orig;
  out.link_type = pcap_link_;
  out.interface_id = 0;
  src_->consume(16 + static_cast<size_t>(incl));
  ++stats_.packets;
  return true;
}

void CaptureReader::read_shb() {
  // Byte order is only known after reading the byte-order magic at offset 8.
  const uint8_t* p = src_->ensure(12);
  if (!p) throw CaptureError("truncated pcapng section header");
  uint32_t bom = load_le<uint32_t>(p + 8);
  if (bom == kPcapNgByteOrder) {
    swap_ = false;
  } else if (bom == bswap(kPcapNgByteOrder)) {
    swap_ = true;
  } else {
    throw CaptureError("bad pcapng byte-order magic");
  }
  uint32_t len = load_ord<uint32_t>(p + 4, swap_);
  if (len < 28 || len % 4 != 0 || len > kMaxRecord) throw CaptureError("bad SHB length");
  p = src_->ensure(len);
  if (!p) throw CaptureError("truncated SHB");
  if (load_ord<uint32_t>(p + len - 4, swap_) != len) throw CaptureError("SHB trailing length mismatch");
  uint16_t major = load_ord<uint16_t>(p + 12, swap_);
  if (major != 1) throw CaptureError("unsupported pcapng major version");
  ifaces_.clear();  // interface ids are scoped to a section
  ++stats_.sections;
  src_->consume(len);
}

void CaptureReader::parse_idb(const uint8_t* body, uint32_t body_len) {
  if (body_len < 8) throw CaptureError("short IDB");
  Interface itf;
  itf.link_type = load_ord<uint16_t>(body, swap_);
  itf.snaplen = load_ord<uint32_t>(body + 4, swap_);
  // options: code u16, length u16, value padded to 4
  uint32_t off = 8;
  while (off + 4 <= body_len) {
    uint16_t code = load_ord<uint16_t>(body + off, swap_);
    uint16_t olen = load_ord<uint16_t>(body + off + 2, swap_);
    off += 4;
    if (code == 0) break;  // opt_endofopt
    if (off + olen > body_len) throw CaptureError("IDB option overruns block");
    if (code == 9 && olen >= 1) {  // if_tsresol
      uint8_t r = body[off];
      int e = r & 0x7f;
      if (r & 0x80) {
        // 2^-e seconds per tick; store as mul/div with div = 2^e
        if (e > 62) throw CaptureError("if_tsresol too fine");
        itf.mul = 1'000'000'000;
        itf.div = int64_t{1} << e;
      } else if (e <= 9) {
        itf.mul = pow10(9 - e);
        itf.div = 1;
      } else {
        if (e > 18) throw CaptureError("if_tsresol too fine");
        itf.mul = 1;
        itf.div = pow10(e - 9);
      }
    } else if (code == 14 && olen >= 8) {  // if_tsoffset, seconds
      itf.offset_ns = load_ord<int64_t>(body + off, swap_) * 1'000'000'000;
    }
    off += (olen + 3u) & ~3u;
  }
  ifaces_.push_back(itf);
}

int64_t CaptureReader::to_ns(const Interface& itf, uint64_t ts) const {
  __int128 v = static_cast<__int128>(ts) * itf.mul / itf.div;
  return static_cast<int64_t>(v) + itf.offset_ns;
}

bool CaptureReader::next_pcapng(Packet& out) {
  for (;;) {
    const uint8_t* h = src_->ensure(8);
    if (!h) {
      if (src_->ensure(1)) truncated_ = true;
      return false;
    }
    uint32_t type_raw = load_le<uint32_t>(h);
    if (type_raw == kPcapNgShb) {  // palindromic, same in both orders
      read_shb();
      continue;
    }
    uint32_t type = load_ord<uint32_t>(h, swap_);
    uint32_t len = load_ord<uint32_t>(h + 4, swap_);
    if (len < 12 || len % 4 != 0 || len > kMaxRecord) {
      throw CaptureError("bad pcapng block length " + std::to_string(len) + " at offset " +
                         std::to_string(src_->position()));
    }
    const uint8_t* b = src_->ensure(len);
    if (!b) {
      truncated_ = true;
      return false;
    }
    if (load_ord<uint32_t>(b + len - 4, swap_) != len) throw CaptureError("pcapng trailing length mismatch");
    const uint8_t* body = b + 8;
    uint32_t body_len = len - 12;

    if (type == kPcapNgEpb) {
      if (body_len < 20) throw CaptureError("short EPB");
      uint32_t iface = load_ord<uint32_t>(body, swap_);
      if (iface >= ifaces_.size()) throw CaptureError("EPB references unknown interface");
      uint64_t ts = (static_cast<uint64_t>(load_ord<uint32_t>(body + 4, swap_)) << 32) |
                    load_ord<uint32_t>(body + 8, swap_);
      uint32_t cap = load_ord<uint32_t>(body + 12, swap_);
      uint32_t orig = load_ord<uint32_t>(body + 16, swap_);
      if (20ull + cap > body_len) throw CaptureError("EPB captured length overruns block");
      const Interface& itf = ifaces_[iface];
      out.ts_ns = to_ns(itf, ts);
      out.data = body + 20;
      out.caplen = cap;
      out.origlen = orig;
      out.link_type = itf.link_type;
      out.interface_id = iface;
      src_->consume(len);
      ++stats_.packets;
      return true;
    }
    if (type == kPcapNgObsoletePb) {
      if (body_len < 20) throw CaptureError("short PB");
      uint32_t iface = load_ord<uint16_t>(body, swap_);
      if (iface >= ifaces_.size()) throw CaptureError("PB references unknown interface");
      uint64_t ts = (static_cast<uint64_t>(load_ord<uint32_t>(body + 4, swap_)) << 32) |
                    load_ord<uint32_t>(body + 8, swap_);
      uint32_t cap = load_ord<uint32_t>(body + 12, swap_);
      uint32_t orig = load_ord<uint32_t>(body + 16, swap_);
      if (20ull + cap > body_len) throw CaptureError("PB captured length overruns block");
      const Interface& itf = ifaces_[iface];
      out.ts_ns = to_ns(itf, ts);
      out.data = body + 20;
      out.caplen = cap;
      out.origlen = orig;
      out.link_type = itf.link_type;
      out.interface_id = iface;
      src_->consume(len);
      ++stats_.packets;
      return true;
    }
    if (type == kPcapNgSpb) {
      if (body_len < 4) throw CaptureError("short SPB");
      if (ifaces_.empty()) throw CaptureError("SPB without interface");
      uint32_t orig = load_ord<uint32_t>(body, swap_);
      uint32_t cap = orig;
      if (ifaces_[0].snaplen && cap > ifaces_[0].snaplen) cap = ifaces_[0].snaplen;
      if (cap > body_len - 4) cap = body_len - 4;
      out.ts_ns = 0;  // SPB has no timestamp
      out.data = body + 4;
      out.caplen = cap;
      out.origlen = orig;
      out.link_type = ifaces_[0].link_type;
      out.interface_id = 0;
      src_->consume(len);
      ++stats_.packets;
      return true;
    }
    if (type == kPcapNgIdb) {
      parse_idb(body, body_len);
    } else {
      ++stats_.blocks_skipped;
    }
    src_->consume(len);
  }
}

}  // namespace mdp
