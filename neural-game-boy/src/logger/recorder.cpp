#include "logger/recorder.h"

#include <zlib.h>

#include <cstring>
#include <stdexcept>

namespace gb::rec {

void pack_frame(const u8* s, u8* out) {
  for (int i = 0; i < kPackedFrameBytes; ++i, s += 4)
    out[i] = u8((s[0] & 3) | ((s[1] & 3) << 2) | ((s[2] & 3) << 4) | ((s[3] & 3) << 6));
}

void unpack_frame(const u8* p, u8* s) {
  for (int i = 0; i < kPackedFrameBytes; ++i, s += 4) {
    u8 b = p[i];
    s[0] = b & 3;
    s[1] = (b >> 2) & 3;
    s[2] = (b >> 4) & 3;
    s[3] = (b >> 6) & 3;
  }
}

Recorder::Recorder(const std::string& path, RecorderOptions opts) : opts_(std::move(opts)) {
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) throw std::runtime_error("cannot open " + path);
  Header h{};
  std::memcpy(h.magic, kMagic, 8);
  h.version = kVersion;
  h.width = kScreenW;
  h.height = kScreenH;
  h.frameskip = opts_.frameskip;
  h.chunk_steps = opts_.chunk_steps;
  h.rom_hash = opts_.rom_hash;
  h.seed = opts_.seed;
  h.num_steps = 0;
  std::strncpy(h.policy, opts_.policy.c_str(), sizeof h.policy - 1);
  std::fwrite(&h, sizeof h, 1, f_);
  bytes_ = sizeof h;
  actions_.reserve(opts_.chunk_steps);
  frames_.reserve(size_t(opts_.chunk_steps) * kPackedFrameBytes);
}

Recorder::~Recorder() {
  try {
    close();
  } catch (...) {
  }
}

void Recorder::add(const u8* shades, u8 action) {
  actions_.push_back(action);
  size_t off = frames_.size();
  frames_.resize(off + kPackedFrameBytes);
  pack_frame(shades, &frames_[off]);
  ++steps_;
  if (actions_.size() >= opts_.chunk_steps) flush_chunk();
}

void Recorder::flush_chunk() {
  const u32 n = u32(actions_.size());
  if (n == 0) return;
  // XOR delta against the previous frame, back to front so each frame is
  // still intact when its successor reads it.
  for (u32 k = n - 1; k >= 1; --k) {
    u8* cur = &frames_[size_t(k) * kPackedFrameBytes];
    const u8* prev = cur - kPackedFrameBytes;
    for (int i = 0; i < kPackedFrameBytes; ++i) cur[i] ^= prev[i];
  }
  std::vector<u8> raw;
  raw.reserve(n + frames_.size());
  raw.insert(raw.end(), actions_.begin(), actions_.end());
  raw.insert(raw.end(), frames_.begin(), frames_.end());

  uLongf clen = compressBound(uLong(raw.size()));
  comp_.resize(clen);
  if (compress2(comp_.data(), &clen, raw.data(), uLong(raw.size()), opts_.zlib_level) != Z_OK)
    throw std::runtime_error("zlib compress failed");
  u32 hdr[3] = {n, u32(raw.size()), u32(clen)};
  std::fwrite(hdr, sizeof hdr, 1, f_);
  std::fwrite(comp_.data(), 1, clen, f_);
  bytes_ += sizeof hdr + clen;
  actions_.clear();
  frames_.clear();
}

void Recorder::close() {
  if (!f_) return;
  flush_chunk();
  // Patch num_steps in the header.
  std::fseek(f_, offsetof(Header, num_steps), SEEK_SET);
  u64 n = steps_;
  std::fwrite(&n, sizeof n, 1, f_);
  std::fclose(f_);
  f_ = nullptr;
}

Trajectory read_trajectory(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  Trajectory t;
  if (std::fread(&t.header, sizeof t.header, 1, f) != 1 || std::memcmp(t.header.magic, kMagic, 8) != 0) {
    std::fclose(f);
    throw std::runtime_error("not a gbrec file: " + path);
  }
  std::vector<u8> comp, raw;
  u32 hdr[3];
  while (std::fread(hdr, sizeof hdr, 1, f) == 1) {
    const u32 n = hdr[0];
    comp.resize(hdr[2]);
    raw.resize(hdr[1]);
    if (std::fread(comp.data(), 1, comp.size(), f) != comp.size()) break;
    uLongf rlen = uLongf(raw.size());
    if (uncompress(raw.data(), &rlen, comp.data(), uLong(comp.size())) != Z_OK) {
      std::fclose(f);
      throw std::runtime_error("corrupt chunk in " + path);
    }
    t.actions.insert(t.actions.end(), raw.begin(), raw.begin() + n);
    u8* packed = raw.data() + n;
    for (u32 k = 1; k < n; ++k) {
      u8* cur = packed + size_t(k) * kPackedFrameBytes;
      const u8* prev = cur - kPackedFrameBytes;
      for (int i = 0; i < kPackedFrameBytes; ++i) cur[i] ^= prev[i];
    }
    size_t off = t.frames.size();
    t.frames.resize(off + size_t(n) * kScreenW * kScreenH);
    for (u32 k = 0; k < n; ++k)
      unpack_frame(packed + size_t(k) * kPackedFrameBytes, &t.frames[off + size_t(k) * kScreenW * kScreenH]);
  }
  std::fclose(f);
  return t;
}

}  // namespace gb::rec
