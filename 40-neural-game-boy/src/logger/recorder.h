#pragma once
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "core/types.h"
#include "logger/format.h"

namespace gb::rec {

struct RecorderOptions {
  u32 frameskip = 1;
  u32 chunk_steps = 256;
  u64 rom_hash = 0;
  u64 seed = 0;
  std::string policy = "random";
  int zlib_level = 6;
};

// Pack 160x144 shades (0..3) into 5760 bytes.
void pack_frame(const u8* shades, u8* out);
void unpack_frame(const u8* packed, u8* shades);

// Streams (frame, action) steps into a .gbrec file, compressing one chunk at
// a time so memory stays bounded for arbitrarily long recordings.
class Recorder {
 public:
  Recorder(const std::string& path, RecorderOptions opts);
  ~Recorder();
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;

  void add(const u8* shades, u8 action);
  void close();

  u64 steps() const { return steps_; }
  u64 bytes_written() const { return bytes_; }

 private:
  void flush_chunk();

  std::FILE* f_ = nullptr;
  RecorderOptions opts_;
  std::vector<u8> actions_;
  std::vector<u8> frames_;  // packed, XOR-delta applied at flush
  std::vector<u8> comp_;
  u64 steps_ = 0;
  u64 bytes_ = 0;
};

// Reader used by tests and the C++ tools. The Python reader in
// python/gbwm/gbrec.py implements the same format.
struct Trajectory {
  Header header{};
  std::vector<u8> actions;
  std::vector<u8> frames;  // unpacked shades, num_steps * 160 * 144
};
Trajectory read_trajectory(const std::string& path);

}  // namespace gb::rec
