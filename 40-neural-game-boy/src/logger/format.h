#pragma once
#include <cstdint>

// On-disk layout of a .gbrec trajectory file (all integers little endian).
//
//   Header (64 bytes)
//     char     magic[8]      "GBREC\0v1"
//     uint32   version       1
//     uint16   width         160
//     uint16   height        144
//     uint32   frameskip     emulator frames per logged step
//     uint32   chunk_steps   max steps per chunk
//     uint64   rom_hash      FNV-1a 64 of the ROM image
//     uint64   seed          policy seed
//     uint64   num_steps     total steps (patched on close)
//     char     policy[16]    NUL padded policy name
//   Chunk (repeated)
//     uint32   n             steps in this chunk
//     uint32   raw_bytes     n + n * 5760
//     uint32   comp_bytes    size of the zlib stream that follows
//     byte     zlib[comp_bytes]
//   Decompressed chunk payload
//     uint8    actions[n]            button mask held after frame t
//     uint8    frames[n][5760]       2 bits per pixel, 4 pixels per byte,
//                                    pixel i in bits 2*(i%4)..2*(i%4)+1;
//                                    frame 0 of the chunk is stored as is,
//                                    frame k>0 is XORed with frame k-1.
//
// Step semantics: frames[t] is the screen the policy saw, actions[t] is the
// button mask it then held for `frameskip` frames, so frames[t+1] is the
// consequence of actions[t]. Chunks are independent, so a reader can seek.
namespace gb::rec {

inline constexpr char kMagic[8] = {'G', 'B', 'R', 'E', 'C', '\0', 'v', '1'};
inline constexpr std::uint32_t kVersion = 1;
inline constexpr int kPackedFrameBytes = 160 * 144 / 4;  // 5760

#pragma pack(push, 1)
struct Header {
  char magic[8];
  std::uint32_t version;
  std::uint16_t width;
  std::uint16_t height;
  std::uint32_t frameskip;
  std::uint32_t chunk_steps;
  std::uint64_t rom_hash;
  std::uint64_t seed;
  std::uint64_t num_steps;
  char policy[16];
};
#pragma pack(pop)
static_assert(sizeof(Header) == 64, "header must be 64 bytes");

}  // namespace gb::rec
