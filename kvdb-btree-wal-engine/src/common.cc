#include "kvdb/common.h"

#include <array>
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

namespace kvdb {

void throw_errno(const std::string& what) {
  throw IoError(what + ": " + std::strerror(errno));
}

void sync_fd(int fd, SyncMode mode) {
  switch (mode) {
    case SyncMode::kNone:
      return;
    case SyncMode::kFsync:
      if (::fsync(fd) != 0) throw_errno("fsync");
      return;
    case SyncMode::kFullFsync:
#ifdef F_FULLFSYNC
      if (::fcntl(fd, F_FULLFSYNC) == 0) return;
      // Some filesystems (tmpfs, network) reject F_FULLFSYNC; fall back.
#endif
      if (::fsync(fd) != 0) throw_errno("fsync");
      return;
  }
}

namespace {
std::array<uint32_t, 256> make_crc_table() {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    t[i] = c;
  }
  return t;
}
const std::array<uint32_t, 256> kCrcTable = make_crc_table();
}  // namespace

uint32_t crc32_portable(const void* data, size_t n, uint32_t seed) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint32_t c = ~seed;
  for (size_t i = 0; i < n; ++i) c = kCrcTable[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return ~c;
}

uint32_t crc32(const void* data, size_t n, uint32_t seed) {
#if defined(__ARM_FEATURE_CRC32)
  // ARMv8 CRC32 instructions use the same (reflected IEEE) polynomial as the
  // table, so results are identical; about 8 bytes per instruction.
  const auto* p = static_cast<const unsigned char*>(data);
  uint32_t c = ~seed;
  for (; n >= 8; n -= 8, p += 8) {
    uint64_t w;
    std::memcpy(&w, p, 8);
    c = __crc32d(c, w);
  }
  for (; n > 0; --n, ++p) c = __crc32b(c, *p);
  return ~c;
#else
  return crc32_portable(data, n, seed);
#endif
}

namespace failpoint {
namespace {
const char* g_point = nullptr;
}
void set(const char* name) { g_point = name; }
void hit(const char* name) {
  if (g_point != nullptr && std::strcmp(g_point, name) == 0) ::_exit(77);
}
}  // namespace failpoint

}  // namespace kvdb
