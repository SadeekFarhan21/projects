// Shared constants and small helpers used across the storage engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace kvdb {

inline constexpr size_t kPageSize = 4096;

using PageId = uint32_t;
// Page 0 always holds the meta page, so 0 can double as "no page" for links.
inline constexpr PageId kMetaPageId = 0;
inline constexpr PageId kNoPage = 0;

// Size limits guarantee that any node holding at least 4 cells can be split
// into two halves that each fit in a page (see DESIGN.md, "Node splits").
inline constexpr size_t kMaxKeySize = 128;
inline constexpr size_t kMaxValueSize = 512;

// How hard a commit tries to reach stable storage.
enum class SyncMode {
  kNone,       // write() only: survives a process crash, not an OS crash
  kFsync,      // fsync(): on macOS this does not flush the drive cache
  kFullFsync,  // fcntl(F_FULLFSYNC): the only real barrier on macOS
};

class IoError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Throws IoError with errno text appended.
[[noreturn]] void throw_errno(const std::string& what);

// Flushes fd according to mode. kNone is a no-op.
void sync_fd(int fd, SyncMode mode);

// Unaligned little-endian load/store helpers (memcpy keeps UBSan quiet).
template <typename T>
inline T load(const char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}
template <typename T>
inline void store(char* p, T v) {
  std::memcpy(p, &v, sizeof(T));
}

// CRC-32 (IEEE, reflected). Chainable: crc32(b, crc32(a)) == crc32(a + b).
uint32_t crc32(const void* data, size_t n, uint32_t seed = 0);
// Table-driven reference implementation, used to test the fast path.
uint32_t crc32_portable(const void* data, size_t n, uint32_t seed = 0);

// Test-only crash injection. When the named point equals the configured one,
// the process calls _exit() immediately, simulating a power cut at that step.
namespace failpoint {
void set(const char* name);  // nullptr disables
void hit(const char* name);
}  // namespace failpoint

}  // namespace kvdb
