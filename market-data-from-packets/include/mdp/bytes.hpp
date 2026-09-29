// Little helpers for reading and writing fixed-width integers from byte buffers.
// All IEX fields are little endian. Capture file headers may be either byte order.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace mdp {

static_assert(std::endian::native == std::endian::little,
              "mdp assumes a little-endian host (x86-64 or arm64)");

template <class T>
constexpr T bswap(T v) noexcept {
  static_assert(std::is_integral_v<T>);
  if constexpr (sizeof(T) == 1) {
    return v;
  } else if constexpr (sizeof(T) == 2) {
    return static_cast<T>(__builtin_bswap16(static_cast<uint16_t>(v)));
  } else if constexpr (sizeof(T) == 4) {
    return static_cast<T>(__builtin_bswap32(static_cast<uint32_t>(v)));
  } else {
    return static_cast<T>(__builtin_bswap64(static_cast<uint64_t>(v)));
  }
}

template <class T>
inline T load_le(const uint8_t* p) noexcept {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

template <class T>
inline T load_be(const uint8_t* p) noexcept {
  return bswap(load_le<T>(p));
}

// Load with an optional byte swap (used by capture readers whose byte order
// is decided at runtime from the file magic).
template <class T>
inline T load_ord(const uint8_t* p, bool swap) noexcept {
  T v = load_le<T>(p);
  return swap ? bswap(v) : v;
}

template <class T>
inline void store_le(uint8_t* p, T v) noexcept {
  std::memcpy(p, &v, sizeof(T));
}

template <class T>
inline void store_be(uint8_t* p, T v) noexcept {
  store_le(p, bswap(v));
}

}  // namespace mdp
