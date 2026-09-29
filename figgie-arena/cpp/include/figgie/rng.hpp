// Small deterministic PRNG (xoshiro256**), seeded by splitmix64, so that
// results are identical across platforms and standard library versions.
#pragma once

#include <cstdint>

namespace figgie {

inline uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

class Rng {
 public:
  explicit Rng(uint64_t seed = 1) { reseed(seed); }
  void reseed(uint64_t seed) {
    uint64_t x = seed;
    for (auto& w : s_) w = splitmix64(x);
  }
  uint64_t next() {
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }
  // Uniform integer in [0, n). Lemire's multiply-shift; bias is < 2^-32 for our n.
  uint32_t below(uint32_t n) {
    return static_cast<uint32_t>((static_cast<unsigned __int128>(next() >> 32) * n) >> 32);
  }
  double uniform() { return (next() >> 11) * 0x1.0p-53; }

 private:
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t s_[4];
};

// Derive an independent stream seed from (base, index) pairs.
inline uint64_t mix_seed(uint64_t base, uint64_t idx) {
  uint64_t x = base ^ (idx * 0xD1B54A32D192ED03ull);
  splitmix64(x);
  return splitmix64(x);
}

}  // namespace figgie
