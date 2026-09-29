// check.h: a deliberately tiny test harness (no external dependencies).
#pragma once
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace check {

struct Registry {
  std::vector<std::pair<std::string, std::function<void()>>> tests;
  int failures = 0;
  int checks = 0;
  static Registry& get() {
    static Registry r;
    return r;
  }
};

struct Reg {
  Reg(const char* name, std::function<void()> fn) { Registry::get().tests.emplace_back(name, fn); }
};

#define TEST(name)                                        \
  static void name();                                     \
  static ::check::Reg reg_##name(#name, name);            \
  static void name()

#define EXPECT(cond)                                                               \
  do {                                                                             \
    ::check::Registry::get().checks++;                                             \
    if (!(cond)) {                                                                 \
      ::check::Registry::get().failures++;                                         \
      std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
    }                                                                              \
  } while (0)

// Max error of got vs want, with a mixed tolerance |g - w| <= atol + rtol * |w|.
struct Cmp {
  double max_abs = 0, max_rel = 0;
  bool ok = true;
  bool nan = false;
};

inline Cmp compare(const float* got, const float* want, size_t n, double atol, double rtol) {
  Cmp c;
  for (size_t i = 0; i < n; ++i) {
    const double g = got[i], w = want[i];
    if (std::isnan(g) != std::isnan(w)) { c.nan = true; c.ok = false; continue; }
    if (std::isinf(w) && g == w) continue;
    const double e = std::fabs(g - w);
    c.max_abs = std::max(c.max_abs, e);
    c.max_rel = std::max(c.max_rel, e / std::max(std::fabs(w), 1e-30));
    if (!(e <= atol + rtol * std::fabs(w))) c.ok = false;
  }
  return c;
}

#define EXPECT_CLOSE(label, got, want, n, atol, rtol)                                         \
  do {                                                                                          \
    auto c_ = ::check::compare((got), (want), (n), (atol), (rtol));                             \
    ::check::Registry::get().checks++;                                                          \
    if (!c_.ok) {                                                                               \
      ::check::Registry::get().failures++;                                                      \
      std::printf("    FAIL %s:%d %s: max_abs=%.3g max_rel=%.3g nan=%d\n", __FILE__, __LINE__, \
                  std::string(label).c_str(), c_.max_abs, c_.max_rel, int(c_.nan));             \
    }                                                                                           \
  } while (0)

inline std::vector<float> randn(size_t n, uint64_t seed, float mean = 0.f, float std = 1.f) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> d(mean, std);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

// Optional argv[1]: run only tests whose name contains that substring.
inline int run_all(const char* suite, int argc = 0, char** argv = nullptr) {
  auto& r = Registry::get();
  size_t ran = 0;
  for (auto& [name, fn] : r.tests) {
    if (argc > 1 && name.find(argv[1]) == std::string::npos) continue;
    ++ran;
    const int before = r.failures;
    fn();
    std::printf("[%s] %-44s %s\n", suite, name.c_str(), r.failures == before ? "ok" : "FAILED");
  }
  std::printf("%s: %zu tests, %d checks, %d failures\n", suite, ran, r.checks,
              r.failures);
  return r.failures == 0 ? 0 : 1;
}

}  // namespace check
