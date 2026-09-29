// CPU reference tests: hand-checked cases, mathematical properties, and the
// float "algorithm" implementations against the double gold references.
#include <cmath>
#include <limits>
#include <vector>

#include "check.h"
#include "gkl/cpu_ref.h"

using namespace gkl::cpu;
using check::randn;

TEST(sgemm_ref_hand_example) {
  // [1 2 3; 4 5 6] * [7 8; 9 10; 11 12] = [58 64; 139 154]
  const float A[] = {1, 2, 3, 4, 5, 6}, B[] = {7, 8, 9, 10, 11, 12};
  float C[4] = {1, 1, 1, 1};
  sgemm_ref(2, 2, 3, 1.f, A, B, 0.f, C);
  const float want[] = {58, 64, 139, 154};
  EXPECT_CLOSE("hand", C, want, 4, 0.0, 0.0);
  sgemm_ref(2, 2, 3, 2.f, A, B, -1.f, C);  // C = 2AB - C = AB
  EXPECT_CLOSE("alpha/beta", C, want, 4, 0.0, 0.0);
}

TEST(sgemm_ref_beta_zero_ignores_nan) {
  const float A[] = {1, 2}, B[] = {3, 4};
  float C[1] = {std::numeric_limits<float>::quiet_NaN()};
  sgemm_ref(1, 1, 2, 1.f, A, B, 0.f, C);
  EXPECT(C[0] == 11.f);
}

TEST(sgemm_blocked_matches_ref) {
  const int shapes[][3] = {{1, 1, 1}, {7, 13, 5}, {64, 64, 64}, {100, 300, 257}, {513, 65, 33}};
  for (auto& s : shapes) {
    const int M = s[0], N = s[1], K = s[2];
    auto A = randn(size_t(M) * K, 1), B = randn(size_t(K) * N, 2), C0 = randn(size_t(M) * N, 3);
    auto want = C0, got = C0;
    sgemm_ref(M, N, K, 0.5f, A.data(), B.data(), 2.f, want.data());
    sgemm_cpu_blocked(M, N, K, 0.5f, A.data(), B.data(), 2.f, got.data());
    EXPECT_CLOSE("blocked", got.data(), want.data(), got.size(), 1e-4 * std::sqrt(K), 1e-4);
  }
}

TEST(softmax_ref_properties) {
  auto x = randn(8 * 300, 4, 0.f, 5.f);
  std::vector<float> y(x.size()), y2(x.size()), xs(x);
  softmax_ref(x.data(), y.data(), 8, 300);
  for (int r = 0; r < 8; ++r) {
    double s = 0;
    for (int c = 0; c < 300; ++c) {
      s += y[r * 300 + c];
      EXPECT(y[r * 300 + c] >= 0.f);
    }
    EXPECT(std::fabs(s - 1.0) < 1e-5);
  }
  for (auto& v : xs) v += 1000.f;  // shift invariance
  softmax_ref(xs.data(), y2.data(), 8, 300);
  EXPECT_CLOSE("shift", y2.data(), y.data(), y.size(), 1e-6, 1e-3);
}

TEST(softmax_online_matches_ref_incl_large_and_masked) {
  auto x = randn(6 * 1000, 5, 0.f, 30.f);
  x[0] = 1e4f;                       // extreme logit
  x[1000] = -INFINITY;               // row starting with a masked entry
  x[1001] = -INFINITY;
  std::vector<float> want(x.size()), got(x.size());
  softmax_ref(x.data(), want.data(), 6, 1000);
  softmax_online_cpu(x.data(), got.data(), 6, 1000);
  EXPECT_CLOSE("online", got.data(), want.data(), got.size(), 1e-7, 1e-4);
}

TEST(softmax_unsafe_overflows_as_expected) {
  // Documents *why* max subtraction exists: exp(100) overflows float.
  std::vector<float> big = {100.f, 99.f}, yb(2);
  softmax_unsafe_cpu(big.data(), yb.data(), 1, 2);
  EXPECT(std::isnan(yb[0]));  // inf / inf
}

TEST(layernorm_ref_properties) {
  const int R = 4, C = 777;
  auto x = randn(size_t(R) * C, 6, 3.f, 2.f);
  std::vector<float> g(C, 1.f), b(C, 0.f), y(x.size());
  layernorm_ref(x.data(), g.data(), b.data(), y.data(), R, C, 0.f);
  for (int r = 0; r < R; ++r) {
    double m = 0, v = 0;
    for (int c = 0; c < C; ++c) m += y[r * C + c];
    m /= C;
    for (int c = 0; c < C; ++c) v += (y[r * C + c] - m) * (y[r * C + c] - m);
    v /= C;
    EXPECT(std::fabs(m) < 1e-5);
    EXPECT(std::fabs(v - 1.0) < 1e-4);
  }
}

TEST(layernorm_welford_vs_onepass_with_large_mean) {
  const int R = 4, C = 4096;
  auto x = randn(size_t(R) * C, 7, 1e4f, 1.f);  // mean 1e4, std 1
  auto g = randn(C, 8), b = randn(C, 9);
  std::vector<float> want(x.size()), wel(x.size()), one(x.size());
  layernorm_ref(x.data(), g.data(), b.data(), want.data(), R, C, 1e-5f);
  layernorm_welford_cpu(x.data(), g.data(), b.data(), wel.data(), R, C, 1e-5f);
  layernorm_onepass_cpu(x.data(), g.data(), b.data(), one.data(), R, C, 1e-5f);
  // Welford: limited only by float rounding of x - mean (x ~ 1e4 has ulp ~ 1e-3).
  EXPECT_CLOSE("welford", wel.data(), want.data(), want.size(), 2e-2, 2e-2);
  // One-pass E[x^2]-E[x]^2 must be visibly wrong here (this is the point).
  auto c = check::compare(one.data(), want.data(), want.size(), 2e-2, 2e-2);
  EXPECT(!c.ok);
}

TEST(rmsnorm_ref_constant_row) {
  std::vector<float> x(64, -3.f), g(64), y(64);
  for (int i = 0; i < 64; ++i) g[i] = float(i);
  rmsnorm_ref(x.data(), g.data(), y.data(), 1, 64, 0.f);
  for (int i = 0; i < 64; ++i) EXPECT(std::fabs(y[i] + g[i]) < 1e-5);
}

TEST(reduce_ref_and_tree) {
  std::vector<float> ones(1000003, 1.f);
  EXPECT(reduce_sum_ref(ones.data(), ones.size()) == 1000003.0);
  auto x = randn(1 << 20, 10);
  const double want = reduce_sum_ref(x.data(), x.size());
  double abs_sum = 0;
  for (float v : x) abs_sum += std::fabs(v);
  const float tree = reduce_sum_tree_cpu(x.data(), x.size(), 256 * 1024);
  EXPECT(std::fabs(tree - want) < 1e-6 * abs_sum);
  // 16.7M ones: sequential float accumulation stalls at 2^24.
  std::vector<float> many(1 << 25, 1.f);
  EXPECT(reduce_sum_sequential_cpu(many.data(), many.size()) == 16777216.f);
  EXPECT(reduce_sum_tree_cpu(many.data(), many.size(), 1024) == 33554432.f);
}

TEST(attention_ref_identical_keys_average_values) {
  // All keys equal -> uniform attention -> every output row = mean of V rows.
  const int N = 10, D = 8;
  auto Q = randn(N * D, 11);
  std::vector<float> K(N * D, 0.5f), O(N * D);
  auto V = randn(N * D, 12);
  attention_ref(Q.data(), K.data(), V.data(), O.data(), 1, 1, N, D, false);
  for (int d = 0; d < D; ++d) {
    double mean = 0;
    for (int j = 0; j < N; ++j) mean += V[j * D + d];
    mean /= N;
    for (int i = 0; i < N; ++i) EXPECT(std::fabs(O[i * D + d] - mean) < 1e-5);
  }
  // Causal: row 0 can only see key 0.
  attention_ref(Q.data(), K.data(), V.data(), O.data(), 1, 1, N, D, true);
  EXPECT_CLOSE("causal row0", O.data(), V.data(), D, 1e-6, 0.0);
}

TEST(attention_flash_and_naive_cpu_match_ref) {
  struct S { int B, H, N, D; bool causal; int br, bc; };
  const S cases[] = {{1, 1, 1, 8, false, 64, 32},  {2, 3, 37, 16, false, 16, 8},
                     {1, 2, 100, 64, true, 64, 32}, {1, 1, 129, 32, true, 32, 32},
                     {2, 2, 64, 128, false, 32, 32}};
  for (auto& c : cases) {
    const size_t n = size_t(c.B) * c.H * c.N * c.D;
    auto Q = randn(n, 13), K = randn(n, 14), V = randn(n, 15);
    std::vector<float> want(n), flash(n), naive(n);
    attention_ref(Q.data(), K.data(), V.data(), want.data(), c.B, c.H, c.N, c.D, c.causal);
    attention_flash_cpu(Q.data(), K.data(), V.data(), flash.data(), c.B, c.H, c.N, c.D, c.causal,
                        c.br, c.bc);
    attention_naive_cpu(Q.data(), K.data(), V.data(), naive.data(), c.B, c.H, c.N, c.D, c.causal);
    EXPECT_CLOSE("flash_cpu", flash.data(), want.data(), n, 1e-5, 1e-4);
    EXPECT_CLOSE("naive_cpu", naive.data(), want.data(), n, 1e-5, 1e-4);
  }
}

int main(int argc, char** argv) { return check::run_all("test_ref", argc, argv); }
