// Runs the real kernel source (kernels/*.cuh) on the CPU through the SIMT
// emulator and checks every kernel against the gold CPU references.
#include <cmath>
#include <limits>
#include <vector>

#include "check.h"
#include "gkl/cpu_ref.h"
#include "gkl/cuda_compat.h"
#include "gkl/kernels.h"
#include "kernels/reduce_utils.cuh"

using namespace gkl;
using check::randn;

// ------------------------------------------------------------ emulator self tests
__global__ void shfl_probe_kernel(int* out) {
  const int lane = threadIdx.x % 32;
  const int base = threadIdx.x * 4;
  out[base + 0] = __shfl_xor_sync(0xffffffffu, lane, 1);
  out[base + 1] = __shfl_down_sync(0xffffffffu, lane, 4);
  out[base + 2] = __shfl_sync(0xffffffffu, lane * 10, 3);
  out[base + 3] = __shfl_xor_sync(0xffffffffu, lane, 8, 16);  // width 16 segments
}

TEST(emu_shuffle_semantics) {
  std::vector<int> out(64 * 4);
  GKL_LAUNCH(shfl_probe_kernel, dim3(1), dim3(64), 0, out.data());
  for (int t = 0; t < 64; ++t) {
    const int lane = t % 32;
    EXPECT(out[t * 4 + 0] == (lane ^ 1));
    EXPECT(out[t * 4 + 1] == (lane + 4 < 32 ? lane + 4 : lane));
    EXPECT(out[t * 4 + 2] == 30);
    EXPECT(out[t * 4 + 3] == (lane ^ 8));
  }
}

// Block-level reduction where half the threads exit early: exercises the
// barrier "drop" path and cross-block reuse of static shared memory.
__global__ void early_exit_sum_kernel(const float* x, float* out, int n) {
  if (blockIdx.x % 2 == 1) return;  // whole blocks leave (block-uniform)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = i < n ? x[i] : 0.f;
  v = gkl::kernels::block_reduce_sum(v);
  if (threadIdx.x == 0) out[blockIdx.x] = v;
}

TEST(emu_block_reduce_and_early_exit) {
  const int n = 128 * 6;
  std::vector<float> x(n), out(6, -1.f);
  for (int i = 0; i < n; ++i) x[i] = float(i % 7);
  GKL_LAUNCH(early_exit_sum_kernel, dim3(6), dim3(128), 0, x.data(), out.data(), n);
  for (int b = 0; b < 6; ++b) {
    if (b % 2 == 1) { EXPECT(out[b] == -1.f); continue; }
    float want = 0;
    for (int i = b * 128; i < (b + 1) * 128; ++i) want += x[i];
    EXPECT(out[b] == want);
  }
}

// ------------------------------------------------------------ SGEMM
TEST(sgemm_all_algos_match_ref) {
  const int shapes[][3] = {{1, 1, 1}, {17, 33, 9}, {64, 64, 64}, {130, 70, 45}, {129, 257, 33}};
  for (auto algo : {SgemmAlgo::Naive, SgemmAlgo::Tiled, SgemmAlgo::RegBlocked}) {
    for (auto& s : shapes) {
      const int M = s[0], N = s[1], K = s[2];
      auto A = randn(size_t(M) * K, 21), B = randn(size_t(K) * N, 22), C0 = randn(size_t(M) * N, 23);
      auto want = C0, got = C0;
      cpu::sgemm_ref(M, N, K, 1.5f, A.data(), B.data(), -0.5f, want.data());
      emu::sgemm(algo, M, N, K, 1.5f, A.data(), B.data(), -0.5f, got.data());
      EXPECT_CLOSE(std::string("sgemm ") + to_string(algo), got.data(), want.data(), got.size(),
                   1e-4 * std::sqrt(double(K)), 1e-4);
    }
  }
}

TEST(sgemm_beta_zero_does_not_read_c) {
  const int M = 40, N = 50, K = 20;
  auto A = randn(M * K, 24), B = randn(K * N, 25);
  std::vector<float> want(M * N, 0.f);
  cpu::sgemm_ref(M, N, K, 1.f, A.data(), B.data(), 0.f, want.data());
  for (auto algo : {SgemmAlgo::Naive, SgemmAlgo::Tiled, SgemmAlgo::RegBlocked}) {
    std::vector<float> got(M * N, std::numeric_limits<float>::quiet_NaN());
    emu::sgemm(algo, M, N, K, 1.f, A.data(), B.data(), 0.f, got.data());
    EXPECT_CLOSE(std::string("beta0 ") + to_string(algo), got.data(), want.data(), got.size(), 1e-4,
                 1e-4);
  }
}

// ------------------------------------------------------------ softmax
TEST(softmax_both_algos_match_ref) {
  const int shapes[][2] = {{1, 1}, {3, 7}, {5, 1000}, {2, 4097}};
  for (auto algo : {SoftmaxAlgo::Naive, SoftmaxAlgo::Online}) {
    for (auto& s : shapes) {
      const int R = s[0], C = s[1];
      auto x = randn(size_t(R) * C, 31, 0.f, 20.f);
      x[0] = 3000.f;  // would overflow without max subtraction
      if (R > 1 && C > 2) { x[C] = -INFINITY; x[C + 1] = -INFINITY; }  // masked prefix
      std::vector<float> want(x.size()), got(x.size());
      cpu::softmax_ref(x.data(), want.data(), R, C);
      emu::softmax_rows(algo, x.data(), got.data(), R, C);
      EXPECT_CLOSE(std::string("softmax ") + to_string(algo), got.data(), want.data(), got.size(),
                   1e-7, 1e-4);
    }
  }
}

TEST(softmax_online_in_place) {
  auto x = randn(4 * 300, 32);
  std::vector<float> want(x.size());
  cpu::softmax_ref(x.data(), want.data(), 4, 300);
  emu::softmax_rows(SoftmaxAlgo::Online, x.data(), x.data(), 4, 300);
  EXPECT_CLOSE("in place", x.data(), want.data(), x.size(), 1e-7, 1e-4);
}

// ------------------------------------------------------------ norms
TEST(layernorm_matches_ref) {
  const int shapes[][2] = {{1, 1}, {3, 5}, {4, 768}, {2, 5000}};
  for (auto& s : shapes) {
    const int R = s[0], C = s[1];
    auto x = randn(size_t(R) * C, 41, 2.f, 3.f);
    auto g = randn(C, 42), b = randn(C, 43);
    std::vector<float> want(x.size()), got(x.size());
    cpu::layernorm_ref(x.data(), g.data(), b.data(), want.data(), R, C, 1e-5f);
    emu::layernorm(x.data(), g.data(), b.data(), got.data(), R, C, 1e-5f);
    EXPECT_CLOSE("layernorm", got.data(), want.data(), got.size(), 1e-4, 1e-4);
  }
}

TEST(layernorm_large_mean_is_stable) {
  const int R = 3, C = 2048;
  auto x = randn(size_t(R) * C, 44, 1e4f, 1.f);
  std::vector<float> g(C, 1.f), b(C, 0.f), want(x.size()), got(x.size());
  cpu::layernorm_ref(x.data(), g.data(), b.data(), want.data(), R, C, 1e-5f);
  emu::layernorm(x.data(), g.data(), b.data(), got.data(), R, C, 1e-5f);
  // Error floor is the float rounding of x itself (ulp(1e4) ~ 1e-3).
  EXPECT_CLOSE("layernorm mean 1e4", got.data(), want.data(), got.size(), 5e-3, 5e-3);
}

TEST(rmsnorm_matches_ref) {
  const int shapes[][2] = {{1, 1}, {3, 5}, {4, 768}, {2, 5000}};
  for (auto& s : shapes) {
    const int R = s[0], C = s[1];
    auto x = randn(size_t(R) * C, 51, 0.5f, 2.f);
    auto g = randn(C, 52);
    std::vector<float> want(x.size()), got(x.size());
    cpu::rmsnorm_ref(x.data(), g.data(), want.data(), R, C, 1e-6f);
    emu::rmsnorm(x.data(), g.data(), got.data(), R, C, 1e-6f);
    EXPECT_CLOSE("rmsnorm", got.data(), want.data(), got.size(), 1e-5, 1e-4);
  }
}

// ------------------------------------------------------------ reduction
TEST(reduce_sum_matches_ref) {
  for (long long n : {1LL, 31LL, 256LL, 1000LL, 65537LL, 300000LL}) {
    auto x = randn(size_t(n), 61);
    std::vector<float> ws(emu::reduce_workspace_floats(n));
    float out = 0.f;
    emu::reduce_sum(x.data(), n, ws.data(), &out);
    const double want = cpu::reduce_sum_ref(x.data(), n);
    double abs_sum = 0;
    for (float v : x) abs_sum += std::fabs(v);
    EXPECT(std::fabs(out - want) <= 1e-6 * abs_sum + 1e-6);
  }
}

// ------------------------------------------------------------ attention
TEST(attention_naive_and_flash_match_ref) {
  struct S { int B, H, N, D; bool causal; };
  const S cases[] = {{1, 1, 1, 64, false},  {1, 2, 33, 32, false}, {2, 1, 100, 64, false},
                     {1, 2, 100, 64, true}, {1, 1, 70, 128, true}, {1, 1, 65, 128, false}};
  for (auto algo : {AttnAlgo::Naive, AttnAlgo::Flash}) {
    for (auto& c : cases) {
      AttnShape s{c.B, c.H, c.N, c.D, c.causal};
      const size_t n = size_t(c.B) * c.H * c.N * c.D;
      auto Q = randn(n, 71), K = randn(n, 72), V = randn(n, 73);
      std::vector<float> want(n), got(n, -7.f), ws(emu::attention_workspace_floats(algo, s));
      cpu::attention_ref(Q.data(), K.data(), V.data(), want.data(), c.B, c.H, c.N, c.D, c.causal);
      emu::attention(algo, s, Q.data(), K.data(), V.data(), got.data(), ws.data());
      char label[96];
      std::snprintf(label, sizeof label, "attn %s B%d H%d N%d D%d causal%d", to_string(algo), c.B,
                    c.H, c.N, c.D, int(c.causal));
      EXPECT_CLOSE(label, got.data(), want.data(), n, 2e-5, 1e-4);
    }
  }
}

TEST(attention_rejects_unsupported_head_dim) {
  AttnShape s{1, 1, 8, 48, false};
  EXPECT(!emu::attention_supports(AttnAlgo::Flash, s));
  EXPECT(emu::attention_supports(AttnAlgo::Naive, s));
  bool threw = false;
  try {
    std::vector<float> buf(8 * 48);
    emu::attention(AttnAlgo::Flash, s, buf.data(), buf.data(), buf.data(), buf.data(), nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT(threw);
}

int main(int argc, char** argv) { return check::run_all("test_emu_kernels", argc, argv); }
