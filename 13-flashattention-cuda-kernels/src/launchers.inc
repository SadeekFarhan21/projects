// launchers.inc: one launcher implementation, compiled twice.
//   src/cuda/launchers.cu    -> namespace gkl::cuda (nvcc, real GPU)
//   src/emu/launchers_emu.cpp -> namespace gkl::emu (C++, SIMT emulator)
// The includer defines GKL_BACKEND_NS before including this file.
#include <stdexcept>
#include <string>

#include "gkl/kernels.h"
#include "kernels/attention.cuh"
#include "kernels/norms.cuh"
#include "kernels/reduce.cuh"
#include "kernels/sgemm.cuh"
#include "kernels/softmax.cuh"

#if defined(__CUDACC__)
#define GKL_CHECK_LAUNCH()                                                          \
  do {                                                                              \
    cudaError_t e = cudaGetLastError();                                             \
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA launch: ") +   \
                                                   cudaGetErrorString(e));          \
  } while (0)
#else
#define GKL_CHECK_LAUNCH() ((void)0)
#endif

namespace gkl::GKL_BACKEND_NS {
namespace k = ::gkl::kernels;

namespace {
inline unsigned ceil_div(long long a, long long b) { return unsigned((a + b - 1) / b); }
inline void require(bool ok, const char* msg) {
  if (!ok) throw std::invalid_argument(msg);
}

// Register-blocked tile configuration (see kernels/sgemm.cuh).
constexpr int kBM = 128, kBN = 128, kBK = 8, kTM = 8, kTN = 8;
constexpr int kTile = 32;
constexpr int kRowThreads = 256;  // softmax / norms / reduce block size
}  // namespace

void sgemm(SgemmAlgo algo, int M, int N, int K, float alpha, const float* A, const float* B,
           float beta, float* C) {
  require(M > 0 && N > 0 && K > 0, "sgemm: M, N, K must be positive");
  switch (algo) {
    case SgemmAlgo::Naive: {
      dim3 block(16, 16);
      dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
      GKL_LAUNCH(k::sgemm_naive_kernel, grid, block, 0, M, N, K, alpha, A, B, beta, C);
      break;
    }
    case SgemmAlgo::Tiled: {
      auto kfn = k::sgemm_tiled_kernel<kTile>;
      dim3 block(kTile, kTile);
      dim3 grid(ceil_div(N, kTile), ceil_div(M, kTile));
      GKL_LAUNCH(kfn, grid, block, 0, M, N, K, alpha, A, B, beta, C);
      break;
    }
    case SgemmAlgo::RegBlocked: {
      auto kfn = k::sgemm_regblock_kernel<kBM, kBN, kBK, kTM, kTN>;
      dim3 block((kBM / kTM) * (kBN / kTN));
      dim3 grid(ceil_div(N, kBN), ceil_div(M, kBM));
      GKL_LAUNCH(kfn, grid, block, 0, M, N, K, alpha, A, B, beta, C);
      break;
    }
  }
  GKL_CHECK_LAUNCH();
}

void softmax_rows(SoftmaxAlgo algo, const float* x, float* y, int rows, int cols) {
  require(rows > 0 && cols > 0, "softmax_rows: rows and cols must be positive");
  if (algo == SoftmaxAlgo::Naive) {
    GKL_LAUNCH(k::softmax_naive_kernel, dim3(ceil_div(rows, 128)), dim3(128), 0, x, y, rows, cols);
  } else {
    GKL_LAUNCH(k::softmax_online_kernel, dim3(rows), dim3(kRowThreads), 0, x, y, rows, cols);
  }
  GKL_CHECK_LAUNCH();
}

void layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows, int cols,
               float eps) {
  require(rows > 0 && cols > 0, "layernorm: rows and cols must be positive");
  GKL_LAUNCH(k::layernorm_kernel, dim3(rows), dim3(kRowThreads), 0, x, gamma, beta, y, rows, cols,
             eps);
  GKL_CHECK_LAUNCH();
}

void rmsnorm(const float* x, const float* gamma, float* y, int rows, int cols, float eps) {
  require(rows > 0 && cols > 0, "rmsnorm: rows and cols must be positive");
  GKL_LAUNCH(k::rmsnorm_kernel, dim3(rows), dim3(kRowThreads), 0, x, gamma, y, rows, cols, eps);
  GKL_CHECK_LAUNCH();
}

size_t reduce_workspace_floats(long long n) {
  const long long g = (n + 4 * kRowThreads - 1) / (4 * kRowThreads);
  return size_t(g < 1 ? 1 : (g > 1024 ? 1024 : g));
}

void reduce_sum(const float* x, long long n, float* workspace, float* out) {
  require(n > 0, "reduce_sum: n must be positive");
  const unsigned g = unsigned(reduce_workspace_floats(n));
  GKL_LAUNCH(k::reduce_sum_partial_kernel, dim3(g), dim3(kRowThreads), 0, x, n, workspace);
  GKL_CHECK_LAUNCH();
  GKL_LAUNCH(k::reduce_sum_partial_kernel, dim3(1), dim3(kRowThreads), 0,
             static_cast<const float*>(workspace), (long long)g, out);
  GKL_CHECK_LAUNCH();
}

size_t attention_workspace_floats(AttnAlgo algo, const AttnShape& s) {
  return algo == AttnAlgo::Naive ? size_t(s.B) * s.H * s.N * s.N : 0;
}

bool attention_supports(AttnAlgo algo, const AttnShape& s) {
  if (s.B <= 0 || s.H <= 0 || s.N <= 0 || s.D <= 0) return false;
  if (algo == AttnAlgo::Naive) return true;
  return s.D == 32 || s.D == 64 || s.D == 128;
}

namespace {
template <int D, int BR>
void launch_flash(const AttnShape& s, const float* Q, const float* K, const float* V, float* O,
                  float scale) {
  auto kfn = k::flash_attn_fwd_kernel<D, BR>;
  constexpr size_t smem = k::flash_smem_bytes<D, BR>();
  if (smem > 48 * 1024) GKL_ALLOW_BIG_SMEM(kfn, smem);
  dim3 grid(ceil_div(s.N, BR), unsigned(s.B * s.H));
  GKL_LAUNCH(kfn, grid, dim3(k::kAttnWarps * 32), smem, Q, K, V, O, s.N, scale, int(s.causal));
}
}  // namespace

void attention(AttnAlgo algo, const AttnShape& s, const float* Q, const float* K, const float* V,
               float* O, float* workspace) {
  require(attention_supports(algo, s), "attention: unsupported shape (flash needs D in {32,64,128})");
  const float scale = 1.0f / std::sqrt(float(s.D));
  if (algo == AttnAlgo::Naive) {
    require(workspace != nullptr, "attention: naive path needs a B*H*N*N workspace");
    const unsigned bh = unsigned(s.B * s.H);
    GKL_LAUNCH(k::attn_scores_kernel, dim3(ceil_div(s.N, 16), ceil_div(s.N, 16), bh), dim3(16, 16), 0,
               Q, K, workspace, s.N, s.D, scale, int(s.causal));
    GKL_CHECK_LAUNCH();
    GKL_LAUNCH(k::softmax_online_kernel, dim3(unsigned(s.B * s.H * s.N)), dim3(kRowThreads), 0,
               static_cast<const float*>(workspace), workspace, s.B * s.H * s.N, s.N);
    GKL_CHECK_LAUNCH();
    GKL_LAUNCH(k::attn_pv_kernel, dim3(ceil_div(s.D, 32), ceil_div(s.N, 8), bh), dim3(32, 8), 0,
               static_cast<const float*>(workspace), V, O, s.N, s.D);
    GKL_CHECK_LAUNCH();
    return;
  }
  switch (s.D) {
    case 32: launch_flash<32, 64>(s, Q, K, V, O, scale); break;
    case 64: launch_flash<64, 64>(s, Q, K, V, O, scale); break;
    case 128: launch_flash<128, 32>(s, Q, K, V, O, scale); break;
  }
  GKL_CHECK_LAUNCH();
}

}  // namespace gkl::GKL_BACKEND_NS
