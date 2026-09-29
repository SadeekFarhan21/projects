// kernels.h: host-side launcher API for the kernels in kernels/*.cuh.
//
// The same launcher source (src/launchers.inc) is compiled twice:
//   gkl::cuda::*  real CUDA launches, pointers are device pointers
//                 (library gkl_cuda, only built when CUDA is found)
//   gkl::emu::*   the SIMT emulator on the CPU, pointers are host pointers
//                 (library gkl_emu, always built)
// All launchers are asynchronous on the default stream in the CUDA build and
// synchronous in the emulator build. Shapes are validated; bad input throws
// std::invalid_argument.
#pragma once
#include <cstddef>

namespace gkl {

enum class SgemmAlgo { Naive, Tiled, RegBlocked };
enum class SoftmaxAlgo { Naive, Online };
enum class AttnAlgo { Naive, Flash };

// Q, K, V, O are [B, H, N, D] contiguous.
struct AttnShape {
  int B = 1, H = 1, N = 1, D = 64;
  bool causal = false;
};

const char* to_string(SgemmAlgo a);
const char* to_string(SoftmaxAlgo a);
const char* to_string(AttnAlgo a);

#define GKL_KERNEL_API                                                                          \
  /* C = alpha * A(MxK) * B(KxN) + beta * C, row-major */                                       \
  void sgemm(SgemmAlgo algo, int M, int N, int K, float alpha, const float* A, const float* B, \
             float beta, float* C);                                                             \
  void softmax_rows(SoftmaxAlgo algo, const float* x, float* y, int rows, int cols);            \
  void layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows,     \
                 int cols, float eps);                                                          \
  void rmsnorm(const float* x, const float* gamma, float* y, int rows, int cols, float eps);    \
  /* workspace: reduce_workspace_floats(n) floats; *out receives the sum */                    \
  size_t reduce_workspace_floats(long long n);                                                  \
  void reduce_sum(const float* x, long long n, float* workspace, float* out);                   \
  /* workspace: attention_workspace_floats(algo, s) floats (0 for Flash) */                    \
  size_t attention_workspace_floats(AttnAlgo algo, const AttnShape& s);                         \
  bool attention_supports(AttnAlgo algo, const AttnShape& s);                                   \
  void attention(AttnAlgo algo, const AttnShape& s, const float* Q, const float* K,             \
                 const float* V, float* O, float* workspace);

namespace cuda {
GKL_KERNEL_API
}
namespace emu {
GKL_KERNEL_API
}

}  // namespace gkl
