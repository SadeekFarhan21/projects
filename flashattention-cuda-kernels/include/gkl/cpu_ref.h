// cpu_ref.h: CPU implementations.
//
// Two families:
//   *_ref       "gold" references. Straightforward loops, accumulate in double,
//               numerically safe formulations. Every GPU/emulated kernel is
//               checked against these. Clarity over speed.
//   *_cpu       float implementations of the same *algorithms* the GPU kernels
//               use (online softmax, Welford merge, tiled flash attention,
//               cache-blocked GEMM), plus deliberately naive variants used by the
//               numerics experiment. They let the algorithm be studied and timed
//               on a laptop.
// All matrices are row-major and contiguous.
#pragma once
#include <cstddef>

namespace gkl::cpu {

// ---- GEMM: C = alpha * A(MxK) * B(KxN) + beta * C; beta == 0 means C is not read.
void sgemm_ref(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
               float* C);
void sgemm_cpu_blocked(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                       float* C);

// ---- softmax over each row of a rows x cols matrix
void softmax_ref(const float* x, float* y, int rows, int cols);
void softmax_online_cpu(const float* x, float* y, int rows, int cols);
void softmax_unsafe_cpu(const float* x, float* y, int rows, int cols);  // no max subtraction

// ---- norms over each row; gamma/beta have length cols
void layernorm_ref(const float* x, const float* gamma, const float* beta, float* y, int rows,
                   int cols, float eps);
void layernorm_welford_cpu(const float* x, const float* gamma, const float* beta, float* y,
                           int rows, int cols, float eps);
void layernorm_onepass_cpu(const float* x, const float* gamma, const float* beta, float* y,
                           int rows, int cols, float eps);  // var = E[x^2] - E[x]^2, float
void rmsnorm_ref(const float* x, const float* gamma, float* y, int rows, int cols, float eps);

// ---- reductions
double reduce_sum_ref(const float* x, long long n);            // double accumulation
float reduce_sum_sequential_cpu(const float* x, long long n);  // float, left to right
// Mirrors the GPU tree: 'lanes' float accumulators strided like a grid-stride
// loop, then a pairwise tree over them.
float reduce_sum_tree_cpu(const float* x, long long n, int lanes);

// ---- attention: Q, K, V, O are [B, H, N, D]; scale = 1/sqrt(D)
void attention_ref(const float* Q, const float* K, const float* V, float* O, int B, int H, int N,
                   int D, bool causal);
// Tiled FlashAttention forward (online softmax over key tiles of size bc,
// query tiles of size br). Extra memory: O(br * bc + br * D) floats.
void attention_flash_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H,
                         int N, int D, bool causal, int br = 64, int bc = 32);
// Naive float attention that materializes the full N x N score matrix per head.
void attention_naive_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H,
                         int N, int D, bool causal);

}  // namespace gkl::cpu
