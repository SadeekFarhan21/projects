// sgemm.cuh: C = alpha * A * B + beta * C, row-major, fp32.
//   A is M x K, B is K x N, C is M x N. Any M, N, K >= 1 (edges are guarded).
//   BLAS convention: when beta == 0, C is not read (so it may hold NaNs).
//
// Three kernels, each one step up the memory hierarchy.
//
// 1. sgemm_naive_kernel: one thread per output element.
//    Memory: every thread streams a full row of A and a full column of B from
//    global memory (through L1/L2). threadIdx.x maps to the column so a warp's
//    loads of B and stores of C are coalesced; the A load is a warp-wide
//    broadcast. Per output element: 2K loads for 2K flops, so the arithmetic
//    intensity against DRAM is set entirely by how well L1/L2 catch reuse.
//    Expected: memory/latency bound, a few percent of fp32 peak.
//
// 2. sgemm_tiled_kernel<TILE>: classic shared-memory tiling.
//    A TILE x TILE block cooperatively stages a TILE x TILE tile of A and of
//    B in shared memory, then each thread does TILE FMAs from shared memory.
//    Global traffic per block per K-step: 2*TILE^2 floats for 2*TILE^3 flops,
//    i.e. intensity grows by a factor TILE (32 -> ~8 flop/byte ideal).
//    Bank conflicts: As[ty][k] is a broadcast within a warp (a warp is one
//    ty when TILE = 32); Bs[k][tx] is 32 consecutive words, conflict free.
//    Expected: now bound by shared-memory bandwidth, because every FMA needs
//    two shared loads (1 FMA : 2 LDS). Roughly 15 to 25 percent of peak.
//
// 3. sgemm_regblock_kernel<BM,BN,BK,TM,TN>: 2D register blocking.
//    Block tile BM x BN = 128 x 128, K-step BK = 8, each of 256 threads owns a
//    TM x TN = 8 x 8 micro-tile of C held in registers. Per k, a thread loads
//    TM values of A and TN of B from shared memory into registers and does
//    TM*TN = 64 FMAs: 16 LDS per 64 FMA instead of 2 per 1. A is stored
//    transposed in shared memory (As[k][m]) so the per-thread A fragment is
//    contiguous. Global intensity per block: (2*BM*BN*BK) flops over
//    (BM+BN)*BK*4 bytes = 32 flop/byte, above the L4 and H100 fp32 ridge
//    points, so this kernel should be compute bound for large sizes.
//    Known costs left on the table for v1: the transposed store into As is
//    bank conflicted, the Bs fragment read is 4-way conflicted (lanes are TN = 8
//    words apart), there is no float4 vectorization and no double buffering.
//    Expected: 50 to 70 percent of cuBLAS on large square problems.
#pragma once
#include "gkl/cuda_compat.h"

namespace gkl::kernels {

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                   const float* __restrict__ B, float beta, float* __restrict__ C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[size_t(row) * K + k] * B[size_t(k) * N + col];
  float* c = &C[size_t(row) * N + col];
  *c = beta == 0.f ? alpha * acc : alpha * acc + beta * *c;
}

template <int TILE>
__global__ void sgemm_tiled_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                                   const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  // No early return: every thread must reach the barriers even if its output
  // element is out of range. Out-of-range tile entries are zero filled.
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int ka = k0 + tx, kb = k0 + ty;
    As[ty][tx] = (row < M && ka < K) ? A[size_t(row) * K + ka] : 0.f;
    Bs[ty][tx] = (kb < K && col < N) ? B[size_t(kb) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) {
    float* c = &C[size_t(row) * N + col];
    *c = beta == 0.f ? alpha * acc : alpha * acc + beta * *c;
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_regblock_kernel(int M, int N, int K, float alpha, const float* __restrict__ A,
                          const float* __restrict__ B, float beta, float* __restrict__ C) {
  constexpr int kThreadsN = BN / TN;
  constexpr int kThreads = (BM / TM) * kThreadsN;
  __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
  __shared__ float Bs[BK * BN];  // Bs[k * BN + n]

  const int tid = threadIdx.x;
  const int trow = tid / kThreadsN;  // which TM-row strip of the block tile
  const int tcol = tid % kThreadsN;  // which TN-column strip
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  float a_frag[TM];
  float b_frag[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Stage A (BM x BK) transposed and B (BK x BN). Consecutive threads read
    // consecutive addresses in global memory for both.
    for (int i = tid; i < BM * BK; i += kThreads) {
      const int m = i / BK, k = i % BK;
      const int gr = block_row + m, gk = k0 + k;
      As[k * BM + m] = (gr < M && gk < K) ? A[size_t(gr) * K + gk] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += kThreads) {
      const int k = i / BN, n = i % BN;
      const int gk = k0 + k, gc = block_col + n;
      Bs[k * BN + n] = (gk < K && gc < N) ? B[size_t(gk) * N + gc] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) a_frag[i] = As[k * BM + trow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b_frag[j] = Bs[k * BN + tcol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_frag[i] * b_frag[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + trow * TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = block_col + tcol * TN + j;
      if (c >= N) continue;
      float* out = &C[size_t(r) * N + c];
      *out = beta == 0.f ? alpha * acc[i][j] : alpha * acc[i][j] + beta * *out;
    }
  }
}

}  // namespace gkl::kernels
