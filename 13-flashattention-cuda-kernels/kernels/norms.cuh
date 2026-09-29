// norms.cuh: LayerNorm and RMSNorm over the last dimension of a rows x cols
// row-major matrix. gamma/beta have length cols.
//
//   LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta, var biased (1/n),
//              matching torch.nn.functional.layer_norm.
//   RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma.
//
// Layout: one block of 256 threads per row. Pass 1 reads the row with a
// blockDim.x stride (coalesced) and reduces statistics; pass 2 re-reads the
// row (usually an L2 hit for rows up to tens of KB) and writes y.
//
// Statistics: LayerNorm uses per-thread Welford accumulation merged with the
// Chan et al. formula (warp shuffles, then shared memory). The textbook one-pass
// alternative, var = E[x^2] - E[x]^2, cancels catastrophically when |mean| is
// large relative to the standard deviation; tests/test_ref.cpp and
// results/numerics.csv measure that failure. Two-pass (mean, then
// sum (x - mean)^2) is also exact enough but costs a third read of the row.
//
// Roofline: ~8 flop per element for 8 bytes of compulsory traffic (read x,
// write y; gamma and beta are reused across rows and stay in L2). About
// 1 flop/byte: bandwidth bound on both L4 and H100. Target: 70 to 85 percent of
// DRAM bandwidth for cols >= 1024.
#pragma once
#include "kernels/reduce_utils.cuh"

namespace gkl::kernels {

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, float* __restrict__ y, int rows,
                                 int cols, float eps) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;

  Welford s{0.f, 0.f, 0.f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s = welford_push(s, xr[c]);
  s = block_reduce_welford(s);
  const float mean = s.mean;
  const float rstd = rsqrtf(s.m2 / float(cols) + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma,
                               float* __restrict__ y, int rows, int cols, float eps) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;

  float ss = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) ss += xr[c] * xr[c];
  ss = block_reduce_sum(ss);
  const float rrms = rsqrtf(ss / float(cols) + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * rrms * gamma[c];
}

}  // namespace gkl::kernels
