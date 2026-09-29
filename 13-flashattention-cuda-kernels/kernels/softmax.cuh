// softmax.cuh: row-wise softmax y[r, :] = exp(x[r, :] - max) / sum exp(x[r, :] - max)
// over a rows x cols row-major matrix. In-place (x == y) is allowed.
//
// 1. softmax_naive_kernel: one thread per row, three passes (max, sum, write).
//    Memory: adjacent threads touch addresses `cols` floats apart, so every
//    load is its own 32-byte sector: uncoalesced. 3 reads + 1 write of the
//    row. Kept as a teaching baseline; expected to reach only a small fraction
//    of DRAM bandwidth.
//
// 2. softmax_online_kernel: one block (256 threads) per row, online softmax
//    (Milakov and Gimelshein, 2018). Each thread walks the row with a stride of
//    blockDim.x (coalesced), keeping a running (max m, sum d) pair and
//    rescaling d whenever m grows. The per-thread pairs are merged with warp
//    shuffles, then across warps through shared memory. A second pass writes
//    exp(x - m) / d. DRAM traffic: 2 reads + 1 write per element instead of 3 + 1
//    for the safe three-pass version; the second read is often an L2 hit when
//    the row fits.
//    Roofline: about 3 flop (sub, exp, mul) per 12 bytes, 0.25 flop/byte, far
//    left of the ridge on any GPU. Target: 70 to 85 percent of DRAM bandwidth
//    for rows of at least ~1K elements. Short rows (< 256) waste most of the
//    block; a warp-per-row variant is on the roadmap.
#pragma once
#include "kernels/reduce_utils.cuh"

namespace gkl::kernels {

__global__ void softmax_naive_kernel(const float* __restrict__ x, float* __restrict__ y, int rows,
                                     int cols) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;
  float m = -INFINITY;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
  float d = 0.f;
  for (int c = 0; c < cols; ++c) d += __expf(xr[c] - m);
  const float inv = 1.f / d;
  for (int c = 0; c < cols; ++c) yr[c] = __expf(xr[c] - m) * inv;
}

// Not __restrict__: the attention path calls this in place.
__global__ void softmax_online_kernel(const float* x, float* y, int rows, int cols) {
  const int r = blockIdx.x;
  if (r >= rows) return;  // block-uniform, so safe with the barriers below
  const float* xr = x + size_t(r) * cols;
  float* yr = y + size_t(r) * cols;

  MaxSum s{-INFINITY, 0.f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = xr[c];
    if (v > s.m) {
      s.d = s.d * __expf(s.m - v) + 1.f;  // exp(-inf) = 0 on the first element
      s.m = v;
    } else if (s.m != -INFINITY) {
      // Guard: v = s.m = -inf (a masked attention score seen before any
      // finite one) would otherwise add exp(NaN).
      s.d += __expf(v - s.m);
    }
  }
  s = block_reduce_maxsum(s);
  const float inv = 1.f / s.d;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = __expf(xr[c] - s.m) * inv;
}

}  // namespace gkl::kernels
