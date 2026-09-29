// reduce.cuh: sum of an n-element fp32 vector.
//
// Two launches, no atomics:
//   pass 1: reduce_sum_partial_kernel, G blocks x 256 threads. Each thread
//           accumulates a grid-stride slice in a register, then the block
//           reduces with warp shuffles (5 xor steps) and one shared-memory
//           hop across warps. Block b writes partials[b].
//   pass 2: the same kernel with one block over the G partials.
//
// Why two passes and not atomicAdd: the result is bitwise deterministic for a
// given n and G (float addition order is fixed), which makes GPU-vs-CPU tests
// and benchmark comparisons reproducible. The cost is one extra tiny launch
// (~3 to 5 us), which only matters for small n.
//
// Memory: grid-stride loads are fully coalesced (warp reads 128 contiguous
// bytes per iteration). The launcher uses G = min(ceil(n / 1024), 1024), enough
// blocks to fill every SM several times over (L4 has 58, H100 SXM 132) while
// letting each thread sum several elements before any synchronization.
// Roofline: 1 flop per 4 bytes, 0.25 flop/byte, purely DRAM bound. Target:
// 80 to 90 percent of peak DRAM bandwidth for n >= 16M. A float4-vectorized
// load is the obvious next step (fewer load instructions, same bytes).
#pragma once
#include "kernels/reduce_utils.cuh"

namespace gkl::kernels {

__global__ void reduce_sum_partial_kernel(const float* __restrict__ x, long long n,
                                          float* __restrict__ partials) {
  float acc = 0.f;
  const long long stride = (long long)blockDim.x * gridDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) acc += x[i];
  acc = block_reduce_sum(acc);
  if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}

}  // namespace gkl::kernels
