// attention.cuh: single-head scaled dot-product attention forward,
//   O = softmax(Q K^T * scale [+ causal mask]) V,  scale = 1 / sqrt(D).
// Q, K, V, O are [B, H, N, D] contiguous fp32; each (b, h) pair is independent.
//
// ----------------------------------------------------------------- naive
// Three launches that materialize the N x N score matrix S in global memory:
//   attn_scores_kernel : S = Q K^T * scale (masked entries = -inf)
//   softmax_online_kernel (softmax.cuh), in place on S
//   attn_pv_kernel     : O = P V
// Memory: S costs B*H*N*N*4 bytes, written once and read twice (softmax) plus
// once more (PV), on top of repeated uncached re-reads of K and V. At N = 4096,
// B*H = 32 that is 2 GB of scratch: the quadratic memory wall that motivates
// FlashAttention. Roofline: each score costs 4D flops (QK and PV) and about
// 20 bytes of S traffic (write, softmax read+write+read, PV read), so D/5
// flop/byte at best: bandwidth bound at D = 64.
//
// ----------------------------------------------------------------- flash
// flash_attn_fwd_kernel<D, BR>: FlashAttention-2 style tiling, fp32, no tensor
// cores. One block = 4 warps = BR query rows of one (b, h). Each warp owns
// BR/4 rows. The block streams K and V through shared memory in tiles of
// BC = 32 keys; S and P never leave registers.
//
//   shared memory: Qs [BR][D] (pre-scaled by `scale`), Ks [BC][D+1], Vs [BC][D]
//   per row state : running max m, running sum l, output accumulator o[D/32]
//                   per lane (lane owns dims lane, lane+32, ...)
//
//   per K/V tile, per owned row:
//     lane j computes s_j = q . k_j                (Ks padded to D+1 so the 32
//                                                   lanes hit 32 different banks)
//     m_new = max(m, warp_max(s_j))                (5 xor shuffles)
//     p_j = exp(s_j - m_new);  alpha = exp(m - m_new)
//     l = l * alpha + warp_sum(p_j)                (5 xor shuffles)
//     o = o * alpha + sum_j p_j * V[j, :]          (p_j broadcast by __shfl_sync;
//                                                   Vs reads are 32 consecutive
//                                                   words, conflict free)
//   finally O = o / l.
//
// Memory: HBM traffic is Q + O once, plus K and V once per query block, i.e.
// (N / BR) passes over K and V instead of N^2 score traffic. Extra shared
// memory is O(BR*D + BC*D); no N^2 buffer at all.
// Causal: blocks stop loading K/V tiles past their last query row, which halves
// the work; the diagonal tile is masked per element.
// Sizes: D = 64 uses BR = 64 (32,896 B of smem); D = 128 uses BR = 32
// (49,280 B, 128 bytes over the 49,152 B static limit, hence dynamic smem plus
// the opt-in attribute in the launcher). D = 32 uses BR = 64.
// Roofline: 4*N^2*D flops against ~(2 + 2N/BR)*N*D*4 bytes per head, e.g.
// ~32 flop/byte at D = 64, BR = 64, so it is compute bound on fp32 CUDA cores.
// Realistic target for this fp32 SIMT version: 30 to 50 percent of fp32 peak;
// the per-row serial shuffle chain (32 broadcasts per tile) is the known
// limiter. Tensor-core MMA versions are a later milestone.
#pragma once
#include "kernels/reduce_utils.cuh"

namespace gkl::kernels {

constexpr int kAttnBC = 32;       // keys per K/V tile = warp size (one key per lane)
constexpr int kAttnWarps = 4;     // warps per flash block

// S[bh, i, j] for a 16 x 16 tile of (i, j); threadIdx.x -> j.
__global__ void attn_scores_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                   float* __restrict__ S, int N, int D, float scale, int causal) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t bh = blockIdx.z;
  if (i >= N || j >= N) return;
  const float* q = Q + (bh * N + i) * D;
  const float* k = K + (bh * N + j) * D;
  float s = 0.f;
  for (int d = 0; d < D; ++d) s += q[d] * k[d];
  S[(bh * N + i) * N + j] = (causal && j > i) ? -INFINITY : s * scale;
}

// O[bh, i, d] = sum_j P[bh, i, j] V[bh, j, d]; threadIdx.x -> d.
__global__ void attn_pv_kernel(const float* __restrict__ P, const float* __restrict__ V,
                               float* __restrict__ O, int N, int D) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t bh = blockIdx.z;
  if (i >= N || d >= D) return;
  const float* p = P + (bh * N + i) * N;
  const float* v = V + bh * N * D;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += p[j] * v[size_t(j) * D + d];
  O[(bh * N + i) * D + d] = acc;
}

template <int D, int BR>
constexpr size_t flash_smem_bytes() {
  return sizeof(float) * (size_t(BR) * D + size_t(kAttnBC) * (D + 1) + size_t(kAttnBC) * D);
}

template <int D, int BR>
__global__ void __launch_bounds__(kAttnWarps * 32)
    flash_attn_fwd_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                          const float* __restrict__ V, float* __restrict__ O, int N, float scale,
                          int causal) {
  static_assert(D % 32 == 0, "D must be a multiple of 32 (lanes own D/32 dims each)");
  static_assert(BR % kAttnWarps == 0, "BR must split evenly across warps");
  constexpr int BC = kAttnBC;
  constexpr int RPW = BR / kAttnWarps;  // query rows per warp
  constexpr int DPL = D / 32;           // output dims per lane
  constexpr int KSTRIDE = D + 1;        // padding kills the 32-way bank conflict on Ks

  GKL_DYN_SMEM(float, smem);
  float* Qs = smem;
  float* Ks = Qs + BR * D;
  float* Vs = Ks + BC * KSTRIDE;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const size_t bh = blockIdx.y;
  const int q0 = blockIdx.x * BR;
  const float* Qb = Q + bh * size_t(N) * D;
  const float* Kb = K + bh * size_t(N) * D;
  const float* Vb = V + bh * size_t(N) * D;
  float* Ob = O + bh * size_t(N) * D;

  for (int i = tid; i < BR * D; i += blockDim.x) {
    const int r = i / D, c = i % D;
    Qs[i] = (q0 + r < N) ? Qb[size_t(q0 + r) * D + c] * scale : 0.f;
  }

  float m[RPW], l[RPW], o[RPW][DPL];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) o[r][e] = 0.f;
  }

  const int kv_end = causal ? min(N, q0 + BR) : N;
  for (int k0 = 0; k0 < kv_end; k0 += BC) {
    __syncthreads();  // previous tile fully consumed (and Qs visible on iteration 0)
    for (int i = tid; i < BC * D; i += blockDim.x) {
      const int j = i / D, c = i % D;
      const bool in = k0 + j < N;
      Ks[j * KSTRIDE + c] = in ? Kb[size_t(k0 + j) * D + c] : 0.f;
      Vs[j * D + c] = in ? Vb[size_t(k0 + j) * D + c] : 0.f;
    }
    __syncthreads();

    const int kj = k0 + lane;  // the key this lane scores
#pragma unroll 1
    for (int r = 0; r < RPW; ++r) {
      // Rows past N still run (control flow must stay warp uniform for the
      // shuffles); their results are simply never stored.
      const int qi = q0 + warp * RPW + r;
      const float* q = Qs + (warp * RPW + r) * D;
      float s = 0.f;
#pragma unroll 8
      for (int c = 0; c < D; ++c) s += q[c] * Ks[lane * KSTRIDE + c];
      const bool valid = kj < N && (!causal || kj <= qi);
      s = valid ? s : -INFINITY;

      const float m_new = fmaxf(m[r], warp_reduce_max(s));
      // m_new = -inf means no valid key yet for this row; keep everything 0.
      const float p = m_new == -INFINITY ? 0.f : __expf(s - m_new);
      const float alpha = m[r] == -INFINITY ? 0.f : __expf(m[r] - m_new);
      l[r] = l[r] * alpha + warp_reduce_sum(p);
#pragma unroll
      for (int e = 0; e < DPL; ++e) o[r][e] *= alpha;
      for (int j = 0; j < BC; ++j) {
        const float pj = __shfl_sync(kFullMask, p, j);
#pragma unroll
        for (int e = 0; e < DPL; ++e) o[r][e] += pj * Vs[j * D + lane + 32 * e];
      }
      m[r] = m_new;
    }
  }

#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int qi = q0 + warp * RPW + r;
    if (qi >= N) continue;
    const float inv = 1.f / l[r];
#pragma unroll
    for (int e = 0; e < DPL; ++e) Ob[size_t(qi) * D + lane + 32 * e] = o[r][e] * inv;
  }
}

}  // namespace gkl::kernels
