// reduce_utils.cuh: warp- and block-level reduction building blocks shared by
// softmax, the norms and the global reduction.
//
// Design
//   Level 1, warp: butterfly reduction with __shfl_xor_sync. log2(32) = 5
//   register-to-register exchanges, no shared memory and no __syncthreads.
//   Using XOR (not shfl_down) leaves the result in *every* lane, which saves a
//   broadcast when the caller needs it everywhere (softmax, norms).
//
//   Level 2, block: each warp writes one partial to a 32-entry shared array,
//   one barrier, warp 0 reduces the partials with the same butterfly, and the
//   result is broadcast through shared memory. Two barriers per call plus a
//   trailing one so the helper can be called several times in one kernel.
//
// Invariants: blockDim.x is a multiple of 32 and <= 1024 (so <= 32 warps);
// every thread of the block calls block_reduce_* (it contains barriers).
#pragma once
#include "gkl/cuda_compat.h"

namespace gkl::kernels {

constexpr unsigned kFullMask = 0xffffffffu;

template <class T>
GKL_DEVICE T warp_reduce_sum(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(kFullMask, v, off);
  return v;
}

template <class T>
GKL_DEVICE T warp_reduce_max(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(kFullMask, v, off));
  return v;
}

// Sum over the whole block; every thread receives the result.
template <class T>
GKL_DEVICE T block_reduce_sum(T v) {
  __shared__ T partial[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int nwarps = (blockDim.x + 31) / 32;
  v = warp_reduce_sum(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    T w = lane < nwarps ? partial[lane] : T(0);
    w = warp_reduce_sum(w);
    if (lane == 0) partial[0] = w;
  }
  __syncthreads();
  const T out = partial[0];
  __syncthreads();  // partial[] may be reused by the next call
  return out;
}

// Online-softmax state: running max m and running sum d of exp(x - m).
struct MaxSum {
  float m;
  float d;
};

// Merge two online-softmax states. The -inf guard matters: when both sides
// are empty (m = -inf), exp(-inf - -inf) would be NaN.
GKL_DEVICE MaxSum merge_maxsum(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.m, b.m);
  if (m == -INFINITY) return {m, 0.f};
  return {m, a.d * __expf(a.m - m) + b.d * __expf(b.m - m)};
}

GKL_DEVICE MaxSum warp_reduce_maxsum(MaxSum s) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    MaxSum o;
    o.m = __shfl_xor_sync(kFullMask, s.m, off);
    o.d = __shfl_xor_sync(kFullMask, s.d, off);
    s = merge_maxsum(s, o);
  }
  return s;
}

GKL_DEVICE MaxSum block_reduce_maxsum(MaxSum s) {
  __shared__ float pm[32];
  __shared__ float pd[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int nwarps = (blockDim.x + 31) / 32;
  s = warp_reduce_maxsum(s);
  if (lane == 0) { pm[warp] = s.m; pd[warp] = s.d; }
  __syncthreads();
  if (warp == 0) {
    MaxSum w = lane < nwarps ? MaxSum{pm[lane], pd[lane]} : MaxSum{-INFINITY, 0.f};
    w = warp_reduce_maxsum(w);
    if (lane == 0) { pm[0] = w.m; pd[0] = w.d; }
  }
  __syncthreads();
  const MaxSum out{pm[0], pd[0]};
  __syncthreads();
  return out;
}

// Welford state for mean/variance: count n, mean, and M2 = sum (x - mean)^2.
struct Welford {
  float n;
  float mean;
  float m2;
};

GKL_DEVICE Welford welford_push(Welford s, float x) {
  s.n += 1.f;
  const float delta = x - s.mean;
  s.mean += delta / s.n;
  s.m2 += delta * (x - s.mean);
  return s;
}

// Chan et al. parallel merge of two Welford states.
GKL_DEVICE Welford welford_merge(Welford a, Welford b) {
  const float n = a.n + b.n;
  if (n == 0.f) return {0.f, 0.f, 0.f};
  const float delta = b.mean - a.mean;
  const float wb = b.n / n;
  return {n, a.mean + delta * wb, a.m2 + b.m2 + delta * delta * a.n * wb};
}

GKL_DEVICE Welford warp_reduce_welford(Welford s) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    Welford o;
    o.n = __shfl_xor_sync(kFullMask, s.n, off);
    o.mean = __shfl_xor_sync(kFullMask, s.mean, off);
    o.m2 = __shfl_xor_sync(kFullMask, s.m2, off);
    s = welford_merge(s, o);
  }
  return s;
}

GKL_DEVICE Welford block_reduce_welford(Welford s) {
  __shared__ float pn[32];
  __shared__ float pmean[32];
  __shared__ float pm2[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int nwarps = (blockDim.x + 31) / 32;
  s = warp_reduce_welford(s);
  if (lane == 0) { pn[warp] = s.n; pmean[warp] = s.mean; pm2[warp] = s.m2; }
  __syncthreads();
  if (warp == 0) {
    Welford w = lane < nwarps ? Welford{pn[lane], pmean[lane], pm2[lane]} : Welford{0.f, 0.f, 0.f};
    w = warp_reduce_welford(w);
    if (lane == 0) { pn[0] = w.n; pmean[0] = w.mean; pm2[0] = w.m2; }
  }
  __syncthreads();
  const Welford out{pn[0], pmean[0], pm2[0]};
  __syncthreads();
  return out;
}

}  // namespace gkl::kernels
