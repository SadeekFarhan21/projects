// cuda_compat.h: lets kernels/*.cuh compile either as real CUDA (nvcc or
// clang -x cuda) or as plain C++ on top of the SIMT emulator (simt_emu.h).
//
// Kernel code only uses the macros below for the three things plain CUDA
// syntax cannot express in C++: launches, dynamic shared memory, and the
// "opt in to >48 KB shared memory" attribute call. Everything else
// (threadIdx, __syncthreads, __shfl_xor_sync, __shared__ arrays) is written
// exactly as it would be for nvcc.
#pragma once

#include <cmath>
#include <cstddef>

#if defined(__CUDACC__)
// ---------------------------------------------------------------- real CUDA
#include <cuda_runtime.h>

#define GKL_DEVICE __device__ __forceinline__
#define GKL_DYN_SMEM(T, name) extern __shared__ __align__(16) unsigned char gkl_dyn_smem_raw[]; \
  T* name = reinterpret_cast<T*>(gkl_dyn_smem_raw)
#define GKL_LAUNCH(kernel, grid, block, smem, ...) kernel<<<(grid), (block), (smem)>>>(__VA_ARGS__)
#define GKL_ALLOW_BIG_SMEM(kernel, bytes) \
  cudaFuncSetAttribute((kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes))

#else
// ---------------------------------------------------------------- emulator
#include "gkl/simt_emu.h"

#define GKL_EMULATED 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __launch_bounds__(...)
#define GKL_DEVICE inline

#define threadIdx (::gkl::emu::tl_threadIdx)
#define blockIdx (::gkl::emu::tl_blockIdx)
#define blockDim (::gkl::emu::g_blockDim)
#define gridDim (::gkl::emu::g_gridDim)

#define GKL_DYN_SMEM(T, name) T* name = reinterpret_cast<T*>(::gkl::emu::dyn_smem())
#define GKL_LAUNCH(kernel, grid, block, smem, ...) \
  ::gkl::emu::launch((grid), (block), (smem), (kernel), __VA_ARGS__)
#define GKL_ALLOW_BIG_SMEM(kernel, bytes) ((void)(kernel), (void)(bytes))

inline void __syncthreads() { ::gkl::emu::syncthreads(); }
inline void __syncwarp(unsigned = 0xffffffffu) { ::gkl::emu::syncwarp(); }

// Semantics follow the CUDA programming guide: `width` splits the warp into
// independent segments; __shfl_down_sync returns the caller's own value when
// the source lane would fall outside its segment.
template <class T>
T __shfl_sync(unsigned, T v, int src, int width = 32) {
  const int lane = ::gkl::emu::lane_id();
  return ::gkl::emu::shfl_from(v, (lane & ~(width - 1)) + (src & (width - 1)));
}
template <class T>
T __shfl_xor_sync(unsigned, T v, int mask, int width = 32) {
  const int lane = ::gkl::emu::lane_id();
  int src = lane ^ mask;
  if ((src & ~(width - 1)) != (lane & ~(width - 1))) src = lane;
  return ::gkl::emu::shfl_from(v, src);
}
template <class T>
T __shfl_down_sync(unsigned, T v, unsigned delta, int width = 32) {
  const int lane = ::gkl::emu::lane_id();
  int src = lane + int(delta);
  if ((src & ~(width - 1)) != (lane & ~(width - 1))) src = lane;
  return ::gkl::emu::shfl_from(v, src);
}

inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
template <class T>
inline T __ldg(const T* p) { return *p; }
#endif
