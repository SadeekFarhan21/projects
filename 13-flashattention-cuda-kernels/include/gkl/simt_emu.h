// simt_emu.h: a tiny CPU emulator for the subset of CUDA that the kernels in
// kernels/*.cuh use. It exists so the *actual kernel source* can be executed
// and tested on a machine without an NVIDIA GPU.
//
// Execution model
//   * One launch = one pool of blockDim.x*y*z OS threads.
//   * Blocks run one after another; all threads of the current block run
//     concurrently, so __syncthreads() is a real barrier across real threads.
//   * `__shared__` is mapped to `static` (see cuda_compat.h). Because only one
//     block is live at a time, one static array per kernel instantiation is
//     exactly "one shared array per block".
//   * Warp shuffles go through a per-warp exchange buffer plus a per-warp
//     barrier, so every lane of a warp must reach the shuffle (the same rule as
//     the *_sync intrinsics on hardware).
//   * A thread that returns from the kernel early "drops" out of the block and
//     warp barriers, mirroring how exited threads no longer participate in
//     __syncthreads() on Volta and later.
//
// What it does NOT model: timing, memory coalescing, bank conflicts, caches,
// independent thread scheduling hazards, atomics, or more than one block in
// flight. It is a correctness tool, never a performance tool.
#pragma once

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

struct alignas(8) float2 { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

namespace gkl::emu {

// Reusable barrier that also supports permanently leaving (arrive_and_drop)
// and being re-armed between blocks (reset). std::barrier cannot be reset.
class Barrier {
 public:
  explicit Barrier(int n = 0) : expected_(n) {}
  void reset(int n) {
    std::lock_guard<std::mutex> lk(mu_);
    expected_ = n;
    arrived_ = 0;
  }
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lk(mu_);
    const uint64_t gen = gen_;
    if (++arrived_ == expected_) {
      release_locked();
    } else {
      cv_.wait(lk, [&] { return gen_ != gen; });
    }
  }
  void arrive_and_drop() {
    std::lock_guard<std::mutex> lk(mu_);
    --expected_;
    if (expected_ > 0 && arrived_ == expected_) release_locked();
  }

 private:
  void release_locked() {
    arrived_ = 0;
    ++gen_;
    cv_.notify_all();
  }
  std::mutex mu_;
  std::condition_variable cv_;
  int expected_ = 0;
  int arrived_ = 0;
  uint64_t gen_ = 0;
};

struct Warp {
  Barrier bar;
  uint64_t slot[32] = {};
};

struct BlockCtx {
  int nthreads = 0;
  Barrier block;
  std::vector<std::unique_ptr<Warp>> warps;
  std::vector<unsigned char> dyn_smem;

  int lanes_in_warp(int w) const { return std::min(32, nthreads - 32 * w); }
  void rearm() {
    block.reset(nthreads);
    for (size_t w = 0; w < warps.size(); ++w) warps[w]->bar.reset(lanes_in_warp(int(w)));
  }
};

inline thread_local dim3 tl_threadIdx;
inline thread_local dim3 tl_blockIdx;
inline thread_local int tl_linear_tid = 0;
inline thread_local BlockCtx* tl_ctx = nullptr;
inline dim3 g_blockDim;
inline dim3 g_gridDim;

inline std::mutex& launch_mutex() {
  static std::mutex m;
  return m;
}

inline void syncthreads() { tl_ctx->block.arrive_and_wait(); }
inline void syncwarp() { tl_ctx->warps[tl_linear_tid / 32]->bar.arrive_and_wait(); }
inline void* dyn_smem() { return tl_ctx->dyn_smem.data(); }

// Generic shuffle: every lane publishes its value, waits, reads lane `src`,
// waits again so the slot can be reused by the next shuffle.
template <class T>
T shfl_from(T v, int src_lane) {
  static_assert(sizeof(T) <= 8, "shuffle payload must fit in 64 bits");
  Warp& w = *tl_ctx->warps[tl_linear_tid / 32];
  const int lane = tl_linear_tid % 32;
  uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(T));
  w.slot[lane] = bits;
  w.bar.arrive_and_wait();
  T out;
  std::memcpy(&out, &w.slot[src_lane], sizeof(T));
  w.bar.arrive_and_wait();
  return out;
}

inline int lane_id() { return tl_linear_tid % 32; }

template <class Kernel, class... Args>
void launch(dim3 grid, dim3 block, size_t smem_bytes, Kernel kernel, Args... args) {
  std::lock_guard<std::mutex> guard(launch_mutex());  // one launch at a time
  const int nt = int(block.x * block.y * block.z);
  const int nb = int(grid.x * grid.y * grid.z);
  assert(nt > 0 && nt <= 1024 && "blockDim must be in [1, 1024]");
  g_blockDim = block;
  g_gridDim = grid;

  BlockCtx ctx;
  ctx.nthreads = nt;
  const int nwarps = (nt + 31) / 32;
  for (int w = 0; w < nwarps; ++w) ctx.warps.emplace_back(std::make_unique<Warp>());
  ctx.dyn_smem.assign(smem_bytes + 16, 0);
  ctx.rearm();
  Barrier between_blocks(nt);

  std::vector<std::thread> pool;
  pool.reserve(nt);
  for (int t = 0; t < nt; ++t) {
    pool.emplace_back([&, t] {
      tl_ctx = &ctx;
      tl_linear_tid = t;
      tl_threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
      for (int b = 0; b < nb; ++b) {
        tl_blockIdx = dim3(b % grid.x, (b / grid.x) % grid.y, b / (grid.x * grid.y));
        kernel(args...);
        // Leave this block's barriers so threads still inside can proceed.
        ctx.block.arrive_and_drop();
        ctx.warps[t / 32]->bar.arrive_and_drop();
        between_blocks.arrive_and_wait();
        if (t == 0) ctx.rearm();
        between_blocks.arrive_and_wait();
      }
      tl_ctx = nullptr;
    });
  }
  for (auto& th : pool) th.join();
}

}  // namespace gkl::emu
