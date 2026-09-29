"""Microbenchmarks for the M4 Pro roofline: achievable DRAM bandwidth and FMA throughput.

Writes results/roofline_micro.json and results/roofline_micro.csv.
"""

from __future__ import annotations

import csv

import numpy as np

import tcm
from tcm import _metal
from common import RESULTS, machine_info, write_json

SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void copy4(device const float4* a [[buffer(0)]], device float4* b [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { b[i] = a[i]; }

// each thread reads 4 float4 and writes one float: ~94% of traffic is reads
kernel void read4(device const float4* a [[buffer(0)]], device float* b [[buffer(1)]],
                  uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
  float4 s = a[i] + a[i + n] + a[i + 2 * n] + a[i + 3 * n];
  b[i] = s.x + s.y + s.z + s.w;
}

kernel void fill4(device float4* b [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  b[i] = float4(float(i));
}

// 8 independent float4 FMA chains per thread: 32 scalar FMAs per iteration
kernel void fma_f32(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float4 x0 = float4(i * 1e-9f), x1 = x0 + 1e-3f, x2 = x0 + 2e-3f, x3 = x0 + 3e-3f;
  float4 x4 = x0 + 4e-3f, x5 = x0 + 5e-3f, x6 = x0 + 6e-3f, x7 = x0 + 7e-3f;
  const float4 m = float4(0.999f), c = float4(1e-4f);
  for (uint k = 0; k < ITERS; ++k) {
    x0 = fma(x0, m, c); x1 = fma(x1, m, c); x2 = fma(x2, m, c); x3 = fma(x3, m, c);
    x4 = fma(x4, m, c); x5 = fma(x5, m, c); x6 = fma(x6, m, c); x7 = fma(x7, m, c);
  }
  float4 s = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
  out[i] = s.x + s.y + s.z + s.w;
}

kernel void fma_f16(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  half4 x0 = half4(half(i & 7) * 1e-3h), x1 = x0 + 1e-3h, x2 = x0 + 2e-3h, x3 = x0 + 3e-3h;
  half4 x4 = x0 + 4e-3h, x5 = x0 + 5e-3h, x6 = x0 + 6e-3h, x7 = x0 + 7e-3h;
  const half4 m = half4(0.999h), c = half4(1e-4h);
  for (uint k = 0; k < ITERS; ++k) {
    x0 = fma(x0, m, c); x1 = fma(x1, m, c); x2 = fma(x2, m, c); x3 = fma(x3, m, c);
    x4 = fma(x4, m, c); x5 = fma(x5, m, c); x6 = fma(x6, m, c); x7 = fma(x7, m, c);
  }
  half4 s = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
  out[i] = s.x + s.y + s.z + s.w;
}
"""

ITERS = 2048


def one_round(dev, src) -> list[dict]:
    rows = []

    def best_time(disp, reps):
        dev.run([disp], repeat=2)
        return min(dev.run([disp], repeat=reps)["gpu_s"] / reps for _ in range(5))

    # bandwidth ----------------------------------------------------------------
    for mb in (16, 64, 256, 1024):
        n_bytes = mb * 1024 * 1024
        n4 = n_bytes // 16
        a, b = dev.alloc(n_bytes), dev.alloc(n_bytes)
        tg = (256, 1, 1)
        t = best_time(_metal.Dispatch(dev.compile(src, "copy4"), [a, b], (n4 // 256, 1, 1), tg), 10)
        rows.append({"bench": "copy", "size_mb": mb, "seconds": t, "gbps": 2 * n_bytes / t / 1e9})
        t = best_time(_metal.Dispatch(dev.compile(src, "read4"), [a, b], (n4 // 4 // 256, 1, 1), tg), 10)
        rows.append({"bench": "read", "size_mb": mb, "seconds": t, "gbps": (n_bytes + n_bytes // 16) / t / 1e9})
        t = best_time(_metal.Dispatch(dev.compile(src, "fill4"), [b], (n4 // 256, 1, 1), tg), 10)
        rows.append({"bench": "write", "size_mb": mb, "seconds": t, "gbps": n_bytes / t / 1e9})
        del a, b
    # compute ------------------------------------------------------------------
    nthreads = 20 * 1024 * 64
    out = dev.alloc(nthreads * 4)
    for name in ("fma_f32", "fma_f16"):
        for tgs in (256, 1024):
            t = best_time(_metal.Dispatch(dev.compile(src, name), [out], (nthreads // tgs, 1, 1), (tgs, 1, 1)), 5)
            flops = 2.0 * 32 * ITERS * nthreads
            rows.append({"bench": name, "size_mb": tgs, "seconds": t, "gflops": flops / t / 1e9})
    return rows


def main(rounds: int = 4) -> None:
    """The GPU is shared, so run several rounds and keep the best time per row."""
    dev = tcm.device()
    src = SRC.replace("ITERS", f"{ITERS}u")
    info = machine_info()
    best: dict[tuple, dict] = {}
    for _ in range(rounds):
        for r in one_round(dev, src):
            k = (r["bench"], r["size_mb"])
            if k not in best or r["seconds"] < best[k]["seconds"]:
                best[k] = r
    rows = list(best.values())
    for r in rows:
        print(r)
    peak_bw = max(r["gbps"] for r in rows if r["bench"] == "copy")
    peak_read = max(r["gbps"] for r in rows if r["bench"] == "read")
    peak_f32 = max(r["gflops"] for r in rows if r["bench"] == "fma_f32")
    peak_f16 = max(r["gflops"] for r in rows if r["bench"] == "fma_f16")
    summary = {
        "machine": info,
        "rounds": rounds,
        "peak_copy_gbps": peak_bw,
        "peak_read_gbps": peak_read,
        "peak_fma_f32_gflops": peak_f32,
        "peak_fma_f16_gflops": peak_f16,
        "ridge_f32_flop_per_byte": peak_f32 / peak_bw,
        "rows": rows,
    }
    write_json("roofline_micro.json", summary)
    with open(RESULTS / "roofline_micro.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "size_mb", "seconds", "gbps", "gflops"])
        w.writeheader()
        w.writerows(rows)
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
