"""Shared helpers for benchmarks: results paths, machine info, baseline timers."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def gpu_utilization() -> int | None:
    """Device Utilization % reported by the GPU driver (other processes included)."""
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"], capture_output=True, text=True, timeout=5).stdout
        i = out.find('"Device Utilization %"=')
        if i < 0:
            return None
        return int(out[i:].split("=", 1)[1].split(",", 1)[0].split("}", 1)[0])
    except Exception:
        return None


def machine_info() -> dict:
    import tcm

    d = tcm.device()
    try:
        load = os.getloadavg()
    except OSError:
        load = (float("nan"),) * 3
    return {
        "device": d.name,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "loadavg_1m": round(load[0], 2),
        "gpu_util_pct_before_run": gpu_utilization(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def write_json(name: str, obj) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1))
    return p


def time_mlx(fn, args, repeat: int) -> float:
    """Wall seconds per call; `repeat` independent calls are built lazily then evaluated together."""
    import mlx.core as mx

    for _ in range(3):
        mx.eval(fn(*args))
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        outs = [fn(*args) for _ in range(repeat)]
        mx.eval(outs)
        best = min(best, (time.perf_counter() - t0) / repeat)
        del outs
    return best


def time_torch(fn, args, repeat: int) -> float:
    import torch

    for _ in range(3):
        fn(*args)
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn(*args)
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - t0) / repeat)
    return best


def time_tcm(prog, repeat: int) -> tuple[float, float]:
    """(wall seconds per call, GPU seconds per call); `repeat` calls in one command buffer."""
    prog.run(repeat=3)
    best_w, best_g = float("inf"), float("inf")
    for _ in range(3):
        r = prog.run(repeat=repeat)
        best_w = min(best_w, r["wall_s"] / repeat)
        best_g = min(best_g, r["gpu_s"] / repeat)
    return best_w, best_g


def repeats_for(seconds_per_call: float, target: float = 0.15, lo: int = 3, hi: int = 500) -> int:
    return int(np.clip(target / max(seconds_per_call, 1e-7), lo, hi))
