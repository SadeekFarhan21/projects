"""Environment throughput through the Python binding.

One env step is one tick of one game: every learner submits an action, every
scripted opponent acts, in a fresh random order. Learner actions are uniform
random and pre-generated so the timing covers the binding and the engine, not
numpy's RNG. Single threaded (the VecEnv steps its games sequentially).

Usage:
    uv run python scripts/bench_env.py
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

import figgie_arena as fa

OPPONENTS = {
    "random": ["random", "random", "random"],
    "scripted": ["passive", "taker", "random"],
    "bayes": ["bayes", "bayes", "bayes"],
}


def bench(num_envs: int, opps: list[str], seconds: float, seed: int = 0) -> dict:
    env = fa.VecEnv(num_envs, 1, opps, fa.Config(), seed, False)
    env.reset()
    rng = np.random.default_rng(seed)
    pool = np.stack([
        rng.integers(0, 8, size=(64, num_envs, 1)),
        rng.integers(0, 4, size=(64, num_envs, 1)),
        rng.integers(1, 40, size=(64, num_envs, 1)),
    ], axis=-1).astype(np.int32)
    calls = 0
    t0 = time.perf_counter()
    while True:
        for i in range(16):
            env.step(pool[(calls + i) % 64])
        calls += 16
        el = time.perf_counter() - t0
        if el >= seconds:
            break
    return {"num_envs": num_envs, "step_calls": calls, "seconds": el,
            "env_steps_per_sec": calls * num_envs / el, "calls_per_sec": calls / el,
            "episodes": int(env.episodes)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    rows = []
    for name, opps in OPPONENTS.items():
        for n in (1, 8, 64, 256):
            r = bench(n, opps, args.seconds)
            r["opponents"] = name
            rows.append(r)
            print(f"{name:9s} N={n:4d}  {r['env_steps_per_sec']:>12,.0f} env steps/s  "
                  f"({r['calls_per_sec']:,.0f} step calls/s)")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "env_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["opponents", "num_envs", "step_calls", "seconds",
                                          "env_steps_per_sec", "calls_per_sec", "episodes"])
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "env_bench.json", "w") as f:
        json.dump({"machine": platform.platform(), "processor": platform.processor(),
                   "threads": 1, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
