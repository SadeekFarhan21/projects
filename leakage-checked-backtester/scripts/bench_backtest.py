"""Backtest throughput benchmark: strategy-days per second across engines and sizes.

A strategy-day is one day of one strategy over the whole cross-section of N assets
(mark to market, drift, trade, cost). Also reported as asset-days per second.

Usage: uv run python scripts/bench_backtest.py [--quick] [--real]
Writes results/bench_backtest.{csv,json,png}.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numba
import numpy as np
import polars as pl

from qrp.backtest.engine import CostModel, run_backtest
from qrp.panel import synthetic_panel

RESULTS = Path(__file__).resolve().parents[1] / "results"


def timeit(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_case(engine, S, T, N, threads, repeats, costs):
    p = synthetic_panel(T=T, N=N, seed=0, missing=0.01)
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1.0 / N, (S, T, N))
    if engine.startswith("numba"):
        numba.set_num_threads(threads)
        run_backtest(W[:1, :10], p["close"][:10], costs, engine="numba")  # JIT warmup
    eng = "numba" if engine.startswith("numba") else engine
    sec = timeit(lambda: run_backtest(W, p["close"], costs, delay=1, engine=eng), repeats)
    return {"engine": engine, "threads": threads, "S": S, "T": T, "N": N, "seconds": sec,
            "strategy_days_per_s": S * T / sec, "asset_days_per_s": S * T * N / sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--real", action="store_true", help="also time the real Binance panel")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    costs = CostModel(10, 5)
    ncpu = numba.config.NUMBA_NUM_THREADS
    T = 1825  # five years of daily crypto bars
    rows = []
    load_before = os.getloadavg()

    # the pure-Python oracle only at a small size, it is ~1000x slower
    rows.append(bench_case("reference", 1, 365, 100, 1, 1, costs))
    print(rows[-1], flush=True)

    Ns = [100, 300] if a.quick else [100, 300, 1000]
    Ss = [1, 16] if a.quick else [1, 16, 128]
    for N in Ns:
        for S in Ss:
            if S * T * N > 2.5e8:
                continue
            for engine, threads in (("numpy", 1), ("numba_1t", 1), ("numba_mt", ncpu)):
                rep = 1 if engine == "numpy" and S * N >= 16 * 1000 else 5
                rows.append(bench_case(engine, S, T, N, threads, rep, costs))
                r = rows[-1]
                print(f"{engine:9s} S={S:4d} N={N:5d} {r['seconds']:.4f}s "
                      f"{r['strategy_days_per_s']:,.0f} strat-days/s", flush=True)

    # single strategy with the forward-filled market precomputed (the sweep use case)
    from qrp.backtest.engine import prepare_market
    for N in Ns:
        p = synthetic_panel(T=T, N=N, seed=0, missing=0.01)
        W = np.random.default_rng(0).normal(0, 1.0 / N, (1, T, N))
        mk = prepare_market(p["close"])
        numba.set_num_threads(1)
        sec = timeit(lambda: run_backtest(W, p["close"], costs, market=mk), 5)
        rows.append({"engine": "numba_1t_cached_market", "threads": 1, "S": 1, "T": T, "N": N,
                     "seconds": sec, "strategy_days_per_s": T / sec, "asset_days_per_s": T * N / sec})
        print(rows[-1], flush=True)

    if a.real:
        from qrp.data.store import BarStore
        from qrp.panel import Panel
        panel = Panel.from_long(BarStore("data").load())
        Tr, Nr = panel.shape
        rng = np.random.default_rng(0)
        for S in (1, 64):
            W = rng.normal(0, 1.0 / Nr, (S, Tr, Nr))
            numba.set_num_threads(ncpu)
            run_backtest(W[:1, :10], panel["close"][:10], costs)
            sec = timeit(lambda: run_backtest(W, panel["close"], costs), 5)
            rows.append({"engine": "numba_mt_real", "threads": ncpu, "S": S, "T": Tr, "N": Nr,
                         "seconds": sec, "strategy_days_per_s": S * Tr / sec,
                         "asset_days_per_s": S * Tr * Nr / sec})
            print(rows[-1], flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "bench_backtest.csv")
    meta = {"machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numba_threads": ncpu, "python": platform.python_version(),
            "numpy": np.__version__, "numba": numba.__version__, "T": T,
            "costs": costs.__dict__, "timing": "best of 5 (1 for the largest numpy cases)",
            "loadavg_before": load_before, "loadavg_after": os.getloadavg(),
            "note": "machine shared with other concurrent jobs; see loadavg",
            "rows": rows}
    (RESULTS / "bench_backtest.json").write_text(json.dumps(meta, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.2 * len(Ns), 3.6), sharey=True)
    for ax, N in zip(np.atleast_1d(axes), Ns):
        for eng, color in (("numpy", "#888888"), ("numba_1t", "#4C72B0"), ("numba_mt", "#C44E52")):
            d = df.filter((pl.col("engine") == eng) & (pl.col("N") == N)).sort("S")
            ax.plot(d["S"], d["strategy_days_per_s"], "o-", color=color, label=eng)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"N = {N} assets, T = {T} days")
        ax.set_xlabel("strategies per call (S)")
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel("strategy-days per second")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.suptitle("Backtest throughput on Apple M4 Pro (higher is better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "bench_backtest.png", dpi=150)
    print(f"wrote {RESULTS}/bench_backtest.csv .json .png")


if __name__ == "__main__":
    main()
