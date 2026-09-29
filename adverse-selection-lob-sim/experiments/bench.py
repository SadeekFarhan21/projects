"""Throughput benchmarks: raw order book ops/s and full-simulation events/s.

Writes results/bench.json. Usage: uv run python experiments/bench.py
Note: numbers are single-threaded wall-clock on whatever else the machine is
doing at the time; repeat counts and medians are reported to show the spread.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from marketsim import BUY, SELL, OrderBook, SimConfig, run

OUT = Path(__file__).resolve().parents[1] / "results" / "bench.json"


def book_workload(n_ops: int, seed: int) -> list[tuple]:
    """Pre-generate ops so RNG cost is not timed: 60% limit, 15% market, 25% cancel."""
    rng = np.random.default_rng(seed)
    kinds = rng.random(n_ops)
    sides = np.where(rng.random(n_ops) < 0.5, BUY, SELL)
    offs = rng.geometric(0.3, n_ops)
    qtys = rng.integers(1, 5, n_ops)
    picks = rng.random(n_ops)
    ops = []
    for k, s, o, q, p in zip(kinds, sides, offs, qtys, picks):
        if k < 0.60:
            # Passive-ish around 10000: bids below, asks above, some crossing.
            price = 10_000 - int(o) + 2 if s == BUY else 10_000 + int(o) - 2
            ops.append(("L", int(s), price, int(q)))
        elif k < 0.75:
            ops.append(("M", int(s), 0, int(q)))
        else:
            ops.append(("C", 0, 0, float(p)))
    return ops


def bench_book(n_ops: int = 500_000, repeats: int = 5) -> dict:
    ops = book_workload(n_ops, 0)
    rates, n_fills, depth = [], 0, 0
    for _ in range(repeats):
        book = OrderBook()
        live: list[int] = []
        fills = 0
        t0 = time.perf_counter()
        for kind, side, price, q in ops:
            if kind == "L":
                oid, f = book.submit_limit(0, side, price, q)
                fills += len(f)
                if oid is not None:
                    live.append(oid)
            elif kind == "M":
                fills += len(book.submit_market(0, side, q))
            elif live:
                i = int(q * len(live))
                live[i], live[-1] = live[-1], live[i]
                book.cancel(live.pop())  # may already be filled: returns False
        dt = time.perf_counter() - t0
        rates.append(n_ops / dt)
        n_fills, depth = fills, len(book)
    return {"n_ops": n_ops, "repeats": repeats, "ops_per_s_median": statistics.median(rates),
            "ops_per_s_all": rates, "fills": n_fills, "resting_orders_at_end": depth}


def bench_sim(repeats: int = 3) -> dict:
    cfg = SimConfig(t_end=20_000.0, informed_frac=0.2)
    rows = []
    for seed in range(repeats):
        res = run(cfg.with_(seed=seed))
        c = res.counters
        rows.append({"seed": seed, "events": c["events"], "wall_s": c["wall_s"], "events_per_s": c["events_per_s"],
                     "trades": c["n_trades"]})
    return {"config": "SimConfig(t_end=20000, informed_frac=0.2)", "runs": rows,
            "events_per_s_median": statistics.median(r["events_per_s"] for r in rows)}


def main() -> None:
    out = {"machine": platform.platform(), "python": platform.python_version(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "load_avg_1_5_15_at_start": os.getloadavg(),
           "order_book": bench_book(), "simulation": bench_sim()}
    out["load_avg_1_5_15_at_end"] = os.getloadavg()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    ob, sim = out["order_book"], out["simulation"]
    print(f"order book: {ob['ops_per_s_median']:,.0f} ops/s median over {ob['repeats']} x {ob['n_ops']:,} ops "
          f"(min {min(ob['ops_per_s_all']):,.0f}, max {max(ob['ops_per_s_all']):,.0f}); {ob['fills']:,} fills")
    for r in sim["runs"]:
        print(f"simulation seed {r['seed']}: {r['events']:,} events in {r['wall_s']:.2f}s "
              f"= {r['events_per_s']:,.0f} events/s, {r['trades']:,} trades")
    print(f"load average at start/end: {out['load_avg_1_5_15_at_start']} / {out['load_avg_1_5_15_at_end']}")
    print(f"simulation median: {sim['events_per_s_median']:,.0f} events/s")


if __name__ == "__main__":
    main()
