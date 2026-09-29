"""Ablation for the two market-maker inventory fixes (see DEVLOG, Problems).

Grid: noise_k in {0, 0.1} x mm_inventory_learning in {0, 0.02} x alpha in
{0, 0.2, 0.4}, seed 0, t_end 20000. Writes results/inventory_ablation.csv.
Usage: uv run python experiments/inventory_ablation.py
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from marketsim import SimConfig, run
from marketsim.metrics import summarize

OUT = Path(__file__).resolve().parents[1] / "results" / "inventory_ablation.csv"


def one(args):
    k, beta, alpha = args
    res = run(SimConfig(seed=0, informed_frac=alpha, noise_k=k, mm_inventory_learning=beta))
    inv = np.abs(res.samples["mm_inventory"])
    s = summarize(res)
    return {"noise_k": k, "mm_inventory_learning": beta, "informed_frac": alpha,
            "mean_abs_inventory": float(inv.mean()), "p95_abs_inventory": float(np.percentile(inv, 95)),
            "frac_time_at_limit": float(np.mean(inv >= res.config.mm_max_inventory)),
            "mm_pnl_per_fill": s["mm_pnl_per_fill"], "mean_abs_mid_minus_V": s["mean_abs_mid_minus_V"]}


def main() -> None:
    grid = list(product([0.0, 0.1], [0.0, 0.02], [0.0, 0.2, 0.4]))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, grid))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.3g}" for k, v in r.items()))


if __name__ == "__main__":
    main()
