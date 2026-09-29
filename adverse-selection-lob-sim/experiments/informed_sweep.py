"""Sweep the informed fraction alpha and measure spread and MM profitability.

Three market-maker variants:
  fixed     AS quotes, GM learning (w = alpha), no adverse-selection widening
  adaptive  same, plus half-spread += EWMA of measured markout loss
  naive     AS quotes, fixed learning rate w = 0.3 regardless of alpha

Writes results/sweep/runs.csv, results/sweep/aggregate.csv and two PNGs.
Usage: uv run python experiments/informed_sweep.py [--seeds 8] [--t-end 20000]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marketsim import SimConfig, run
from marketsim.metrics import summarize

OUT = Path(__file__).resolve().parents[1] / "results" / "sweep"
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
VARIANTS = {
    "fixed": dict(mm_adaptive=False),
    "adaptive": dict(mm_adaptive=True),
    "naive": dict(mm_adaptive=False, mm_learning_mode="fixed", mm_learning=0.3),
}
AGG_KEYS = ["mean_spread", "mean_mm_quote_spread", "mm_pnl_per_fill", "mm_spread_capture_per_fill", "mm_adverse_selection_per_fill",
            "mm_pnl_total", "mm_spread_capture", "mm_adverse_selection", "mm_inventory_carry", "mm_n_fills",
            "mm_edge_vs_V_per_fill", "gm_breakeven_half_spread", "informed_share_of_trades",
            "mean_abs_mid_minus_V", "excess_kurtosis", "acf_ret_lag1", "acf_abs_lag1", "informed_edge",
            "informed_edge_per_trade", "noise_edge", "noise_lp_edge", "mm_edge", "mm_max_abs_inventory"]


def one(job: tuple[str, float, int, float]) -> dict:
    variant, alpha, seed, t_end = job
    cfg = SimConfig(seed=seed, informed_frac=alpha, t_end=t_end, **VARIANTS[variant])
    s = summarize(run(cfg))
    s["variant"] = variant
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--t-end", type=float, default=20_000.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(v, a, s, args.t_end) for v in VARIANTS for a in ALPHAS for s in range(args.seeds)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    print(f"{len(rows)} runs in {wall:.1f}s on {args.workers} workers")

    cols = ["variant"] + [k for k in rows[0] if k != "variant"]
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    agg = []
    for v in VARIANTS:
        for a in ALPHAS:
            sel = [r for r in rows if r["variant"] == v and r["informed_frac"] == a]
            rec = {"variant": v, "informed_frac": a, "n_seeds": len(sel)}
            for k in AGG_KEYS:
                x = np.array([r[k] for r in sel], dtype=float)
                rec[k] = float(np.mean(x))
                rec[k + "_se"] = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
            agg.append(rec)
    with open(OUT / "aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    with open(OUT / "meta.json", "w") as f:
        json.dump({"alphas": ALPHAS, "variants": VARIANTS, "seeds": args.seeds, "t_end": args.t_end,
                   "wall_s": wall, "n_runs": len(rows), "base_config": SimConfig().to_dict()}, f, indent=2)

    # Print a compact table for the log.
    print(f"{'variant':9s} {'alpha':>5s} {'book_spr':>8s} {'mm_spr':>7s} {'pnl/fill':>9s} {'capture':>8s} {'advsel':>8s} "
          f"{'inf_edge':>8s} {'GM h*':>6s} {'|mid-V|':>7s}")
    for r in agg:
        print(f"{r['variant']:9s} {r['informed_frac']:5.2f} {r['mean_spread']:8.2f} {r['mean_mm_quote_spread']:7.2f} "
              f"{r['mm_pnl_per_fill']:9.3f} {r['mm_spread_capture_per_fill']:8.3f} "
              f"{r['mm_adverse_selection_per_fill']:8.3f} {r['informed_edge_per_trade']:8.2f} "
              f"{r['gm_breakeven_half_spread']:6.2f} {r['mean_abs_mid_minus_V']:7.2f}")
    plot(agg)


def replot() -> None:
    """Redraw the figure from results/sweep/aggregate.csv without rerunning."""
    with open(OUT / "aggregate.csv") as f:
        agg = [{k: (v if k == "variant" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    plot(agg)


def _series(agg, v, k):
    sel = [r for r in agg if r["variant"] == v]
    return (np.array([r["informed_frac"] for r in sel]), np.array([r[k] for r in sel]),
            np.array([r[k + "_se"] for r in sel]))


def plot(agg) -> None:
    colors = {"fixed": "#1f77b4", "adaptive": "#d62728", "naive": "#7f7f7f"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for v in ("fixed", "adaptive"):
        a, y, se = _series(agg, v, "mean_mm_quote_spread")
        ax.errorbar(a, y / 2, yerr=se / 2, marker="o", color=colors[v], label=f"{v} MM: own quoted half-spread")
    a, y, se = _series(agg, "adaptive", "mean_spread")
    ax.plot(a, y / 2, ls=":", color=colors["adaptive"], label="adaptive run: book half-spread")
    a, y, se = _series(agg, "fixed", "mm_adverse_selection_per_fill")
    ax.errorbar(a, -y, yerr=se, marker="s", ls="--", color="black", label="fixed MM: adverse selection per fill")
    a, y, se = _series(agg, "fixed", "gm_breakeven_half_spread")
    ax.errorbar(a, y, yerr=se, marker="^", ls="--", color="#9467bd", label="informed edge per taker trade")
    ax.set_xlabel("informed fraction of taker arrivals (alpha)")
    ax.set_ylabel("ticks")
    ax.set_title("Half-spread vs adverse selection")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for v in ("fixed", "adaptive", "naive"):
        a, y, se = _series(agg, v, "mm_pnl_per_fill")
        ax.errorbar(a, y, yerr=se, marker="o", color=colors[v], label=f"{v} MM: PnL per fill")
    a, y, _ = _series(agg, "fixed", "mm_spread_capture_per_fill")
    ax.plot(a, y, ls=":", color=colors["fixed"], label="fixed: spread capture")
    a, y, _ = _series(agg, "fixed", "mm_adverse_selection_per_fill")
    ax.plot(a, y, ls="-.", color=colors["fixed"], label="fixed: adverse selection")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("informed fraction of taker arrivals (alpha)")
    ax.set_ylabel("ticks per unit filled")
    ax.set_title("Market-maker PnL per fill")
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.text(0.01, 0.01, "Source: results/sweep/aggregate.csv (mean over seeds, bars = 1 s.e.)", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "spread_and_pnl_vs_alpha.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    replot() if "--replot" in sys.argv else main()
