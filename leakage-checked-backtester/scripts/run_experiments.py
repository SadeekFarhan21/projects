"""Run the v0 experiment suite on the real Binance panel and save raw outputs.

Every experiment is a tracked run (runs/<id>/) built from configs/xs_ridge.toml plus
overrides. Writes results/experiments.{csv,json}, results/runs_compare.txt and plots.

Usage: uv run python scripts/run_experiments.py
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl

from qrp.cli import main as qrp_cli
from qrp.research import load_config, load_panel, run_research

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASE = ROOT / "configs" / "xs_ridge.toml"

LEAKY_FWD = [{"name": "fwd5", "op": "leaky_forward_return", "window": 5},
             {"name": "fwd5_r", "op": "xs_rank", "inputs": ["fwd5"]}]
LEAKY_CTR = [{"name": "ctr11", "op": "leaky_centered_mean", "window": 11},
             {"name": "ctr11_r", "op": "xs_rank", "inputs": ["ctr11"]}]

# (experiment id, group, overrides, extra feature specs, extra model features)
EXPERIMENTS = [
    ("baseline", "core", [], [], []),
    ("no_costs", "costs", ["costs.fee_bps=0", "costs.slippage_bps=0"], [], []),
    ("fee_5", "costs", ["costs.fee_bps=5", "costs.slippage_bps=0"], [], []),
    ("fee_20", "costs", ["costs.fee_bps=20", "costs.slippage_bps=5"], [], []),
    ("fee_40", "costs", ["costs.fee_bps=40", "costs.slippage_bps=10"], [], []),
    ("impact_aum_50m", "costs", ["costs.impact_coef=0.5", "costs.aum=50000000"], [], []),
    ("daily_rebalance", "core", ["portfolio.rebalance_every=1"], [], []),
    ("delay_0", "timing", ["backtest.delay=0"], [], []),
    ("random_control", "control", ["model.kind=random"], [], []),
    ("signal_reversal_1d", "baseline_signals",
     ["model.kind=signal", "model.signal=ret1_r", "model.sign=-1"], [], []),
    ("signal_momentum_30d", "baseline_signals",
     ["model.kind=signal", "model.signal=mom30_r", "model.sign=1"], [], []),
    ("leak_forward_return", "leakage", ["checks.fail_on_leak=false"], LEAKY_FWD, ["fwd5_r"]),
    ("leak_centered_mean", "leakage", ["checks.fail_on_leak=false"], LEAKY_CTR, ["ctr11_r"]),
    ("kfold_h20_purged", "purging",
     ["cv.kind=kfold", "model.horizon=20", "cv.embargo=10", "cv.purge=true"], [], []),
    ("kfold_h20_unpurged", "purging",
     ["cv.kind=kfold", "model.horizon=20", "cv.embargo=0", "cv.purge=false"], [], []),
    ("wf_h20_purged", "purging", ["model.horizon=20", "cv.embargo=10"], [], []),
    ("wf_h20_unpurged", "purging", ["model.horizon=20", "cv.embargo=0", "cv.purge=false"], [], []),
]


def plot(df: pl.DataFrame, curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, color in (("baseline", "#4C72B0"), ("no_costs", "#8FB0D9"), ("delay_0", "#DD8452"),
                       ("random_control", "#999999")):
        d, e = curves[key]
        ax1.plot(d, e / e[0], label=key, color=color, lw=1.4)
    ax1.set_title("Honest runs", fontsize=10)
    for key, color in (("baseline", "#4C72B0"), ("leak_centered_mean", "#C44E52"),
                       ("leak_forward_return", "#8C1C13")):
        d, e = curves[key]
        ax2.plot(d, e / e[0], label=key, color=color, lw=1.4)
    ax2.set_title("Leaky features (audit disabled)", fontsize=10)
    import matplotlib.dates as mdates
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    ax1.set_ylabel("out of sample equity (log scale, start = 1)")
    fig.suptitle("Binance top 100 USDT pairs, weekly rebalanced ridge", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "experiments_equity.png", dpi=150)

    cs = df.filter(pl.col("experiment").is_in(["no_costs", "fee_5", "baseline", "fee_20", "fee_40"]))
    bps = {"no_costs": 0, "fee_5": 5, "baseline": 15, "fee_20": 25, "fee_40": 50}
    x = np.array([bps[e] for e in cs["experiment"]])
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(x[order], cs["sharpe"].to_numpy()[order], "o-", color="#4C72B0")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("one way cost per unit traded (bps)")
    ax.set_ylabel("out of sample Sharpe")
    ax.set_title("Cost sensitivity, weekly rebalanced ridge", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "cost_sensitivity.png", dpi=150)


def plot_only() -> None:
    """Redraw the figures from results/experiments.csv and the tracked runs' artifacts."""
    df = pl.read_csv(RESULTS / "experiments.csv")
    curves = {}
    for exp, rid in zip(df["experiment"], df["run_id"]):
        eq = pl.read_parquet(ROOT / "runs" / rid / "artifacts" / "equity.parquet").filter("oos")
        curves[exp] = (eq["date"].to_numpy(), eq["equity"].to_numpy())
    plot(df, curves)


def main():
    RESULTS.mkdir(exist_ok=True)
    cfg0 = load_config(BASE)
    t0 = time.time()
    panel, snapshot = load_panel(cfg0)
    print(f"panel {panel.shape} snapshot {snapshot} loaded in {time.time() - t0:.2f}s", flush=True)

    rows, curves = [], {}
    for exp_id, group, ovs, extra_feats, extra_model in EXPERIMENTS:
        cfg = load_config(BASE, ovs + [f"run.name={exp_id}"])
        cfg["features"] = cfg["features"] + extra_feats
        cfg["model"]["features"] = cfg["model"]["features"] + extra_model
        t1 = time.time()
        out = run_research(cfg, panel=panel, snapshot=snapshot)
        m = out["metrics"]
        rows.append({"experiment": exp_id, "group": group, "run_id": out["run"].id,
                     "overrides": " ".join(ovs), "seconds": round(time.time() - t1, 2),
                     **{k: m.get(k) for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown",
                                              "ic", "avg_daily_turnover", "ann_cost_drag",
                                              "n_days", "leak_free")}})
        curves[exp_id] = (out["dates"][out["oos"]], out["result"].equity[out["oos"]])
        print(f"{exp_id:22s} sharpe={m['sharpe']:+.3f} ic={m['ic']:+.4f} "
              f"ret={m['ann_return']:+.3f} to={m['avg_daily_turnover']:.3f} "
              f"leak_free={m['leak_free']}", flush=True)

    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "experiments.csv")
    (RESULTS / "experiments.json").write_text(json.dumps(
        {"snapshot": snapshot, "panel_shape": list(panel.shape),
         "start": str(panel.dates[0]), "end": str(panel.dates[-1]), "rows": rows}, indent=2))

    # the tracker's own compare output, saved verbatim
    ids = {r["experiment"]: r["run_id"] for r in rows}
    buf = io.StringIO()
    with redirect_stdout(buf):
        qrp_cli(["runs", "compare", ids["baseline"], ids["no_costs"], ids["delay_0"],
                 ids["leak_centered_mean"]])
    (RESULTS / "runs_compare.txt").write_text(buf.getvalue())

    plot(df, curves)
    print(f"total {time.time() - t0:.1f}s; wrote results/experiments.* runs_compare.txt *.png")


if __name__ == "__main__":
    import sys
    plot_only() if "--plot-only" in sys.argv else main()
