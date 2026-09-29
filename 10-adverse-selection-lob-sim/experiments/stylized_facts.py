"""One long baseline run plus ablations, measuring stylized facts.

Runs (same seed, same everything else):
  baseline     informed_frac=0.4, fixed AS market maker
  no_jumps     baseline with jump_rate=0
  no_informed  baseline with informed_frac=0
  price_learning_bug  no_jumps with the original (buggy) MM and price-inelastic noise

Writes results/stylized/{summary.json, acf.csv, jump_event_study.csv} and PNGs.
Usage: uv run python experiments/stylized_facts.py [--t-end 200000] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marketsim import SimConfig, run
from marketsim.metrics import acf, excess_kurtosis, hill_tail_index, log_returns, mm_pnl_decomposition, summarize

OUT = Path(__file__).resolve().parents[1] / "results" / "stylized"
MAX_LAG = 50
RET_EVERY = 10  # returns over 10 time units
SOURCE = "Source: results/stylized/"


def configs(t_end: float, seed: int) -> dict[str, SimConfig]:
    base = SimConfig(seed=seed, t_end=t_end, informed_frac=0.4)
    return {"baseline": base, "no_jumps": base.with_(jump_rate=0.0), "no_informed": base.with_(informed_frac=0.0),
            # The original model, kept to document the bug: trade-price
            # learning, price-inelastic noise, no inventory learning.
            "price_learning_bug": base.with_(jump_rate=0.0, mm_learning_signal="price", noise_k=0.0,
                                             mm_inventory_learning=0.0)}


def _run(item):
    name, cfg = item
    return name, run(cfg)


def event_study(res, pre: int = 20, post: int = 150) -> dict[str, np.ndarray]:
    """Average spread and |mid - V| around fundamental jumps (sample index lags)."""
    smp = res.samples
    t = smp["t"]
    dt = res.config.sample_dt
    lags = np.arange(-pre, post + 1)
    spread_rows, gap_rows = [], []
    for jt in res.jump_times:
        i = int(round(jt / dt))
        if i - pre < 0 or i + post >= len(t):
            continue
        sl = slice(i - pre, i + post + 1)
        spread_rows.append(smp["spread"][sl])
        gap_rows.append(np.abs(smp["mid"][sl] - smp["fundamental"][sl]))
    return {"lag": lags, "spread": np.nanmean(spread_rows, axis=0), "abs_gap": np.nanmean(gap_rows, axis=0),
            "n_events": len(spread_rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=200_000.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as ex:
        results = dict(ex.map(_run, configs(args.t_end, args.seed).items()))

    summary, acfs = {}, {}
    for name, res in results.items():
        s = summarize(res, return_every=RET_EVERY, max_lag=MAX_LAG)
        # Aggregational Gaussianity: kurtosis should fall as the horizon grows.
        for h in (1, 10, 100):
            r = log_returns(res.samples["mid"], h)
            s[f"excess_kurtosis_h{h}"] = excess_kurtosis(r)
            s[f"hill_h{h}"] = hill_tail_index(r)
        s["n_jumps"] = res.counters["n_jumps"]
        d100 = np.diff(res.samples["mid"][::100])
        s["max_abs_move_100"] = float(np.max(np.abs(d100)))
        s["mean_abs_mm_inventory"] = float(np.mean(np.abs(res.samples["mm_inventory"])))
        s["wall_s"] = res.counters["wall_s"]
        s["config"] = res.config.to_dict()
        summary[name] = s
        r = log_returns(res.samples["mid"], RET_EVERY)
        acfs[name] = {"ret": acf(r, MAX_LAG), "abs": acf(np.abs(r), MAX_LAG), "n": len(r)}

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT / "acf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag"] + [f"{n}_{k}" for n in acfs for k in ("ret", "abs")])
        for lag in range(MAX_LAG):
            w.writerow([lag + 1] + [f"{acfs[n][k][lag]:.5f}" for n in acfs for k in ("ret", "abs")])

    base = results["baseline"]
    ev = event_study(base)
    with open(OUT / "jump_event_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag", "mean_spread", "mean_abs_mid_minus_V", "n_events"])
        for i, lag in enumerate(ev["lag"]):
            w.writerow([int(lag), f"{ev['spread'][i]:.4f}", f"{ev['abs_gap'][i]:.4f}", ev["n_events"]])

    for name, s in summary.items():
        print(f"{name:12s} kurt(h=10)={s['excess_kurtosis']:.2f} kurt h1/h10/h100="
              f"{s['excess_kurtosis_h1']:.2f}/{s['excess_kurtosis_h10']:.2f}/{s['excess_kurtosis_h100']:.2f} "
              f"hill={s['hill_tail_index']:.2f} acf_r1={s['acf_ret_lag1']:.3f} acf|r|1/5/20="
              f"{s['acf_abs_lag1']:.3f}/{s['acf_abs_lag5']:.3f}/{s['acf_abs_lag20']:.3f} spread={s['mean_spread']:.2f} "
              f"mm_pnl={s['mm_pnl_total']:.0f} (capture {s['mm_spread_capture']:.0f}, advsel "
              f"{s['mm_adverse_selection']:.0f}, carry {s['mm_inventory_carry']:.0f}) fills={s['mm_n_fills']:.0f} "
              f"max|move100|={s['max_abs_move_100']:.1f} mean|inv|={s['mean_abs_mm_inventory']:.1f} "
              f"|mid-V|={s['mean_abs_mid_minus_V']:.2f} wall={s['wall_s']:.1f}s")
    print(f"jump event study: {ev['n_events']} jumps; spread pre={np.nanmean(ev['spread'][:20]):.2f} "
          f"peak={np.nanmax(ev['spread'][20:]):.2f}; |mid-V| pre={np.nanmean(ev['abs_gap'][:20]):.2f} "
          f"at +1={ev['abs_gap'][21]:.2f} at +50={ev['abs_gap'][70]:.2f} at +150={ev['abs_gap'][-1]:.2f}")

    plot_returns(results, acfs)
    plot_spread(base, ev)
    plot_pnl(base)


def plot_returns(results, acfs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    r = log_returns(results["baseline"].samples["mid"], RET_EVERY)
    z = (r - r.mean()) / r.std()
    bins = np.linspace(-8, 8, 81)
    h, e = np.histogram(z, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax.semilogy(c[h > 0], h[h > 0], "o", ms=3, label="simulated (baseline)")
    ax.semilogy(c, np.exp(-c ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="standard normal")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel(f"standardized {RET_EVERY}-unit log return")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Return distribution: fat tails")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    lags = np.arange(1, MAX_LAG + 1)
    ax.plot(lags, acfs["baseline"]["abs"], "-o", ms=3, color="#d62728", label="|r| baseline")
    ax.plot(lags, acfs["no_jumps"]["abs"], "-", color="#ff9896", label="|r| no jumps")
    ax.plot(lags, acfs["baseline"]["ret"], "-o", ms=3, color="#1f77b4", label="r baseline")
    band = 1.96 / np.sqrt(acfs["baseline"]["n"])
    ax.axhspan(-band, band, color="grey", alpha=0.2, label="95% band for iid")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(f"lag (x {RET_EVERY} time units)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Volatility clustering")
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, SOURCE + "acf.csv, summary.json", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "returns_and_acf.png", dpi=140)
    plt.close(fig)


def plot_spread(res, ev) -> None:
    smp = res.samples
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    # A window around the first jump that lands at least 500 units in.
    jt = next((j for j in res.jump_times if j > 500), 500.0)
    sl = slice(int(jt) - 100, int(jt) + 300)
    ax.plot(smp["t"][sl], smp["best_bid"][sl], color="#1f77b4", lw=0.8, label="best bid")
    ax.plot(smp["t"][sl], smp["best_ask"][sl], color="#d62728", lw=0.8, label="best ask")
    ax.plot(smp["t"][sl], smp["fundamental"][sl], color="black", lw=1.2, label="fundamental V")
    ax.set_xlabel("time")
    ax.set_ylabel("price (ticks)")
    ax.set_title("Quotes around a jump")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    spread = smp["spread"][np.isfinite(smp["spread"])]
    vals, counts = np.unique(spread, return_counts=True)
    keep = vals <= 20
    ax.bar(vals[keep], counts[keep] / counts.sum(), color="#7f7f7f")
    ax.set_xlabel("quoted spread (ticks)")
    ax.set_ylabel("fraction of snapshots")
    ax.set_title("Spread distribution")

    ax = axes[2]
    ax.plot(ev["lag"], ev["spread"], color="#7f7f7f", label="mean spread")
    ax2 = ax.twinx()
    ax2.plot(ev["lag"], ev["abs_gap"], color="black", ls="--", label="mean |mid - V|")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("time since jump")
    ax.set_ylabel("spread (ticks)")
    ax2.set_ylabel("|mid - V| (ticks)")
    ax.set_title(f"Event study over {ev['n_events']} jumps")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
    fig.text(0.01, 0.01, SOURCE + "jump_event_study.csv (baseline run)", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "spread_dynamics.png", dpi=140)
    plt.close(fig)


def plot_pnl(res) -> None:
    f = res.mm_fills
    s, p, q = f["side"], f["price"], f["qty"]
    capture = np.cumsum(s * q * (f["mid_before"] - p))
    adverse = np.cumsum(s * q * (f["mid_after_h"] - f["mid_before"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(f["t"], capture, color="#2ca02c", label="spread capture")
    ax.plot(f["t"], adverse, color="#d62728", label="adverse selection (10-unit markout)")
    ax.plot(res.samples["t"], res.samples["mm_wealth"], color="black", label="total PnL, marked to mid")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative PnL (ticks)")
    d = mm_pnl_decomposition(res)
    ax.set_title(f"MM PnL decomposition (inventory carry = {d['inventory_carry']:.0f})")
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, SOURCE + "summary.json (baseline run)", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "mm_pnl_decomposition.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
