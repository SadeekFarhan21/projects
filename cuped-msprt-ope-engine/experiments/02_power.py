"""Empirical power of the from-scratch Welch test against the analytic noncentral-t power.

    uv run python experiments/02_power.py --sims 2000 --n 1000
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from _common import PALETTE, RESULTS, plt, save, style, wilson, write_json
from expope.stats import MeanStats, analytic_power_welch, welch_from_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    effects = np.round(np.linspace(0, 0.2, 11), 3)
    rows = []
    for dist in ("normal", "lognormal"):
        for d in effects:
            rej = 0
            for _ in range(args.sims):
                if dist == "normal":
                    c = rng.normal(0, 1, args.n)
                    t = rng.normal(d, 1, args.n)
                else:
                    # Log-normal with unit standard deviation, shifted so the mean difference is d.
                    s = 0.8
                    sd = np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))
                    c = rng.lognormal(0, s, args.n) / sd
                    t = rng.lognormal(0, s, args.n) / sd + d
                r = welch_from_stats(MeanStats.from_array(c), MeanStats.from_array(t))
                rej += r.p_value < 0.05
            lo, hi = wilson(rej, args.sims)
            rows.append(dict(dist=dist, effect=float(d), n_per_arm=args.n, sims=args.sims, power_empirical=rej / args.sims,
                             power_lo=lo, power_hi=hi, power_analytic=analytic_power_welch(float(d), 1.0, args.n)))
            print(rows[-1])
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "power.csv")
    df = df.with_columns((pl.col("power_empirical") - pl.col("power_analytic")).abs().alias("abs_gap"))
    inside = df.filter((pl.col("power_analytic") >= pl.col("power_lo")) & (pl.col("power_analytic") <= pl.col("power_hi")))
    write_json("power_summary.json", dict(config=vars(args), max_abs_gap=df["abs_gap"].max(),
                                          max_abs_gap_by_dist={k: g["abs_gap"].max() for (k,), g in df.group_by("dist")},
                                          analytic_inside_wilson_95=f"{inside.height}/{df.height}"))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grid = np.linspace(0, 0.2, 200)
    ax.plot(grid, [analytic_power_welch(g, 1.0, args.n) for g in grid], color="#333", lw=1.5, label="Analytic (noncentral t)")
    for i, dist in enumerate(("normal", "lognormal")):
        g = df.filter(pl.col("dist") == dist)
        ax.errorbar(g["effect"], g["power_empirical"], yerr=[g["power_empirical"] - g["power_lo"], g["power_hi"] - g["power_empirical"]],
                    fmt="o", ms=4, color=PALETTE[i], capsize=2, label=f"Simulated, {dist} data")
    style(ax, f"Welch test power with n = {args.n} per arm", "True effect in standard deviations", "Power at alpha 0.05")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "power_curve.png")


if __name__ == "__main__":
    main()
