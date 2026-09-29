"""CUPED variance reduction versus the theoretical 1 - rho^2, plus CUPED on the event-log data.

    uv run python experiments/05_cuped.py --sims 1000 --n 1000
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import polars as pl

from _common import PALETTE, RESULTS, plt, save, style, write_json
from expope.metrics import MetricsEngine
from expope.sim import EventLogConfig, generate_event_log
from expope.stats import cuped_ttest, welch_ttest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for rho in np.round(np.arange(0, 0.96, 0.1), 2):
        naive, adj, se_ratio = [], [], []
        for _ in range(args.sims):
            x = rng.normal(size=2 * args.n)
            y = rho * x + math.sqrt(1 - rho**2) * rng.normal(size=2 * args.n)
            y[args.n :] += 0.05
            r0 = welch_ttest(y[: args.n], y[args.n :])
            r1, _ = cuped_ttest(y[: args.n], x[: args.n], y[args.n :], x[args.n :])
            naive.append(r0.estimate)
            adj.append(r1.estimate)
            se_ratio.append((r1.se / r0.se) ** 2)
        ratio = np.var(adj, ddof=1) / np.var(naive, ddof=1)
        rows.append(dict(rho=float(rho), theory=1 - rho**2, empirical_var_ratio=ratio, mean_se2_ratio=float(np.mean(se_ratio)),
                         mean_cuped_est=float(np.mean(adj)), true_effect=0.05))
        print(rows[-1])
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "cuped_variance.csv")

    # Realistic check on the synthetic event log: variance reduction per metric vs the observed rho.
    d = generate_event_log(50, EventLogConfig(n_users=4000), seed=args.seed)
    eng = MetricsEngine()
    eng.load(**d)
    eng.build_user_metrics()
    um = eng.con.execute("SELECT * FROM user_metrics").pl()
    ev_rows = []
    for metric, cov in (("revenue", "pre_revenue"), ("sessions", "pre_sessions")):
        rho = float(np.corrcoef(um[metric].to_numpy(), um[cov].to_numpy())[0, 1])
        res = eng.analyze().filter(pl.col("metric") == f"{metric}_per_user")
        w = res.filter(pl.col("method") == "welch")["se"].to_numpy()
        c = res.filter(pl.col("method") == "cuped")["se"].to_numpy()
        ev_rows.append(dict(metric=f"{metric}_per_user", pooled_rho=rho, theory=1 - rho**2, mean_se2_ratio=float(np.mean((c / w) ** 2))))
    print(ev_rows)
    write_json("cuped_summary.json", dict(config=vars(args), synthetic=rows, event_log=ev_rows,
                                          max_abs_gap=float((df["empirical_var_ratio"] - df["theory"]).abs().max())))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    grid = np.linspace(0, 0.95, 100)
    ax.plot(grid, 1 - grid**2, color="#333", lw=1.5, label="Theory, 1 minus rho squared")
    ax.plot(df["rho"], df["empirical_var_ratio"], "o", color=PALETTE[0], label="Simulated variance ratio")
    for r in ev_rows:
        ax.plot(r["pooled_rho"], r["mean_se2_ratio"], "s", color=PALETTE[1], ms=7)
        ax.annotate(r["metric"].replace("_", " "), (r["pooled_rho"], r["mean_se2_ratio"]), xytext=(8, 4), textcoords="offset points", fontsize=8)
    style(ax, "CUPED variance reduction", "Correlation between outcome and pre period covariate", "Variance with CUPED over without")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "cuped_variance.png")


if __name__ == "__main__":
    main()
