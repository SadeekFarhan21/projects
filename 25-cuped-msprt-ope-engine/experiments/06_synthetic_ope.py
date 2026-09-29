"""OPE on a synthetic contextual bandit with a known policy value.

For each replicate: draw n logged rounds from the behaviour policy, fit a cross-fitted
LightGBM reward model, run every estimator, and build bootstrap 95 percent intervals.
Also runs DM and DR with a deliberately weak (context-free) reward model to show double robustness.

    uv run python experiments/06_synthetic_ope.py --reps 200 --n 5000
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from _common import PALETTE, RESULTS, plt, save, style, wilson, write_json
from expope.ope import (
    RewardModelConfig,
    bootstrap_ci,
    estimate_all,
    estimator_terms,
    fit_predict_q_hat,
    per_action_mean_q_hat,
    row_terms,
)
from expope.sim import SyntheticBandit


def run_rep(job):
    rep, args = job
    env = SyntheticBandit(seed=args.seed)
    rng = np.random.default_rng([args.seed + 1, rep])
    d = env.sample_logs(args.n, rng)
    q_gbm = fit_predict_q_hat(d["context"], d["action"], d["reward"], env.n_actions, cfg=RewardModelConfig(seed=rep, n_jobs=1))
    q_bad = per_action_mean_q_hat(d["action"], d["reward"], args.n, env.n_actions)
    rows = []
    for model, q in {"gbm": q_gbm, "weak": q_bad}.items():
        t = row_terms(d["reward"], d["action"], d["pscore"], d["action_dist"], None, q)
        ests = estimate_all(t, tau=args.tau)
        names = ests.keys() if model == "gbm" else ("dm", "dr", "switch_dr")
        for name in names:
            num, den = estimator_terms(name, t, args.tau)
            ci = bootstrap_ci(num, den, n_boot=args.n_boot, seed=rep)
            rows.append(dict(rep=rep, estimator=name, reward_model=model if name not in ("ips", "snips") else "none",
                             estimate=ests[name], lower=ci["lower"], upper=ci["upper"]))
    w = row_terms(d["reward"], d["action"], d["pscore"], d["action_dist"]).w
    ws = dict(mean=float(w.mean()), max=float(w.max()), p99=float(np.percentile(w, 99)), share_above_tau=float((w > args.tau).mean()))
    return rows, ws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--tau", type=float, default=2.0, help="Switch-DR weight threshold")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = SyntheticBandit(seed=args.seed)
    truth = env.true_value(n=2_000_000)
    print("true policy value", truth)
    t0 = time.perf_counter()
    jobs = [(rep, args) for rep in range(args.reps)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (rep_rows, ws) in enumerate(ex.map(run_rep, jobs, chunksize=2)):
            rows += [dict(r, truth=truth) for r in rep_rows]
            if i == 0:
                weight_stats = ws
            if i % 20 == 0:
                print(i, f"{time.perf_counter() - t0:.1f}s", flush=True)
    raw = pl.DataFrame(rows)
    raw.write_csv(RESULTS / "raw" / "synthetic_ope_reps.csv")
    summ = (
        raw.with_columns(
            ((pl.col("estimate") - pl.col("truth")) / pl.col("truth")).alias("rel_err"),
            ((pl.col("lower") <= pl.col("truth")) & (pl.col("truth") <= pl.col("upper"))).alias("covered"),
        )
        .group_by("estimator", "reward_model", maintain_order=True)
        .agg(
            pl.len().alias("reps"),
            pl.col("rel_err").mean().alias("mean_rel_bias"),
            pl.col("rel_err").abs().mean().alias("mean_abs_rel_err"),
            ((pl.col("estimate") - pl.col("truth")) ** 2).mean().sqrt().alias("rmse"),
            pl.col("estimate").std().alias("sd"),
            pl.col("covered").mean().alias("ci_coverage"),
            pl.col("covered").sum().alias("covered_n"),
        )
    )
    summ = summ.with_columns(
        pl.struct("covered_n", "reps").map_elements(lambda s: wilson(s["covered_n"], s["reps"])[0], return_dtype=pl.Float64).alias("cov_lo"),
        pl.struct("covered_n", "reps").map_elements(lambda s: wilson(s["covered_n"], s["reps"])[1], return_dtype=pl.Float64).alias("cov_hi"),
    )
    summ.write_csv(RESULTS / "synthetic_ope.csv")
    write_json("synthetic_ope_summary.json", dict(config=vars(args), truth=truth, weights_rep0=weight_stats,
                                                  wall_s=time.perf_counter() - t0, rows=summ.to_dicts()))
    print(summ)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{r['estimator']}\n{r['reward_model']}" for r in summ.to_dicts()]
    x = np.arange(summ.height)
    ax.bar(x - 0.2, summ["mean_rel_bias"].abs() * 100, 0.4, color=PALETTE[1], label="Absolute relative bias")
    ax.bar(x + 0.2, summ["rmse"] / truth * 100, 0.4, color=PALETTE[0], label="Relative RMSE")
    ax.set_xticks(x, labels, fontsize=8)
    style(ax, f"Synthetic OPE over {args.reps} replicates, n = {args.n}", "", "Percent of true policy value")
    ax.legend(frameon=False)
    save(fig, "synthetic_ope_error.png")


if __name__ == "__main__":
    main()
