"""1,000 A/A tests end to end through the DuckDB metrics layer.

Every experiment has its own population of users, a 50/50 split and no treatment effect.
The metrics layer computes per-user metrics and sufficient statistics in SQL, then the
from-scratch tests run on those sums. Reports the false positive rate per metric and method,
the SRM check FPR, family-wise error with and without Holm/BH, and p-value uniformity.

    uv run python experiments/01_aa_fpr.py --n-experiments 1000 --n-users 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
from scipy import stats as sps

from _common import PALETTE, RAW, plt, save, style, wilson, write_json
from expope.metrics import MetricsEngine
from expope.sim import EventLogConfig, generate_event_log
from expope.stats import benjamini_hochberg, holm


def plot(fpr: pl.DataFrame, res: pl.DataFrame, n_fam: int, alpha: float) -> None:
    """FPR with Wilson intervals (horizontal, so long metric names stay readable) and a p-value histogram."""
    rows = fpr.to_dicts()[::-1]
    lab = [f"{r['metric'].replace('_', ' ')}, {r['method']}" for r in rows]
    y = np.arange(len(rows))
    f = np.array([r["fpr"] for r in rows])
    lo = np.array([r["fpr_lo"] for r in rows])
    hi = np.array([r["fpr_hi"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axvspan(0.04, 0.06, color="#3a7d44", alpha=0.12, label="4 to 6 percent band")
    ax.axvline(alpha, color="#555", lw=1, ls="--")
    ax.errorbar(f, y, xerr=[f - lo, hi - f], fmt="o", color=PALETTE[0], capsize=3)
    ax.set_yticks(y, lab, fontsize=9)
    ax.set_xlim(0, 0.1)
    style(ax, f"A/A false positive rate over {n_fam} experiments", "False positive rate at alpha 0.05", "")
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "aa_fpr.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    p = res.filter((pl.col("metric") == "revenue_per_user") & (pl.col("method") == "welch"))["p_value"].to_numpy()
    ax.hist(p, bins=20, range=(0, 1), color=PALETTE[0], alpha=0.85)
    ax.axhline(len(p) / 20, color="#555", ls="--", lw=1)
    style(ax, "A/A p values for revenue per user", "p value", "Count")
    save(fig, "aa_pvalue_hist.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-experiments", type=int, default=1000)
    ap.add_argument("--n-users", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved results")
    args = ap.parse_args()
    if args.plot_only:
        res = pl.read_csv(RAW / "aa_results.csv")
        plot(pl.read_csv(RAW.parent / "aa_fpr.csv"), res, res["experiment_id"].n_unique(), args.alpha)
        return

    t0 = time.perf_counter()
    data = generate_event_log(args.n_experiments, EventLogConfig(n_users=args.n_users), seed=args.seed)
    t_gen = time.perf_counter() - t0
    n_events = data["events"].height

    eng = MetricsEngine()
    t0 = time.perf_counter()
    eng.load(**data)
    t_load = time.perf_counter() - t0
    del data
    t0 = time.perf_counter()
    eng.build_user_metrics()
    t_user = time.perf_counter() - t0
    t0 = time.perf_counter()
    stats = eng.sufficient_stats()
    counts = eng.assignment_counts()
    t_stats = time.perf_counter() - t0
    from expope.metrics import analyze

    t0 = time.perf_counter()
    res = analyze(stats, counts, eng.experiments(), alpha=args.alpha)
    t_analyze = time.perf_counter() - t0
    res.write_csv(RAW / "aa_results.csv")

    rows = []
    for (metric, method), g in res.group_by(["metric", "method"], maintain_order=True):
        k = int((g["p_value"] < args.alpha).sum())
        n = g.height
        lo, hi = wilson(k, n)
        ks = sps.kstest(g["p_value"].to_numpy(), "uniform").pvalue
        rows.append(dict(metric=metric, method=method, n_tests=n, false_positives=k, fpr=k / n, fpr_lo=lo, fpr_hi=hi, ks_uniform_p=ks))
    exp_level = res.unique("experiment_id").sort("experiment_id")
    for col in ("srm_p_assigned", "srm_p_exposed"):
        k = int((exp_level[col] < args.alpha).sum())
        lo, hi = wilson(k, exp_level.height)
        ks = sps.kstest(exp_level[col].to_numpy(), "uniform").pvalue
        rows.append(dict(metric=col, method="chi2", n_tests=exp_level.height, false_positives=k, fpr=k / exp_level.height, fpr_lo=lo, fpr_hi=hi, ks_uniform_p=ks))
    fpr = pl.DataFrame(rows)
    fpr.write_csv(RAW.parent / "aa_fpr.csv")

    # Family-wise error over the 5 primary tests (Welch or delta, one per metric) per experiment.
    prim = res.filter(pl.col("method").is_in(["welch", "delta"])).sort("experiment_id", "metric")
    fam = prim.group_by("experiment_id", maintain_order=True).agg(pl.col("p_value"))
    unc = holm_rej = bh_rej = 0
    for ps in fam["p_value"].to_list():
        p = np.array(ps)
        unc += bool((p < args.alpha).any())
        holm_rej += bool((holm(p) < args.alpha).any())
        bh_rej += bool((benjamini_hochberg(p) < args.alpha).any())
    n_fam = fam.height

    summary = dict(
        config=vars(args),
        n_events=n_events,
        n_analysis_units=int(stats["n"].sum()),
        timings_s=dict(generate=t_gen, load=t_load, user_metrics_sql=t_user, sufficient_stats_sql=t_stats, analyze_python=t_analyze),
        events_per_second_user_metrics=n_events / t_user,
        fpr=fpr.to_dicts(),
        family_of_5=dict(
            n_families=n_fam,
            fwer_uncorrected=unc / n_fam,
            fwer_holm=holm_rej / n_fam,
            fwer_bh=bh_rej / n_fam,
        ),
    )
    path = write_json("aa_summary.json", summary)
    print(fpr)
    print(summary["family_of_5"], summary["timings_s"])
    print("wrote", path)

    plot(fpr, res, n_fam, args.alpha)


if __name__ == "__main__":
    main()
