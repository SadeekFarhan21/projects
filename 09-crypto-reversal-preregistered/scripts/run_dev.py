"""Development-period study: walk-forward over 2020H1..2024H1 for every
pre-registered trial, then freeze the selection for the holdout.

Never reads data after 2024-06-30 (the dataset is truncated before any
feature is computed).
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketpred.data import PROJECT_ROOT, load_panel
from marketpred.features import FEATURES, build_dataset
from marketpred.metrics import ANN, deflated_sharpe
from marketpred.models import RidgeModel, registry
from marketpred.study import BASE_BPS, evaluate
from marketpred.walkforward import DEV_END, DEV_START, half_year_folds, walk_forward

OUT = PROJECT_ROOT / "results" / "dev"
SOURCE = "Source: Binance spot daily klines (data.binance.vision), own calculations."


def data_report(panel, ds, dev_dates) -> dict:
    m = ds.mask.loc[dev_dates[0]:dev_dates[-1]]
    size = m.sum(axis=1)
    size = size[size > 0]
    held = ds.long[ds.long.index.get_level_values("date") >= dev_dates[0]]
    r = ds.raw["ret_1d"].where(ds.mask).stack()
    r = r[r.index.get_level_values(0) >= dev_dates[0]]
    ext = r.abs().sort_values(ascending=False).head(5)
    return {
        "archive_symbols_usdt": int(panel.symbol.nunique()),
        "episodes_after_relisting_split": int(ds.mask.shape[1]),
        "dev_days_with_universe": int(len(size)),
        "universe_size_min_median_max": [int(size.min()), float(size.median()), int(size.max())],
        "distinct_symbols_ever_in_dev_universe": int(m.any().sum()),
        "member_rows_dev": int(len(held)),
        "member_rows_missing_fwd_ret": int(held["fwd_ret"].isna().sum()),
        "largest_abs_daily_returns_in_universe": {f"{d.date()} {s}": float(r.loc[(d, s)]) for d, s in ext.index},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {}
    t0 = time.perf_counter()
    panel = load_panel()
    panel = panel[panel.date <= DEV_END]  # hard cut: nothing from the holdout
    timings["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ds = build_dataset(panel, end=DEV_END)
    timings["build_dataset_s"] = time.perf_counter() - t0
    folds = half_year_folds()
    dev_dates = pd.date_range(DEV_START, DEV_END)
    (OUT / "data_report.json").write_text(json.dumps(data_report(panel, ds, dev_dates), indent=2))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fwd = ds.long["fwd_ret"]
    rows, fold_ic, nets, coefs = [], [], {}, []
    for mname, factory in registry().items():
        t0 = time.perf_counter()
        pred, fitted = walk_forward(factory, ds.long, folds)
        timings[f"walk_forward_{mname}_s"] = time.perf_counter() - t0
        ic, res = evaluate(pred, fwd)
        for f in folds:
            sl = ic.loc[f.test_start:f.test_end]
            fold_ic.append({"model": mname, "fold": f.name, "ic_mean": sl.mean(), "days": len(sl)})
        if isinstance(fitted[0], RidgeModel):
            for f, m in zip(folds, fitted):
                coefs.append({"model": mname, "fold": f.name, **m.coef})
        for cname, (bt, s) in res.items():
            trial = f"{mname}|{cname}"
            nets[trial] = bt.net(BASE_BPS)
            rows.append({"trial": trial, "model": mname, "construction": cname, **s})
            with open(OUT / "trials.jsonl", "a") as fh:
                fh.write(json.dumps({"run_id": run_id, "trial": trial, "period": [DEV_START, DEV_END],
                                     "sharpe_net": s["sharpe_net"], "ic_mean": s["ic_mean"]}) + "\n")
        print(f"{mname:11s} IC {res['rank'][1]['ic_mean']:+.4f} (t {res['rank'][1]['ic_t_nw']:+.2f})  "
              f"SR net rank {res['rank'][1]['sharpe_net']:+.2f}  quintile {res['quintile'][1]['sharpe_net']:+.2f}")

    summary = pd.DataFrame(rows).sort_values("sharpe_net", ascending=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(fold_ic).to_csv(OUT / "fold_ic.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "ridge_coefs.csv", index=False)
    pd.DataFrame(nets).to_csv(OUT / "daily_net_returns.csv")

    # Selection rule (PREREGISTRATION.md section 9) and deflated Sharpe.
    best = summary.sort_values(["sharpe_net", "ic_mean"], ascending=False).iloc[0]
    n_trials = sum(1 for _ in open(OUT / "trials.jsonl"))
    dsr = deflated_sharpe(nets[best.trial], n_trials, summary["sr_per_period_net"].tolist())
    selection = {"run_id": run_id, "selected_trial": best.trial, "model": best.model,
                 "construction": best.construction, "dev_sharpe_net": float(best.sharpe_net),
                 "dev_ic_mean": float(best.ic_mean), "dev_psr": float(best.psr_net),
                 "trials_logged_total": n_trials, "trials_this_run": len(summary),
                 "deflated_sharpe": dsr, "cost_bps": BASE_BPS}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))
    (OUT / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(selection, indent=2))
    plots(summary, nets, pd.DataFrame(fold_ic), best.trial)


def plots(summary, nets, fold_ic, best):
    # 1. Cumulative net return of every trial, two highlighted.
    fig, ax = plt.subplots(figsize=(9, 5))
    for t, r in nets.items():
        if t not in (best, "rev_1d|rank"):
            ax.plot(r.cumsum(), color="0.8", lw=0.8)
    ax.plot(nets["rev_1d|rank"].cumsum(), color="tab:blue", lw=1.6, label="rev_1d, rank weights (H1 signal)")
    if best != "rev_1d|rank":
        ax.plot(nets[best].cumsum(), color="tab:orange", lw=1.6, label=f"dev-selected: {best}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Development walk-forward, cumulative net return at {BASE_BPS:.0f} bps (gray: other trials)")
    ax.set_ylabel("cumulative sum of daily net returns")
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.01, 0.01, SOURCE, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "dev_cumulative_net.png", dpi=150)
    plt.close(fig)

    # 2. Mean IC by model and fold.
    piv = fold_ic.pivot(index="model", columns="fold", values="ic_mean").loc[list(registry())]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lim = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean daily rank IC")
    ax.set_title("Out-of-sample mean rank IC per walk-forward fold")
    fig.text(0.01, 0.01, SOURCE, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "dev_ic_by_fold.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
