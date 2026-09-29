"""One-shot holdout evaluation (PREREGISTRATION.md section 11).

Refuses to run if results/holdout/LOCK.json exists. The lock is written
before any holdout number is computed, so a crashed run still counts as the
one touch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from marketpred.data import PROJECT_ROOT, load_panel
from marketpred.features import build_dataset
from marketpred.metrics import sharpe
from marketpred.models import registry
from marketpred.study import BASE_BPS, COST_GRID, evaluate
from marketpred.walkforward import HOLDOUT_END, HOLDOUT_START, Fold, walk_forward

OUT = PROJECT_ROOT / "results" / "holdout"
LOCK = OUT / "LOCK.json"
SOURCE = "Source: Binance spot daily klines (data.binance.vision), own calculations."


def main():
    if LOCK.exists():
        sys.exit(f"refusing: holdout already evaluated, see {LOCK}")
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((PROJECT_ROOT / "results" / "dev" / "selection.json").read_text())
    prereg = (PROJECT_ROOT / "PREREGISTRATION.md").read_bytes()
    lock = {"started_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": hashlib.sha256(prereg).hexdigest(),
            "selection": sel, "status": "started"}
    LOCK.write_text(json.dumps(lock, indent=2))

    panel = load_panel()
    ds = build_dataset(panel, end=HOLDOUT_END)
    fold = Fold("holdout", pd.Timestamp(HOLDOUT_START), pd.Timestamp(HOLDOUT_END))
    fwd = ds.long["fwd_ret"]

    # Every model is refit once on data up to fold.train_end (2024-06-25).
    # Only rev_1d (H1) and the selected trial (H2) are confirmatory; the rest
    # are reported as post-hoc context and drive no decision.
    rows, nets, extra = [], {}, {}
    for mname, factory in registry().items():
        pred, _ = walk_forward(factory, ds.long, [fold])
        ic, res = evaluate(pred, fwd)
        for cname, (bt, s) in res.items():
            trial = f"{mname}|{cname}"
            nets[trial] = bt.net(BASE_BPS)
            rows.append({"trial": trial, **s})
        if mname in ("rev_1d", sel["model"]):
            _, lag_res = evaluate(pred, fwd, lag=1)
            extra[mname] = {"ic": ic, "res": res, "lag1": lag_res}

    allt = pd.DataFrame(rows).set_index("trial")
    allt.to_csv(OUT / "all_trials_posthoc.csv")
    pd.DataFrame(nets).to_csv(OUT / "daily_net_returns.csv")

    h1 = extra["rev_1d"]["res"]["rank"][1]
    h1_verdict = ("supported" if h1["ic_mean"] > 0 and h1["ic_t_nw"] > 2 else
                  "rejected" if h1["ic_mean"] <= 0 else "inconclusive")
    s2 = extra[sel["model"]]["res"][sel["construction"]][1]
    h2_verdict = "supported" if s2["sharpe_net"] > 0 and s2["psr_net"] > 0.95 else "not supported"

    dev = pd.read_csv(PROJECT_ROOT / "results" / "dev" / "summary.csv").set_index("trial")
    both = dev[["sharpe_net"]].join(allt[["sharpe_net"]], lsuffix="_dev", rsuffix="_holdout")
    report = {
        "period": [HOLDOUT_START, HOLDOUT_END],
        "train_end": str(fold.train_end.date()),
        "H1_rev_1d": {"verdict": h1_verdict, **{k: h1[k] for k in
                      ("ic_mean", "ic_t_nw", "ic_ir_ann", "ic_hit_rate", "days")}},
        "H1_rev_1d_rank_portfolio": h1,
        "H2_selected": {"trial": sel["selected_trial"], "verdict": h2_verdict, **s2},
        "H2_selected_lag1": extra[sel["model"]]["lag1"][sel["construction"]][1],
        "rev_1d_rank_lag1": extra["rev_1d"]["lag1"]["rank"][1],
        "dev_vs_holdout_sharpe_spearman_posthoc": float(both.corr(method="spearman").iloc[0, 1]),
        "universe_size_median": float(ds.mask.loc[HOLDOUT_START:HOLDOUT_END].sum(axis=1).median()),
    }
    (OUT / "holdout_report.json").write_text(json.dumps(report, indent=2, default=float))
    lock.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                h1=h1_verdict, h2=h2_verdict)
    LOCK.write_text(json.dumps(lock, indent=2))
    print(json.dumps(report, indent=2, default=float))

    # Plot: equity of the two confirmatory portfolios, gross and net.
    fig, ax = plt.subplots(figsize=(9, 5))
    for mname, cname, color in [("rev_1d", "rank", "tab:blue"), (sel["model"], sel["construction"], "tab:orange")]:
        bt = extra[mname]["res"][cname][0]
        ax.plot(bt.daily["gross"].cumsum(), color=color, lw=1, ls="--", label=f"{mname}|{cname} gross")
        ax.plot(bt.net(BASE_BPS).cumsum(), color=color, lw=1.6, label=f"{mname}|{cname} net {BASE_BPS:.0f} bps")
        if mname == sel["model"] and cname == sel["construction"] and mname == "rev_1d":
            break
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Holdout (2024-07 to 2026-08), evaluated once")
    ax.set_ylabel("cumulative sum of daily returns")
    ax.legend(frameon=False, loc="best")
    fig.text(0.01, 0.01, SOURCE, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "holdout_equity.png", dpi=150)
    plt.close(fig)

    # Plot: Sharpe vs cost for the two confirmatory portfolios.
    fig, ax = plt.subplots(figsize=(7, 4))
    for mname, cname, color in [("rev_1d", "rank", "tab:blue"), (sel["model"], sel["construction"], "tab:orange")]:
        s = extra[mname]["res"][cname][1]
        ax.plot(COST_GRID, [s[f"sharpe_net_{int(c)}bps"] for c in COST_GRID], "o-", color=color, label=f"{mname}|{cname}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("one-way cost per unit turnover (bps)")
    ax.set_ylabel("annualized net Sharpe")
    ax.set_title("Holdout Sharpe vs trading cost")
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, SOURCE, fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "holdout_cost_sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
