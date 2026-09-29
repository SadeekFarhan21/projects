"""Continuous monitoring: naive t-test with peeking versus the always-valid mixture SPRT.

Each simulated experiment streams users into two arms and is checked after every batch.
The naive analyst stops at the first look with p < alpha. The mSPRT stops when its
always-valid p-value drops below alpha. Under the null (A/A) the stopping rate is the FPR.
Under an effect we also report power and the average stopping point.

    uv run python experiments/04_peeking.py --sims 2000 --max-n 10000 --batch 200
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from scipy import stats as sps

from _common import PALETTE, RESULTS, plt, save, style, wilson, write_json
from expope.stats import msprt_always_valid_p


def simulate(sims: int, max_n: int, batch: int, effect: float, tau2: float, alpha: float, rng, chunk: int = 200):
    looks = np.arange(batch, max_n + 1, batch)
    naive_first = []  # first look index where naive p < alpha, or -1
    msprt_first = []
    naive_by_look = np.zeros(len(looks))
    fixed_rej = 0
    for s in range(0, sims, chunk):
        m = min(chunk, sims - s)
        c = rng.normal(0, 1, (m, max_n))
        t = rng.normal(effect, 1, (m, max_n))
        cs_c, cs_t = np.cumsum(c, 1)[:, looks - 1], np.cumsum(t, 1)[:, looks - 1]
        ss_c, ss_t = np.cumsum(c * c, 1)[:, looks - 1], np.cumsum(t * t, 1)[:, looks - 1]
        n = looks[None, :].astype(float)
        mc, mt = cs_c / n, cs_t / n
        vc = (ss_c - n * mc * mc) / (n - 1)
        vt = (ss_t - n * mt * mt) / (n - 1)
        est = mt - mc
        var = vc / n + vt / n
        # Welch t with df from Welch-Satterthwaite.
        df = var**2 / ((vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
        p_naive = 2 * sps.t.sf(np.abs(est / np.sqrt(var)), df)
        p_msprt = msprt_always_valid_p(est, var, tau2, axis=1)
        rej_n = p_naive < alpha
        rej_m = p_msprt < alpha
        fixed_rej += int((p_naive[:, -1] < alpha).sum())
        naive_by_look += np.logical_or.accumulate(rej_n, axis=1).sum(0)
        naive_first += [int(np.argmax(r)) if r.any() else -1 for r in rej_n]
        msprt_first += [int(np.argmax(r)) if r.any() else -1 for r in rej_m]
    return looks, np.array(naive_first), np.array(msprt_first), naive_by_look / sims, fixed_rej / sims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--max-n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.1, help="mixing standard deviation, in outcome units")
    ap.add_argument("--effect", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tau2 = args.tau**2

    out = {}
    rows = []
    for label, eff in (("null", 0.0), ("effect", args.effect)):
        looks, nf, mf, naive_curve, fixed = simulate(args.sims, args.max_n, args.batch, eff, tau2, args.alpha, rng)
        k_n, k_m = int((nf >= 0).sum()), int((mf >= 0).sum())
        out[label] = dict(
            effect=eff,
            naive_peeking_reject_rate=k_n / args.sims,
            naive_ci=wilson(k_n, args.sims),
            msprt_reject_rate=k_m / args.sims,
            msprt_ci=wilson(k_m, args.sims),
            msprt_mean_stop_n=float(looks[mf[mf >= 0]].mean()) if k_m else None,
            naive_mean_stop_n=float(looks[nf[nf >= 0]].mean()) if k_n else None,
            fixed_horizon_reject_rate=fixed,
        )
        for j, L in enumerate(looks):
            rows.append(dict(scenario=label, n_per_arm=int(L), naive_cum_reject=float(naive_curve[j]),
                             msprt_cum_reject=float(((mf >= 0) & (mf <= j)).mean())))
        print(label, out[label])
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "peeking.csv")
    write_json("peeking_summary.json", dict(config=vars(args), n_looks=len(looks), **out))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    g = df.filter(pl.col("scenario") == "null")
    ax.plot(g["n_per_arm"], g["naive_cum_reject"], color=PALETTE[1], lw=2, label="Naive t test, stop at first p below 0.05")
    ax.plot(g["n_per_arm"], g["msprt_cum_reject"], color=PALETTE[0], lw=2, label="Mixture SPRT, always valid")
    ax.axhline(args.alpha, color="#555", ls="--", lw=1)
    style(ax, f"False positives under peeking, A/A, {len(looks)} looks", "Users per arm at the look", "Share of A/A tests stopped")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "peeking_fpr.png")


if __name__ == "__main__":
    main()
