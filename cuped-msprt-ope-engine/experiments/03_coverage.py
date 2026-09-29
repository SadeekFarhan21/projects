"""95 percent interval coverage with a nonzero true effect, for each A/B method.

welch   revenue per user, zero-inflated log-normal, true lift known in closed form
delta   clicks per session ratio metric, true difference p_t - p_c
cuped   outcome correlated with a pre-period covariate, true shift known

    uv run python experiments/03_coverage.py --sims 1000
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import polars as pl

from _common import RESULTS, wilson, write_json
from expope.stats import cuped_ttest, delta_method_ratio_arrays, welch_ttest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.n
    hits = {"welch_revenue": 0, "delta_ctr": 0, "cuped": 0, "welch_same_data_as_cuped": 0}
    widths = {k: [] for k in hits}

    # Revenue: buy w.p. p, spend ~ LogNormal(mu, s). Treatment raises p.
    p_c, p_t, mu, s = 0.10, 0.11, 3.0, 1.0
    mean_spend = math.exp(mu + s * s / 2)
    true_rev = (p_t - p_c) * mean_spend
    # CTR: sessions ~ 1 + Poisson(4), clicks ~ Binomial(sessions, p).
    ctr_c, ctr_t = 0.20, 0.21
    # CUPED: y = 10 + 2x + e, treatment adds 0.1.
    cuped_eff = 0.1

    for _ in range(args.sims):
        rc = (rng.random(n) < p_c) * rng.lognormal(mu, s, n)
        rt = (rng.random(n) < p_t) * rng.lognormal(mu, s, n)
        r = welch_ttest(rc, rt)
        hits["welch_revenue"] += r.covers(true_rev)
        widths["welch_revenue"].append(r.ci_high - r.ci_low)

        sc, st_ = 1 + rng.poisson(4, n), 1 + rng.poisson(4, n)
        r = delta_method_ratio_arrays(rng.binomial(sc, ctr_c), sc, rng.binomial(st_, ctr_t), st_)
        hits["delta_ctr"] += r.covers(ctr_t - ctr_c)
        widths["delta_ctr"].append(r.ci_high - r.ci_low)

        xc, xt = rng.normal(0, 1, n), rng.normal(0, 1, n)
        yc = 10 + 2 * xc + rng.normal(0, 1, n)
        yt = 10 + 2 * xt + rng.normal(0, 1, n) + cuped_eff
        r, _ = cuped_ttest(yc, xc, yt, xt)
        hits["cuped"] += r.covers(cuped_eff)
        widths["cuped"].append(r.ci_high - r.ci_low)
        r = welch_ttest(yc, yt)
        hits["welch_same_data_as_cuped"] += r.covers(cuped_eff)
        widths["welch_same_data_as_cuped"].append(r.ci_high - r.ci_low)

    rows = []
    for k, h in hits.items():
        lo, hi = wilson(h, args.sims)
        rows.append(dict(method=k, sims=args.sims, n_per_arm=n, covered=h, coverage=h / args.sims, cov_lo=lo, cov_hi=hi,
                         mean_ci_width=float(np.mean(widths[k]))))
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS / "coverage.csv")
    write_json("coverage_summary.json", dict(config=vars(args), truths=dict(revenue=true_rev, ctr=ctr_t - ctr_c, cuped=cuped_eff), rows=rows))
    print(df)


if __name__ == "__main__":
    main()
