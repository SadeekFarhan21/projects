"""IV solver and pricer benchmark: throughput, round-trip error, iteration counts.

Draws a synthetic universe that is wider than anything Deribit quotes
(log moneyness in [-1, 1], one hour to three years, vol 2% to 300%), prices it
with Black-76, inverts, and reprices. Thread count comes from NUMBA_NUM_THREADS.

usage: uv run python experiments/01_iv_benchmark.py
"""

from __future__ import annotations

import json
import time

import numba
import numpy as np

from optvol import black76 as b
from optvol import iv
from optvol.data import RESULTS

N = 2_000_000
REPS = 5


def universe(n, seed=0):
    rng = np.random.default_rng(seed)
    F = 100.0
    K = F * np.exp(rng.uniform(-1, 1, n))
    T = rng.uniform(1 / (365 * 24), 3, n)
    s = rng.uniform(0.02, 3.0, n)
    cp = rng.random(n) < 0.5
    return F, K, T, s, cp


def best_time(fn, reps=REPS):
    fn()  # warm up JIT and caches
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def run(threads):
    numba.set_num_threads(threads)
    F, K, T, s, cp = universe(N)
    Fa = np.full(N, F)
    p = b.price(Fa, K, T, s, cp)
    t_price, _ = best_time(lambda: b.price(Fa, K, T, s, cp))
    t_greeks, _ = best_time(lambda: b.greeks(Fa, K, T, s, cp))
    t_iv, t_iv_med = best_time(lambda: iv.implied_vol(p, Fa, K, T, cp))
    sig, st, it = iv.implied_vol(p, Fa, K, T, cp, return_info=True)
    ok = np.isin(st, [iv.OK, iv.BRENT])
    p2 = b.price(Fa[ok], K[ok], T[ok], sig[ok], cp[ok])
    err = np.abs(p2 - p[ok]) / F
    vega = b.greeks(Fa, K, T, s, cp)["vega"]
    good = ok & (vega > 1e-3)
    return dict(
        threads=threads, n=N,
        price_per_s=N / t_price, greeks_per_s=N / t_greeks,
        iv_per_s_best=N / t_iv, iv_per_s_median=N / t_iv_med,
        share_ok=float(ok.mean()),
        share_at_intrinsic=float((st == iv.AT_INTRINSIC).mean()),
        n_brent=int((st == iv.BRENT).sum()),
        n_not_converged=int((st == iv.NOT_CONVERGED).sum()),
        max_price_err_rel_F=float(err.max()),
        p99_price_err_rel_F=float(np.quantile(err, 0.99)),
        max_vol_err_vega_gt_1e3=float(np.max(np.abs(sig[good] - s[good]))),
        mean_iters=float(it[ok].mean()), p99_iters=float(np.quantile(it[ok], 0.99)),
        max_iters=int(it[ok].max()),
    )


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    max_threads = numba.config.NUMBA_NUM_THREADS
    out = [run(1)]
    if max_threads > 1:
        out.append(run(max_threads))
    for r in out:
        print(json.dumps(r, indent=1))
    (RESULTS / "01_iv_benchmark.json").write_text(json.dumps(out, indent=1))
