import math

import numpy as np
import pandas as pd
from scipy import stats

from marketpred import metrics as M


def test_newey_west_lag0_matches_classic_t():
    x = np.random.default_rng(0).normal(0.1, 1, 500)
    t_classic = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert math.isclose(M.newey_west_t(x, 0), t_classic, rel_tol=1e-9)


def test_newey_west_penalizes_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=2000)
    x = 0.05 + np.convolve(e, np.ones(10) / 10, mode="same")
    assert M.newey_west_t(x, 10) < M.newey_west_t(x, 0)


def test_psr_half_at_threshold_and_monotone():
    r = pd.Series(np.random.default_rng(2).normal(0.001, 0.02, 1000))
    sr = r.mean() / r.std(ddof=1)
    assert math.isclose(M.psr(r, sr), 0.5, abs_tol=1e-12)
    assert M.psr(r, 0.0) > M.psr(r, 0.05)


def test_expected_max_sharpe_and_dsr():
    assert M.expected_max_sharpe(1, 0.01) == 0.0
    a, b = M.expected_max_sharpe(10, 0.001), M.expected_max_sharpe(100, 0.001)
    assert 0 < a < b
    # Monte Carlo check: E[max of N normals with sd s] close to the formula.
    rng = np.random.default_rng(3)
    s, n = 0.03, 50
    mc = rng.normal(0, s, (20000, n)).max(axis=1).mean()
    assert abs(mc - M.expected_max_sharpe(n, s**2)) / mc < 0.05
    r = pd.Series(rng.normal(0.002, 0.02, 1500))
    d = M.deflated_sharpe(r, 50, list(rng.normal(0, 0.03, 50)))
    assert d["dsr"] < M.psr(r)


def test_sharpe_and_drawdown():
    r = pd.Series([0.01, -0.01] * 50)
    assert abs(M.sharpe(r)) < 1e-9
    assert math.isclose(M.max_drawdown(pd.Series([0.1, -0.5, 0.2])), -0.5)


def test_daily_ic_perfect_and_inverse():
    idx = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=3), list("abcdefghijkl")], names=["date", "symbol"])
    f = pd.Series(np.tile(np.arange(12.0), 3), index=idx)
    ic = M.daily_ic(f, f * 2)
    assert np.allclose(ic, 1.0)
    assert np.allclose(M.daily_ic(f, -f), -1.0)
    rho = stats.spearmanr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]).statistic
    g = pd.Series(range(1, 11), index=pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")], list("abcdefghij")], names=["date", "symbol"]))
    y = pd.Series([2, 1, 4, 3, 6, 5, 8, 7, 10, 9], index=g.index, dtype=float)
    assert math.isclose(M.daily_ic(g.astype(float), y).iloc[0], rho)
