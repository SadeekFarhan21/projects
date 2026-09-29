import numpy as np
import pandas as pd

from marketpred.portfolio import backtest, weights_quintile, weights_rank


def _idx(dates, syms):
    return pd.MultiIndex.from_product([pd.to_datetime(dates), syms], names=["date", "symbol"])


def test_rank_weights_neutral_and_gross_one():
    f = pd.Series(np.random.default_rng(0).normal(size=40), index=_idx(["2020-01-01", "2020-01-02"], list("abcdefghijklmnopqrst")))
    w = weights_rank(f)
    g = w.groupby(level="date")
    assert np.allclose(g.sum(), 0) and np.allclose(g.apply(lambda x: x.abs().sum()), 1)


def test_quintile_weights():
    f = pd.Series(np.arange(10.0), index=_idx(["2020-01-01"], list("abcdefghij")))
    w = weights_quintile(f)
    assert w.loc[(slice(None), ["i", "j"])].eq(0.25).all()
    assert w.loc[(slice(None), ["a", "b"])].eq(-0.25).all()
    assert w.abs().sum() == 1.0 and w.sum() == 0.0


def test_backtest_pnl_turnover_and_missing_fwd():
    idx = _idx(["2020-01-01", "2020-01-02"], ["a", "b"])
    w = pd.Series([0.5, -0.5, -0.5, 0.5], index=idx)
    r = pd.Series([0.10, -0.10, 0.02, np.nan], index=idx)  # b delists on day 2
    bt = backtest(w, r)
    assert np.allclose(bt.daily["gross"], [0.10, -0.01])
    assert np.allclose(bt.daily["turnover"], [1.0, 2.0])
    assert np.allclose(bt.net(100), [0.10 - 0.01, -0.01 - 0.02])


def test_lag_shifts_weights():
    idx = _idx(["2020-01-01", "2020-01-02"], ["a", "b"])
    w = pd.Series([0.5, -0.5, 0.0, 0.0], index=idx)
    r = pd.Series([0.1, 0.1, 0.2, -0.2], index=idx)
    bt = backtest(w, r, lag=1)
    assert np.allclose(bt.daily["gross"], [0.0, 0.2])
