import numpy as np
import pandas as pd

from marketpred.universe import eligibility, is_excluded


def test_exclusions():
    bases = {"BTC", "ETH", "J", "SUPER"}  # note: "J" makes JUP look leveraged
    assert is_excluded("USDCUSDT", bases)
    assert is_excluded("WBTCUSDT", bases)
    assert is_excluded("BTCUPUSDT", bases)
    assert is_excluded("ETHBEARUSDT", bases)
    assert not is_excluded("BTCUSDT", bases)
    assert not is_excluded("SUPERUSDT", bases)
    assert not is_excluded("JUPUSDT", {"BTC"})
    assert not is_excluded("LUNAUSDT#2", bases)


def _wide(n_days=100, n_sym=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days)
    cols = [f"S{i}USDT" for i in range(n_sym)]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    dvol = pd.DataFrame(rng.uniform(2e6, 1e8, (n_days, n_sym)), index=idx, columns=cols)
    return close, dvol


def test_history_topn_and_min_names():
    close, dvol = _wide()
    close.iloc[:50, 0] = np.nan  # S0 lists late
    m = eligibility(close, dvol, top_n=25, min_history=60, min_names=20)
    assert not m.iloc[:59].any().any()          # nobody has 60 bars yet
    assert m.iloc[59:].sum(axis=1).eq(25).all()  # top-25 cap binds
    assert not m.iloc[:109, 0].any()             # S0 needs 60 bars of its own


def test_min_dvol_and_thin_days():
    close, dvol = _wide()
    dvol.iloc[:, :15] = 1e5  # half the names are illiquid
    m = eligibility(close, dvol, top_n=50, min_history=10, min_names=20)
    assert not m.any().any()  # only 15 names qualify < 20, so all days skipped


def test_no_lookahead_in_universe():
    close, dvol = _wide()
    m1 = eligibility(close, dvol)
    dvol2 = dvol.copy()
    dvol2.iloc[80:] = dvol2.iloc[80:] * np.random.default_rng(1).uniform(0, 10, dvol2.iloc[80:].shape)
    m2 = eligibility(close, dvol2)
    pd.testing.assert_frame_equal(m1.iloc[:80], m2.iloc[:80])
