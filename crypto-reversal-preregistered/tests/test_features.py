import numpy as np
import pandas as pd

from conftest import make_panel
from marketpred.features import FEATURES, build_dataset, cs_rank, split_relistings


def test_split_relistings():
    d1 = pd.date_range("2022-01-01", periods=5)
    d2 = pd.date_range("2022-02-01", periods=5)
    p = pd.DataFrame({"date": list(d1) + list(d2), "symbol": "LUNAUSDT", "close": 1.0})
    out = split_relistings(p)
    assert sorted(out.symbol.unique()) == ["LUNAUSDT", "LUNAUSDT#2"]
    assert (out[out.date >= "2022-02-01"].symbol == "LUNAUSDT#2").all()


def test_cs_rank_bounds_and_mask():
    idx = pd.date_range("2020-01-01", periods=2)
    w = pd.DataFrame([[1.0, 2.0, 3.0, np.nan], [3.0, 1.0, np.nan, 5.0]], index=idx, columns=list("abcd"))
    mask = pd.DataFrame(True, index=idx, columns=list("abcd"))
    mask.iloc[0, 3] = False
    r = cs_rank(w, mask)
    assert r.iloc[0].tolist()[:3] == [-0.5, 0.0, 0.5] and np.isnan(r.iloc[0, 3])
    assert r.iloc[1, 2] == 0.0  # member with NaN value -> median


def test_forward_return_alignment(panel):
    ds = build_dataset(panel)
    close = panel.pivot(index="date", columns="symbol", values="close")
    d, s = ds.long.index[100]
    exp = close.loc[d + pd.Timedelta(days=1), s] / close.loc[d, s] - 1
    assert np.isclose(ds.long.loc[(d, s), "fwd_ret"], exp)


def test_features_do_not_look_ahead():
    """Changing prices after date T must not change any feature or the
    universe at dates <= T - 1 (fwd_ret at T-1 legitimately uses T)."""
    p1 = make_panel(seed=3)
    p2 = p1.copy()
    T = pd.Timestamp("2019-09-01")
    late = p2.date >= T
    p2.loc[late, ["close", "high", "quote_volume"]] *= np.random.default_rng(9).uniform(0.5, 2, (late.sum(), 1))
    a = build_dataset(p1).long
    b = build_dataset(p2).long
    cut = T - pd.Timedelta(days=1)
    # <= cut, not < cut: with a strict inequality a one-day lead (a feature at
    # T-1 using the bar at T) slipped through this test (found in verification).
    a = a[a.index.get_level_values("date") <= cut]
    b = b[b.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def test_truncation_end():
    ds = build_dataset(make_panel(), end="2019-06-30")
    assert ds.long.index.get_level_values("date").max() <= pd.Timestamp("2019-06-30")
    last = ds.long.xs(pd.Timestamp("2019-06-30"), level="date")
    assert last["fwd_ret"].isna().all()  # the next day was cut off
