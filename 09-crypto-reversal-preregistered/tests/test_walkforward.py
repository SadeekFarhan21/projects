import numpy as np
import pandas as pd

from conftest import make_panel
from marketpred.features import build_dataset
from marketpred.metrics import daily_ic
from marketpred.models import registry
from marketpred.portfolio import backtest, weights_rank
from marketpred.walkforward import EMBARGO_DAYS, Fold, half_year_folds, walk_forward


def test_dev_folds():
    f = half_year_folds()
    assert len(f) == 9 and f[0].name == "2020H1" and f[-1].name == "2024H1"
    assert f[-1].test_end == pd.Timestamp("2024-06-30")
    for a, b in zip(f, f[1:]):
        assert b.test_start == a.test_end + pd.Timedelta(days=1)
    assert (f[0].test_start - f[0].train_end).days == EMBARGO_DAYS + 1


class Spy:
    seen_max = []

    def fit(self, train):
        Spy.seen_max.append(train.index.get_level_values("date").max())
        return self

    def predict(self, test):
        return pd.Series(0.0, index=test.index)


def test_walk_forward_respects_train_end():
    ds = build_dataset(make_panel(n_days=500))
    folds = [Fold("a", pd.Timestamp("2019-08-01"), pd.Timestamp("2019-10-31")),
             Fold("b", pd.Timestamp("2019-11-01"), pd.Timestamp("2020-01-31"))]
    Spy.seen_max.clear()
    pred, _ = walk_forward(Spy, ds.long, folds)
    assert all(s <= f.train_end for s, f in zip(Spy.seen_max, folds))
    d = pred.index.get_level_values("date")
    assert d.min() >= folds[0].test_start and d.max() <= folds[-1].test_end


def test_end_to_end_recovers_planted_reversal():
    ds = build_dataset(make_panel(n_days=500, reversal=0.3, seed=5))
    folds = [Fold("t", pd.Timestamp("2019-09-01"), pd.Timestamp("2020-05-14"))]
    for name in ["rev_1d", "ridge_a10", "gbm_d3"]:
        pred, _ = walk_forward(registry()[name], ds.long, folds)
        ic = daily_ic(pred, ds.long["fwd_ret"])
        assert ic.mean() > 0.1, name
    bt = backtest(weights_rank(pred), ds.long["fwd_ret"].reindex(pred.index))
    assert bt.daily["gross"].mean() > 0


def test_no_signal_means_no_edge():
    ds = build_dataset(make_panel(n_days=500, reversal=0.0, seed=6))
    folds = [Fold("t", pd.Timestamp("2019-09-01"), pd.Timestamp("2020-05-14"))]
    pred, _ = walk_forward(registry()["ridge_a10"], ds.long, folds)
    ic = daily_ic(pred, ds.long["fwd_ret"])
    assert abs(ic.mean()) < 0.05
