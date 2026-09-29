import json

import numpy as np
import pytest

from qrp.cli import main
from qrp.features.ops import op_xs_rank, rolling_mean, rolling_std
from qrp.features.spec import FeatureSpec, compute_features
from qrp.panel import synthetic_panel
from qrp.research import DEFAULTS, _merge, run_research, universe_mask
from qrp.tracker import load_runs


def test_rolling_matches_naive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    x[7, 1] = np.nan
    m, s = rolling_mean(x, 5), rolling_std(x, 5)
    for t in range(50):
        for i in range(3):
            w = x[max(0, t - 4): t + 1, i]
            if t < 4 or np.isnan(w).any():
                assert np.isnan(m[t, i]) and np.isnan(s[t, i])
            else:
                assert m[t, i] == pytest.approx(w.mean(), abs=1e-12)
                assert s[t, i] == pytest.approx(w.std(ddof=1), abs=1e-10)


def test_xs_rank_range_and_nan():
    x = np.array([[3.0, 1.0, np.nan, 2.0]])
    np.testing.assert_allclose(op_xs_rank(x)[0, [0, 1, 3]], [0.5, -0.5, 0.0])
    assert np.isnan(op_xs_rank(x)[0, 2])


def test_spec_validation():
    with pytest.raises(ValueError, match="unknown op"):
        FeatureSpec.from_dict({"name": "a", "op": "nope"})
    with pytest.raises(ValueError, match="explicit inputs"):
        FeatureSpec.from_dict({"name": "a", "op": "xs_rank"})
    p = synthetic_panel(T=50, N=5)
    cyc = [FeatureSpec("a", "lag", ("b",), {}), FeatureSpec("b", "lag", ("a",), {})]
    with pytest.raises(ValueError, match="cycle"):
        compute_features(p, cyc)
    # declaration order does not matter
    specs = [FeatureSpec("r", "xs_rank", ("m",), {}), FeatureSpec("m", "momentum", ("close",),
                                                                  {"window": 5})]
    assert set(compute_features(p, specs)) == {"r", "m"}


def test_universe_is_top_k_and_listed():
    p = synthetic_panel(T=200, N=30, seed=4, missing=0.02)
    U = universe_mask(p, top_k=10, min_history=30, liquidity_window=10)
    assert U.sum(axis=1).max() <= 10
    assert not (U & ~np.isfinite(p["close"])).any()
    assert not U[:29].any()  # nothing has 30 bars yet


def _cfg(tmp_path, **over):
    base = {
        "run": {"name": "t", "runs_root": str(tmp_path / "runs")},
        "features": [{"name": "mom", "op": "momentum", "window": 10},
                     {"name": "mom_r", "op": "xs_rank", "inputs": ["mom"]},
                     {"name": "vol", "op": "volatility", "window": 10},
                     {"name": "vol_r", "op": "xs_rank", "inputs": ["vol"]}],
        "model": {"features": ["mom_r", "vol_r"]},
        "cv": {"min_train_days": 120, "n_splits": 3},
        "universe": {"top_k": 20, "min_history": 20},
    }
    return _merge(DEFAULTS, _merge(base, over))


def test_pipeline_end_to_end_and_tracker(tmp_path, capsys):
    p = synthetic_panel(T=400, N=30, seed=9, missing=0.01)
    out = run_research(_cfg(tmp_path), panel=p)
    m = out["metrics"]
    assert m["leak_free"] is True and m["n_splits"] == 3
    assert not out["weights"][: 120].any()        # no positions before the first test block
    assert np.allclose(out["weights"].sum(axis=1), 0, atol=1e-12)  # dollar neutral
    run_dir = out["run"].dir
    for f in ("config.json", "meta.json", "metrics.json", "artifacts/equity.parquet"):
        assert (run_dir / f).exists()

    out2 = run_research(_cfg(tmp_path, costs={"fee_bps": 0.0}), panel=p)
    assert out2["metrics"]["sharpe"] > m["sharpe"]  # removing costs can only help

    runs = load_runs(tmp_path / "runs")
    assert len(runs) == 2
    main(["runs", "compare", runs[0]["id"], runs[1]["id"], "--root", str(tmp_path / "runs")])
    txt = capsys.readouterr().out
    assert "config.costs.fee_bps" in txt and "metric.sharpe" in txt
    assert "config.features" not in txt  # identical keys are hidden
    main(["runs", "list", "--root", str(tmp_path / "runs"), "--sort", "sharpe"])
    assert runs[0]["id"] in capsys.readouterr().out


def test_planted_signal_is_found(tmp_path):
    """Make tomorrow's return depend on today's 1-day reversal; the ridge model must find it."""
    rng = np.random.default_rng(0)
    T, N = 600, 40
    r = np.zeros((T, N))
    eps = rng.normal(0, 0.02, (T, N))
    r[0] = eps[0]
    for t in range(1, T):
        r[t] = -0.3 * (r[t - 1] - r[t - 1].mean()) + eps[t]
    p = synthetic_panel(T=T, N=N)
    p.fields["close"] = 100 * np.cumprod(1 + r, axis=0)
    p.fields["quote_volume"] = np.full((T, N), 1e6)
    cfg = _cfg(tmp_path, features=[{"name": "rev", "op": "returns", "window": 1},
                                   {"name": "rev_r", "op": "xs_rank", "inputs": ["rev"]}],
               model={"features": ["rev_r"], "horizon": 1},
               backtest={"delay": 0}, costs={"fee_bps": 0.0, "slippage_bps": 0.0},
               universe={"top_k": 40})
    m = run_research(cfg, panel=p, track=False)["metrics"]
    assert m["ic"] > 0.1 and m["sharpe"] > 3
