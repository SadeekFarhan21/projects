"""Look-ahead bias tests. A feature that peeks must fail; every shipped op must pass."""

import numpy as np
import pytest

from qrp.features.ops import CAUSAL_OPS, LEAKY_OPS
from qrp.features.spec import (FeatureSpec, LookAheadError, assert_causal, audit_causality,
                               compute_features)
from qrp.panel import synthetic_panel
from qrp.research import DEFAULTS, _merge, forward_label, run_research


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(T=300, N=25, seed=11, missing=0.02)


def _spec(op, name=None, **params):
    inputs = CAUSAL_OPS.get(op, LEAKY_OPS.get(op))[1] or ["close"]
    return FeatureSpec(name or op, op, tuple(inputs), params)


@pytest.mark.parametrize("op", sorted(CAUSAL_OPS))
def test_every_causal_op_passes_audit(panel, op):
    params = {"window": 10} if op in ("returns", "momentum", "volatility", "ma_ratio",
                                     "volume_z", "zscore_ts") else {}
    if op == "lag":
        params = {"n": 2}
    rep = audit_causality(panel, [_spec(op, **params)])
    assert rep[0].causal, rep[0]


@pytest.mark.parametrize("op", sorted(LEAKY_OPS))
def test_every_leaky_op_fails_audit(panel, op):
    params = {} if op == "leaky_zscore_full" else {"window": 5}
    rep = audit_causality(panel, [_spec(op, **params)])
    assert not rep[0].causal
    assert rep[0].first_bad_row <= rep[0].cut  # a row at or before the cut moved


def test_one_bar_peek_is_caught(panel):
    """The smallest possible leak: a lead of one bar hidden inside a 'lag' op."""
    rep = audit_causality(panel, [FeatureSpec("peek", "lag", ("close",), {"n": -1})])
    assert not rep[0].causal
    assert rep[0].first_bad_row == rep[0].cut  # exactly the row before the future starts


def test_leak_propagates_through_dag(panel):
    """A causal op fed by a leaky one is leaky; the audit checks every node."""
    specs = [FeatureSpec.from_dict({"name": "fwd", "op": "leaky_forward_return", "window": 3}),
             FeatureSpec.from_dict({"name": "fwd_rank", "op": "xs_rank", "inputs": ["fwd"]}),
             FeatureSpec.from_dict({"name": "mom", "op": "momentum", "window": 20})]
    rep = {r.feature: r.causal for r in audit_causality(panel, specs)}
    assert rep == {"fwd": False, "fwd_rank": False, "mom": True}


def test_full_sample_normalization_caught_by_truncation(panel):
    specs = [FeatureSpec.from_dict({"name": "mom", "op": "momentum", "window": 20}),
             FeatureSpec.from_dict({"name": "z", "op": "leaky_zscore_full", "inputs": ["mom"]})]
    rep = {r.feature: r for r in audit_causality(panel, specs)}
    assert not rep["z"].causal


def test_assert_causal_raises(panel):
    with pytest.raises(LookAheadError, match="leaky_centered_mean"):
        assert_causal(panel, [_spec("leaky_centered_mean", window=11)])
    assert_causal(panel, [_spec("momentum", window=5)])  # does not raise


def test_label_is_forward_looking_and_would_be_flagged(panel):
    """The training label must look forward; used as a feature it would leak."""
    y = forward_label(panel["close"], delay=1, horizon=5)
    c = panel["close"]
    np.testing.assert_allclose(y[10], c[16] / c[11] - 1)
    assert np.isnan(y[-6:]).all()


def test_pipeline_refuses_leaky_config(panel, tmp_path):
    cfg = _merge(DEFAULTS, {
        "run": {"name": "leaky", "runs_root": str(tmp_path)},
        "features": [{"name": "mom", "op": "momentum", "window": 10},
                     {"name": "cheat", "op": "leaky_forward_return", "window": 5}],
        "cv": {"min_train_days": 100, "n_splits": 3},
        "universe": {"top_k": 20, "min_history": 20},
    })
    with pytest.raises(LookAheadError, match="cheat"):
        run_research(cfg, panel=panel)
    # the failed run is still recorded, with status "failed"
    import json
    metas = [json.loads(p.read_text()) for p in tmp_path.glob("*/meta.json")]
    assert metas and metas[0]["status"] == "failed"


def test_compute_features_does_not_mutate_panel(panel):
    before = {k: v.copy() for k, v in panel.fields.items()}
    compute_features(panel, [_spec("momentum", window=5), _spec("volume_z", window=5)])
    for k in before:
        np.testing.assert_array_equal(before[k], panel.fields[k])
