"""Cross-check every from-scratch estimator against Open Bandit Pipeline on identical inputs."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from expope.ope import bootstrap_ci_obp_compatible, estimate_all, row_terms

from test_ope import _random_inputs


obp = pytest.importorskip("obp")


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 100_000), st.sampled_from([1, 3]), st.floats(0.5, 20.0))
def test_matches_obp_to_1e9(seed, len_list, tau):
    from obp.ope import (
        DirectMethod,
        DoublyRobust,
        InverseProbabilityWeighting,
        SelfNormalizedInverseProbabilityWeighting,
        SwitchDoublyRobust,
    )

    r, a, p, ad, pos, q = _random_inputs(seed, n=300, a=7, l=len_list)
    kw = dict(reward=r, action=a, pscore=p, action_dist=ad, position=pos)
    ours = estimate_all(row_terms(r, a, p, ad, pos, q), tau=tau)
    ref = {
        "ips": InverseProbabilityWeighting().estimate_policy_value(**kw),
        "snips": SelfNormalizedInverseProbabilityWeighting().estimate_policy_value(**kw),
        "dm": DirectMethod().estimate_policy_value(action_dist=ad, estimated_rewards_by_reg_model=q, position=pos),
        "dr": DoublyRobust().estimate_policy_value(**kw, estimated_rewards_by_reg_model=q),
        "switch_dr": SwitchDoublyRobust(tau=tau).estimate_policy_value(**kw, estimated_rewards_by_reg_model=q),
    }
    for k in ref:
        assert abs(ours[k] - ref[k]) <= 1e-9, (k, ours[k], ref[k])


def test_bootstrap_matches_obp():
    from obp.utils import estimate_confidence_interval_by_bootstrap

    samples = np.random.default_rng(5).exponential(size=500)
    ours = bootstrap_ci_obp_compatible(samples, n_bootstrap_samples=300, random_state=11)
    ref = estimate_confidence_interval_by_bootstrap(samples, n_bootstrap_samples=300, random_state=11)
    vals = list(ref.values())
    assert abs(ours["mean"] - vals[0]) <= 1e-12
    assert abs(ours["lower"] - vals[1]) <= 1e-12
    assert abs(ours["upper"] - vals[2]) <= 1e-12
