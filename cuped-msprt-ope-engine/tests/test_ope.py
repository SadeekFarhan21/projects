from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from expope.ope import (
    bootstrap_ci,
    bootstrap_ci_obp_compatible,
    direct_method,
    doubly_robust,
    estimate_all,
    estimator_terms,
    fit_predict_q_hat,
    ips,
    per_action_mean_q_hat,
    row_terms,
    snips,
    switch_dr,
)
from expope.ope.obd import bts_action_dist, load_bts_prior
from expope.sim import SyntheticBandit

ROOT = Path(__file__).resolve().parents[1]


def _random_inputs(seed, n=200, a=5, l=2):
    rng = np.random.default_rng(seed)
    action = rng.integers(0, a, n)
    position = rng.integers(0, l, n)
    pscore = rng.uniform(0.05, 1.0, n)
    ad = rng.dirichlet(np.ones(a), size=(n, l)).transpose(0, 2, 1)  # (n, A, L), sums to 1 over A
    q = rng.uniform(0, 1, (n, a, l))
    reward = rng.binomial(1, 0.4, n).astype(float)
    return reward, action, pscore, ad, position, q


def test_hand_computed_example():
    reward = np.array([1.0, 0.0, 1.0])
    action = np.array([0, 1, 1])
    pscore = np.array([0.5, 0.5, 0.25])
    ad = np.array([[[1.0], [0.0]], [[0.5], [0.5]], [[0.0], [1.0]]])
    q = np.array([[[0.8], [0.2]], [[0.6], [0.4]], [[0.1], [0.9]]])
    # weights: 2, 1, 4. IPS = (2 + 0 + 4) / 3 = 2
    assert ips(reward, action, pscore, ad) == pytest.approx(2.0)
    assert snips(reward, action, pscore, ad) == pytest.approx(6.0 / 7.0)
    dm_rows = np.array([0.8, 0.5, 0.9])
    assert direct_method(ad, q) == pytest.approx(dm_rows.mean())
    dr_rows = dm_rows + np.array([2 * (1 - 0.8), 1 * (0 - 0.4), 4 * (1 - 0.9)])
    assert doubly_robust(reward, action, pscore, ad, q) == pytest.approx(dr_rows.mean())
    sw_rows = dm_rows + np.array([0.0, 1 * (0 - 0.4), 0.0])  # tau = 1 keeps only w <= 1
    assert switch_dr(reward, action, pscore, ad, q, tau=1.0) == pytest.approx(sw_rows.mean())


@settings(max_examples=50)
@given(st.integers(0, 100_000))
def test_switch_dr_limits(seed):
    r, a, p, ad, pos, q = _random_inputs(seed)
    dr = doubly_robust(r, a, p, ad, q, pos)
    assert switch_dr(r, a, p, ad, q, pos, tau=np.inf) == pytest.approx(dr, rel=1e-12)
    assert switch_dr(r, a, p, ad, q, pos, tau=0.0) == pytest.approx(direct_method(ad, q, pos), rel=1e-12)


@settings(max_examples=50)
@given(st.integers(0, 100_000))
def test_on_policy_identities(seed):
    """If pi_e = pi_b on the logged action then w = 1: IPS and SNIPS equal the mean reward, DR = mean reward."""
    r, a, _, ad, pos, q = _random_inputs(seed)
    p = ad[np.arange(len(a)), a, pos]
    assert ips(r, a, p, ad, pos) == pytest.approx(r.mean())
    assert snips(r, a, p, ad, pos) == pytest.approx(r.mean())
    t = row_terms(r, a, p, ad, pos, q)
    assert np.allclose(t.w, 1.0)


def test_validation_errors():
    r, a, p, ad, pos, q = _random_inputs(0)
    with pytest.raises(ValueError):
        ips(r, a, np.zeros_like(p), ad, pos)
    with pytest.raises(ValueError):
        ips(r, a, p, ad[:, :, 0], pos)
    with pytest.raises(ValueError):
        estimator_terms("dm", row_terms(r, a, p, ad, pos))
    assert set(estimate_all(row_terms(r, a, p, ad, pos))) == {"ips", "snips"}


def test_estimators_unbiased_on_synthetic_bandit():
    env = SyntheticBandit(seed=1)
    truth = env.true_value(n=400_000)
    rng = np.random.default_rng(0)
    est = {k: [] for k in ("ips", "snips", "dm", "dr", "switch_dr")}
    for _ in range(60):
        d = env.sample_logs(3000, rng)
        t = row_terms(d["reward"], d["action"], d["pscore"], d["action_dist"], None, d["q_true"])
        for k, v in estimate_all(t, tau=50.0).items():
            est[k].append(v)
    for k in ("ips", "dr", "dm"):
        m, se = np.mean(est[k]), np.std(est[k]) / np.sqrt(len(est[k]))
        assert abs(m - truth) < 4 * se + 1e-3, (k, m, truth)


def test_dr_is_robust_to_bad_reward_model():
    env = SyntheticBandit(seed=2)
    truth = env.true_value(n=400_000)
    rng = np.random.default_rng(1)
    dm, dr = [], []
    for _ in range(40):
        d = env.sample_logs(4000, rng)
        q_bad = per_action_mean_q_hat(d["action"], d["reward"], 4000, env.n_actions)
        dm.append(direct_method(d["action_dist"], q_bad))
        dr.append(doubly_robust(d["reward"], d["action"], d["pscore"], d["action_dist"], q_bad))
    assert abs(np.mean(dr) - truth) < abs(np.mean(dm) - truth)
    assert abs(np.mean(dr) - truth) < 4 * np.std(dr) / np.sqrt(40) + 1e-3


def test_reward_model_shapes_and_range():
    env = SyntheticBandit(seed=3)
    d = env.sample_logs(1500, np.random.default_rng(0))
    q = fit_predict_q_hat(d["context"], d["action"], d["reward"], env.n_actions)
    assert q.shape == (1500, env.n_actions, 1)
    assert np.all((q >= 0) & (q <= 1))
    # Better than the constant predictor on the true expected reward.
    err_model = np.mean((q - d["q_true"]) ** 2)
    err_const = np.mean((d["reward"].mean() - d["q_true"]) ** 2)
    assert err_model < err_const


def test_bootstrap_ci_contains_estimate_and_ratio_is_recomputed():
    rng = np.random.default_rng(0)
    num, den = rng.exponential(size=2000), rng.uniform(0.5, 1.5, 2000)
    ci = bootstrap_ci(num, den, n_boot=400, seed=1)
    assert ci["lower"] < ci["estimate"] < ci["upper"]
    assert ci["estimate"] == pytest.approx(num.mean() / den.mean())
    ci2 = bootstrap_ci(num, None, n_boot=400, seed=1)
    assert ci2["se"] == pytest.approx(num.std() / np.sqrt(2000), rel=0.15)


def test_bts_action_dist_is_a_distribution():
    prior = ROOT / "data" / "obd" / "prior_bts.yaml"
    if not prior.exists():
        pytest.skip("run scripts/fetch_obd.py first")
    a, b = load_bts_prior(prior)
    ad = bts_action_dist(a, b, n_sim=20_000)
    assert ad.shape == (80, 3)
    np.testing.assert_allclose(ad.sum(axis=0), 1.0)
    assert np.all(ad.sum(axis=1) <= 1.0 + 1e-12)  # an item appears at most once per slate
