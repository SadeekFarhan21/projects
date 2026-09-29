"""VecEnv and tournament harness through the binding."""

from __future__ import annotations

import numpy as np

import figgie_arena as fa
from figgie_arena.stats import bootstrap_ci


def test_vecenv_shapes_rewards_and_invariants() -> None:
    cfg = fa.Config(ticks=25)
    env = fa.VecEnv(16, 1, ["passive", "taker", "bayes"], cfg, 3, True)
    obs = env.reset()
    assert obs.shape == (16, 1, env.obs_dim) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    total, dones = 0.0, 0
    for _ in range(75):
        a = np.stack([rng.integers(0, 8, (16, 1)), rng.integers(0, 4, (16, 1)),
                      rng.integers(1, 40, (16, 1))], -1).astype(np.int32)
        obs, rew, done = env.step(a)
        assert np.isfinite(obs).all()
        assert (rew[~done] == 0).all()
        total += float(rew.sum())
        dones += int(done.sum())
    assert dones == 16 * 3
    assert env.violations == 0
    assert env.steps == 16 * 75


def test_vecenv_is_deterministic() -> None:
    def roll():
        env = fa.VecEnv(4, 2, ["random", "bayes"], fa.Config(ticks=30), 9, False)
        out = [env.reset()]
        a = np.zeros((4, 2, 3), np.int32)
        a[..., 0], a[..., 2] = 1, 12  # everybody bids 12 in spades
        for _ in range(40):
            o, r, d = env.step(a)
            out += [o, r]
        return out
    for x, y in zip(roll(), roll()):
        np.testing.assert_array_equal(x, y)


def test_tournament_rotation_zero_sum_and_determinism() -> None:
    cfg = fa.Config(ticks=60)
    lineup = ["bayes", "passive", "taker", "random"]
    a = fa.run_games(lineup, 30, 5, cfg, 3, True, True, 1)
    b = fa.run_games(lineup, 30, 5, cfg, 3, True, True, 2)
    np.testing.assert_array_equal(a["pnl"], b["pnl"])
    assert a["pnl"].shape == (120, 4) and a["brier"].shape == (120, 4, 3)
    np.testing.assert_allclose(a["pnl"].sum(axis=1), 0.0, atol=1e-9)
    assert a["invariant_violations"] == 0
    assert (a["rotation"] == np.tile(np.arange(4), 30)).all()
    # The random bot always reports the uniform belief: Brier exactly 0.75.
    np.testing.assert_allclose(a["brier"][:, 3, :], 0.75)


def test_bootstrap_ci_covers_mean_and_clusters() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 1.0, 4000)
    m, lo, hi = bootstrap_ci(x, seed=0)
    assert lo < m < hi and lo < 2.0 < hi
    # Perfectly correlated rotations: clustering must widen the interval.
    base = rng.normal(0, 1, 250)
    y = np.repeat(base, 4)
    cl = np.repeat(np.arange(250), 4)
    _, l1, h1 = bootstrap_ci(y, seed=0)
    _, l2, h2 = bootstrap_ci(y, cl, seed=0)
    assert (h2 - l2) > 1.5 * (h1 - l1)
