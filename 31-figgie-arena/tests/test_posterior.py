"""The exact posterior against brute-force references computed in Python."""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest

import figgie_arena as fa


def test_twelve_configs_goal_is_partner_of_twelve() -> None:
    cfgs = fa.all_configs()
    assert len(cfgs) == 12
    for sizes, goal in cfgs:
        assert sizes[goal ^ 1] == 12 and sizes[goal] in (8, 10)


@pytest.mark.parametrize("hand", [[10, 0, 0, 0], [4, 3, 2, 1], [2, 2, 3, 3], [0, 5, 5, 0]])
def test_hand_only_posterior_is_hypergeometric(hand) -> None:
    post = np.array(fa.config_posterior(hand, [], 10))
    ref = np.array([np.prod([comb(sz[s], hand[s]) for s in range(4)]) for sz, _ in fa.all_configs()], float)
    np.testing.assert_allclose(post, ref / ref.sum(), atol=1e-12)


def _brute_constraint(remaining, mins, hand_size):
    """Enumerate every hand composition sequence, weighted by multinomial counts."""
    k = len(mins)
    def rec(j, r):
        if j == k - 1:
            return float(all(r[s] >= mins[j][s] for s in range(4)))
        acc = 0.0
        for x in itertools.product(*(range(r[s] + 1) for s in range(4))):
            if sum(x) != hand_size or any(x[s] < mins[j][s] for s in range(4)):
                continue
            w = np.prod([comb(r[s], x[s]) for s in range(4)])
            acc += w * rec(j + 1, [r[s] - x[s] for s in range(4)])
        return acc
    den = 1.0
    left = sum(remaining)
    for _ in range(k - 1):
        den *= comb(left, hand_size)
        left -= hand_size
    return rec(0, list(remaining)) / den


@pytest.mark.parametrize("remaining,mins,hs", [
    ([7, 9, 6, 8], [[2, 1, 0, 0], [0, 3, 1, 0], [1, 0, 0, 2]], 10),
    ([12, 6, 5, 7], [[3, 0, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0]], 10),
    ([6, 8, 9, 9], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [1, 0, 0, 1]], 8),
])
def test_constraint_probability_matches_enumeration(remaining, mins, hs) -> None:
    got = fa.constraint_probability(remaining, mins, hs)
    assert got == pytest.approx(_brute_constraint(remaining, mins, hs), rel=1e-10)


def test_posterior_calibrated_on_random_deals() -> None:
    """Hand-only goal beliefs should be calibrated: mean predicted prob of the
    true goal matches the frequency implied by the model on many deals."""
    rng = np.random.default_rng(0)
    probs, hits = [], []
    for seed in rng.integers(0, 2**31, size=3000):
        g = fa.Game(fa.Config(), int(seed))
        gp = fa.goal_probs(fa.config_posterior(g.initial_hand(0), [], 10))
        probs.extend(gp)
        hits.extend([float(s == g.goal_suit) for s in range(4)])
    probs, hits = np.array(probs), np.array(hits)
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 1.0)]:
        m = (probs >= lo) & (probs < hi)
        assert abs(probs[m].mean() - hits[m].mean()) < 0.03


def test_tape_constraints_can_only_sharpen_feasibility() -> None:
    hand = [3, 3, 2, 2]
    # Another player has sold 6 spades net: rules out configs with few spades left.
    post = fa.config_posterior(hand, [[6, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 10)
    for (sizes, _), p in zip(fa.all_configs(), post):
        if sizes[0] - hand[0] < 6:
            assert p == 0.0
    assert sum(post) == pytest.approx(1.0)
