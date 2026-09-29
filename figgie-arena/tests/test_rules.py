"""Property tests of the rules engine through the Python binding."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import figgie_arena as fa
from figgie_arena import ActType, Status


@given(seed=st.integers(0, 2**63 - 1), n=st.sampled_from([4, 5]))
@settings(max_examples=300, deadline=None)
def test_deal_invariants(seed: int, n: int) -> None:
    g = fa.Game(fa.Config(n_players=n), seed)
    sizes = g.suit_counts
    assert sorted(sizes) == [8, 10, 10, 12]
    twelve = sizes.index(12)
    assert g.goal_suit == twelve ^ 1  # same colour partner
    assert sizes[g.goal_suit] in (8, 10)
    hands = [g.hand(p) for p in range(n)]
    assert all(sum(h) == 40 // n for h in hands)
    assert [sum(col) for col in zip(*hands)] == sizes
    assert all(g.cash(p) == 350 - 200 // n for p in range(n))
    assert g.pot == 200
    assert g.check_invariants() == ""


def _random_hands(draw, sizes, n):
    deck = [s for s in range(4) for _ in range(sizes[s])]
    perm = draw(st.permutations(deck))
    hs = 40 // n
    return [[perm[p * hs:(p + 1) * hs].count(s) for s in range(4)] for p in range(n)]


@st.composite
def settlements(draw):
    n = draw(st.sampled_from([4, 5]))
    sizes = list(draw(st.permutations([12, 10, 10, 8])))
    hands = _random_hands(draw, sizes, n)
    goal = sizes.index(12) ^ 1
    return n, goal, hands


@given(settlements())
@settings(max_examples=500, deadline=None)
def test_payout_rules(case) -> None:
    n, goal, hands = case
    pay = fa.Game.settle(fa.Config(n_players=n), goal, hands)  # 1/60 chip units
    assert sum(pay) == 60 * 200  # the whole pot is paid out, exactly
    g = [h[goal] for h in hands]
    remainder = 200 - 10 * sum(g)
    winners = [p for p in range(n) if g[p] == max(g)]
    for p in range(n):
        want = 600 * g[p] + (60 * remainder // len(winners) if p in winners else 0)
        assert pay[p] == want
    assert remainder in (100, 120)


@given(seed=st.integers(0, 2**32), steps=st.integers(1, 400), clear=st.booleans(), n=st.sampled_from([4, 5]))
@settings(max_examples=150, deadline=None)
def test_random_actions_conserve_cards_and_money(seed: int, steps: int, clear: bool, n: int) -> None:
    cfg = fa.Config(n_players=n, ticks=50, clear_on_trade=clear)
    g = fa.Game(cfg, seed)
    rng = np.random.default_rng(seed)
    types = list(ActType.__members__.values())
    for _ in range(steps):
        if g.done:
            break
        p = int(rng.integers(0, n))
        t = types[int(rng.integers(0, len(types)))]
        g.apply(p, t, int(rng.integers(-1, 5)), int(rng.integers(-2, 60)))
        assert g.check_invariants() == ""
        if rng.random() < 0.05:
            g.end_tick()
    while not g.done:
        g.end_tick()
    assert g.check_invariants() == ""
    assert math.isclose(sum(g.pnl(p) for p in range(n)), 0.0, abs_tol=1e-9)
    total_cards = np.array([g.hand(p) for p in range(n)]).sum(axis=0).tolist()
    assert total_cards == g.suit_counts


def test_trade_clears_every_book() -> None:
    g = fa.Game(fa.Config(), 0)
    g.reset_with_deal([12, 8, 10, 10], [[3, 3, 2, 2], [3, 2, 3, 2], [3, 2, 2, 3], [3, 1, 3, 3]])
    assert g.apply(0, ActType.BID, 0, 5) == Status.OK
    assert g.apply(1, ActType.ASK, 3, 30) == Status.OK
    assert g.apply(2, ActType.ASK, 1, 7) == Status.OK
    assert g.apply(3, ActType.BUY, 1) == Status.TRADED
    assert all(g.best_bid(s) is None and g.best_ask(s) is None for s in range(4))
    t = g.trades[0]
    assert (t.buyer, t.seller, t.price, t.suit) == (3, 2, 7, 1)
