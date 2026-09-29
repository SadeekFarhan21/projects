import math

import numpy as np
import pytest

from marketsim import SimConfig, run
from marketsim.agents import MarketMaker
from marketsim.engine import Fundamental, Market, Scheduler
from marketsim.metrics import mm_pnl_decomposition, summarize

SHORT = SimConfig(t_end=2_000.0, informed_frac=0.2)


@pytest.fixture(scope="module")
def short_run():
    return run(SHORT)


def test_same_seed_is_bit_identical():
    a, b = run(SHORT), run(SHORT)
    for k in a.samples:
        np.testing.assert_array_equal(a.samples[k], b.samples[k])
    np.testing.assert_array_equal(a.trades["price"], b.trades["price"])


def test_different_seed_differs():
    a, b = run(SHORT), run(SHORT.with_(seed=1))
    assert not np.array_equal(a.samples["mid"], b.samples["mid"])


def test_book_invariants_hold_throughout():
    run(SHORT.with_(t_end=300.0), check_invariants_every=1)


def test_cash_and_inventory_are_conserved(short_run):
    # Every trade has a buyer and a seller, so totals across agents are zero.
    assert short_run.counters["sum_inventory"] == 0
    assert short_run.counters["sum_cash"] == 0.0


def test_pnl_decomposition_is_exact(short_run):
    d = mm_pnl_decomposition(short_run)
    mm_wealth = short_run.counters["mm_pnl_mid"]
    assert math.isclose(d["total"], mm_wealth, abs_tol=1e-6)
    assert math.isclose(d["spread_capture"] + d["adverse_selection"] + d["inventory_carry"], d["total"])
    assert d["spread_capture"] > 0  # the MM buys below and sells above the mid


def test_book_is_never_crossed_or_locked(short_run):
    spread = short_run.samples["spread"]
    assert np.all(spread[np.isfinite(spread)] >= 1)


def test_informed_traders_have_edge_and_noise_pays():
    # Edge at trade time against the true value V. (End-of-run PnL marked at
    # V is too noisy: informed traders never unwind their inventory.)
    r = run(SimConfig(t_end=5_000.0, informed_frac=0.3))
    c = r.counters
    assert c["informed_edge_vs_V"] > 0
    assert c["noise_edge_vs_V"] < 0
    # Edges are zero-sum across all agents (each trade has two sides).
    total = sum(c[f"{n}_edge_vs_V"] for n in ("informed", "noise", "noise_lp", "mm"))
    assert abs(total) < 1e-6


def test_prices_track_fundamental_with_informed_flow():
    s = summarize(run(SimConfig(t_end=5_000.0, informed_frac=0.3)))
    s0 = summarize(run(SimConfig(t_end=5_000.0, informed_frac=0.0)))
    assert s["mean_abs_mid_minus_V"] < s0["mean_abs_mid_minus_V"]


def test_as_quotes_symmetric_when_flat_and_skewed_with_inventory():
    cfg = SimConfig(informed_frac=0.0)
    sched = Scheduler()
    mkt = Market(sched, cfg.p0)
    fund = Fundamental(sched, np.random.default_rng(0), cfg.p0, 0, 0, 0, 1.0)
    mm = MarketMaker(mkt, sched, np.random.default_rng(1), cfg, fund)
    bid, ask = mm.quotes()
    assert cfg.p0 - bid == ask - cfg.p0
    # Closed form: gamma sigma^2 tau / 2 + ln(1 + gamma/k) / gamma
    h = 0.1 * 10 / 2 + math.log(1 + 0.1 / 0.5) / 0.1
    assert math.isclose(mm.half_spread(), h)
    mm.inventory = 5  # long: shade both quotes down to sell inventory
    bid2, ask2 = mm.quotes()
    assert bid2 < bid and ask2 < ask


def test_adaptive_mm_quotes_wider_with_more_informed_flow():
    lo = summarize(run(SimConfig(t_end=5_000.0, informed_frac=0.05, mm_adaptive=True)))
    hi = summarize(run(SimConfig(t_end=5_000.0, informed_frac=0.4, mm_adaptive=True)))
    # The MM's own quoted spread is what the adaptive rule widens. The book
    # spread is muddied by stale LP orders inside the quotes (DEVLOG Problem 6)
    # and is not monotone in alpha for the adaptive MM, so it is not asserted.
    assert hi["mean_mm_quote_spread"] > lo["mean_mm_quote_spread"] + 0.3
