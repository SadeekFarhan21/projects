"""Accounting identities and cross-engine agreement for the backtester."""

import numpy as np
import pytest

from qrp.backtest.engine import CostModel, prepare_market, run_backtest
from qrp.features.ops import rolling_mean, rolling_std
from qrp.panel import simple_returns, synthetic_panel

ENGINES = ["numpy", "numba"]


@pytest.fixture(scope="module")
def market():
    p = synthetic_panel(T=160, N=12, seed=3, missing=0.03)
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.2, (4, 160, 12))
    W[:, :, 0] = 0.3  # one persistent long leg
    return p, W


def test_engines_agree_with_reference(market):
    """The share/cash reference engine is the oracle; vectorized kernels must match it."""
    p, W = market
    close = p["close"]
    sigma = rolling_std(simple_returns(close), 10)
    adv = rolling_mean(p["quote_volume"], 10)
    for costs in (CostModel(0, 0), CostModel(10, 5), CostModel(10, 5, impact_coef=0.5, aum=5e7)):
        for delay in (0, 1, 2):
            ref = run_backtest(W, close, costs, delay, "reference", sigma, adv, keep_positions=True)
            for eng in ENGINES:
                got = run_backtest(W, close, costs, delay, eng, sigma, adv, keep_positions=True)
                for f in ("equity", "gross_pnl", "cost", "turnover", "gross_exposure",
                          "net_exposure", "positions"):
                    np.testing.assert_allclose(getattr(got, f), getattr(ref, f), rtol=1e-10,
                                               atol=1e-12, err_msg=f"{eng} {f} {costs} d={delay}")


@pytest.mark.parametrize("engine", ENGINES + ["reference"])
def test_equity_identity(market, engine):
    """A[t] = A[t-1] + gross_pnl[t] - cost[t], with A[-1] = 1."""
    p, W = market
    res = run_backtest(W, p["close"], CostModel(10, 5), 1, engine)
    prev = np.concatenate([np.ones((W.shape[0], 1)), res.equity[:, :-1]], axis=1)
    np.testing.assert_allclose(res.equity, prev + res.gross_pnl - res.cost, rtol=0, atol=1e-12)
    # and the returns series compounds back to the equity curve
    np.testing.assert_allclose(np.cumprod(1 + res.returns, axis=1), res.equity, rtol=1e-10)


@pytest.mark.parametrize("engine", ENGINES)
def test_gross_pnl_equals_sum_of_asset_pnl(market, engine):
    p, W = market
    res = run_backtest(W, p["close"], CostModel(10, 5), 1, engine, keep_positions=True)
    r, _ = prepare_market(p["close"])
    asset_pnl = res.positions[:, :-1, :] * r[None, 1:, :]
    np.testing.assert_allclose(res.gross_pnl[:, 1:], asset_pnl.sum(axis=2), atol=1e-12)
    assert np.all(res.gross_pnl[:, 0] == 0)


@pytest.mark.parametrize("engine", ENGINES + ["reference"])
def test_single_asset_buy_and_hold(engine):
    p = synthetic_panel(T=100, N=1, seed=5)
    close = p["close"]
    W = np.ones((100, 1))
    res = run_backtest(W, close, CostModel(0, 0), delay=1, engine=engine)
    # bought at close[1] (decided at 0, executed at 1), held to the end
    assert res.equity[-1] == pytest.approx(close[-1, 0] / close[1, 0], rel=1e-12)


@pytest.mark.parametrize("engine", ENGINES)
def test_equal_weight_daily_rebalance_is_cross_sectional_mean(engine):
    p = synthetic_panel(T=80, N=6, seed=2)  # no gaps
    close = p["close"][:, :3]  # first N//2 names are listed from day 0
    W = np.full((80, 3), 1 / 3)
    res = run_backtest(W, close, CostModel(0, 0), delay=0, engine=engine)
    r = simple_returns(close)
    np.testing.assert_allclose(res.returns[1:], r[1:].mean(axis=1), rtol=1e-10)


@pytest.mark.parametrize("engine", ENGINES + ["reference"])
def test_drifting_targets_have_zero_turnover(engine):
    """Targets equal to the buy-and-hold drift weights mean nothing needs trading."""
    p = synthetic_panel(T=60, N=4, seed=7)
    close = p["close"][:, :2]
    shares = np.array([2.0, 1.0])
    val = close * shares
    W = val / val.sum(axis=1, keepdims=True)
    res = run_backtest(W, close, CostModel(0, 0), delay=0, engine=engine)
    assert res.turnover[0] == pytest.approx(1.0)          # initial buy
    np.testing.assert_allclose(res.turnover[1:], 0, atol=1e-12)

    # With costs, the entry fee is paid from cash, so the book is levered by exactly the
    # fee. A 100% invested target then needs a small sell the next day, and each later
    # day's correction is the fee on the previous correction (geometric decay).
    res = run_backtest(W, close, CostModel(10, 5), delay=0, engine=engine)
    assert res.turnover[1] == pytest.approx(15e-4, rel=0.05)
    assert res.turnover[2] < 1e-5 and res.turnover[1:].sum() < 2 * 15e-4


@pytest.mark.parametrize("engine", ENGINES)
def test_linear_cost_accounting(market, engine):
    p, W = market
    zero = run_backtest(W, p["close"], CostModel(0, 0), 1, engine)
    assert np.all(zero.cost == 0)
    lo = run_backtest(W, p["close"], CostModel(5, 0), 1, engine)
    hi = run_backtest(W, p["close"], CostModel(20, 5), 1, engine)
    assert np.all(hi.cost.sum(axis=1) > lo.cost.sum(axis=1))
    # cost[t] = linear * turnover[t] * pre-trade equity (E- = A[t-1] + gross_pnl[t])
    prev = np.concatenate([np.ones((W.shape[0], 1)), hi.equity[:, :-1]], axis=1)
    np.testing.assert_allclose(hi.cost, 25e-4 * hi.turnover * (prev + hi.gross_pnl), atol=1e-14)


@pytest.mark.parametrize("engine", ENGINES)
def test_backtester_does_not_peek_at_future_weights(market, engine):
    """Changing targets from row c on must not change equity before row c + delay."""
    p, W = market
    for delay in (0, 1, 3):
        base = run_backtest(W, p["close"], CostModel(10, 5), delay, engine)
        W2 = W.copy()
        c = 90
        W2[:, c:] = -W2[:, c:] * 3
        alt = run_backtest(W2, p["close"], CostModel(10, 5), delay, engine)
        np.testing.assert_array_equal(base.equity[:, : c + delay], alt.equity[:, : c + delay])
        assert not np.allclose(base.equity[:, c + delay], alt.equity[:, c + delay])


def test_delay_equals_shifted_weights(market):
    p, W = market
    d2 = run_backtest(W, p["close"], CostModel(10, 5), delay=2)
    shifted = np.zeros_like(W)
    shifted[:, 2:] = W[:, :-2]
    d0 = run_backtest(shifted, p["close"], CostModel(10, 5), delay=0)
    np.testing.assert_array_equal(d2.equity, d0.equity)


def test_negative_delay_rejected(market):
    p, W = market
    with pytest.raises(ValueError):
        run_backtest(W, p["close"], delay=-1)


@pytest.mark.parametrize("engine", ENGINES + ["reference"])
def test_untradeable_asset_holds_position(engine):
    """With no bar on a day, the target for that asset is ignored and the position drifts."""
    close = np.array([[10.0, 20.0], [11.0, 20.0], [np.nan, 21.0], [12.0, 22.0]])
    W = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0]])
    res = run_backtest(W, close, CostModel(0, 0), delay=0, engine=engine, keep_positions=True)
    # day 1: equity is 1.05 and asset 0 is rebalanced to 0.5 * 1.05 = 0.525.
    # day 2: asset 0 has no bar, so the 0.0 target is ignored and 0.525 is held.
    assert res.positions[1, 0] == pytest.approx(0.525)
    assert res.positions[2, 0] == pytest.approx(0.525)
    # asset 1 takes a 100% target of pre-trade equity 1.05 + 0.525 * (21/20 - 1)
    assert res.positions[2, 1] == pytest.approx(1.07625)
    # day 3: it trades again and is sold to 0
    assert res.positions[3, 0] == 0.0


@pytest.mark.parametrize("engine", ENGINES + ["reference"])
def test_bankruptcy_stops_trading(engine):
    close = np.array([[10.0], [30.0], [60.0], [70.0]])
    W = np.array([[-1.0], [-1.0], [-1.0], [-1.0]])  # short into a tripling
    res = run_backtest(W, close, CostModel(0, 0), delay=0, engine=engine)
    assert res.equity[1] == 0.0 and np.all(res.equity[1:] == 0.0)


def test_cached_market_is_identical(market):
    p, W = market
    a = run_backtest(W, p["close"], CostModel(10, 5))
    b = run_backtest(W, p["close"], CostModel(10, 5), market=prepare_market(p["close"]))
    np.testing.assert_array_equal(a.equity, b.equity)
