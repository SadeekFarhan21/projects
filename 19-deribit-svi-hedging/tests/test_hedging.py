import numpy as np

from optvol import black76 as b
from optvol import hedging as h
from optvol import svi
from optvol.chain import US_PER_YEAR


def synthetic_panel(seed=0, n_paths_minutes=480, sigma=0.5, n_strikes=9):
    """One lognormal F path, options priced exactly at a flat vol."""
    rng = np.random.default_rng(seed)
    exp = 10 * 3600 * 10**6 + n_paths_minutes * h.MIN_US
    t = np.arange(n_paths_minutes + 1, dtype=np.int64) * h.MIN_US
    T = (exp - t) / US_PER_YEAR
    dT = 60 / (365 * 86400)
    z = rng.standard_normal(t.size - 1)
    F = 100 * np.exp(np.concatenate([[0], np.cumsum(sigma * np.sqrt(dT) * z - 0.5 * sigma**2 * dT)]))
    K = np.linspace(90, 110, n_strikes)
    cp = K >= 100
    mid = b.price(F[:, None], K, T[:, None], sigma, cp)
    ivm = np.full(mid.shape, sigma)
    p = h.Panel("X", exp, t, F, K, cp, [str(k) for k in K], mid, np.full(mid.shape, 0.01), ivm)
    surf = [svi.SVIParams(a=sigma**2 * Ti, b=0.0, rho=0.0, m=0.0, s=0.1) for Ti in T]
    return p, surf


def test_attribution_residual_small_and_hedging_reduces_variance():
    p, surf = synthetic_panel()
    r = h.hedge_ratios(p, surf)
    # flat smile: all three methods agree
    assert np.allclose(r["own_iv"], r["sticky_strike"], atol=1e-10)
    assert np.allclose(r["sticky_moneyness"], r["sticky_strike"], atol=1e-10)
    res1 = h.simulate(p, r, "own_iv", 1, 0, p.t_us.size - 1)
    res30 = h.simulate(p, r, "own_iv", 30, 0, p.t_us.size - 1)
    unh = h.simulate(p, r, None, 1, 0, p.t_us.size - 1)
    assert np.abs(res1.hedge_err).max() < 1e-9  # hedge equals own delta every minute
    assert np.all(np.abs(res1.vega) < 1e-12)  # vol never moves
    # attribution adds up exactly by construction; residual is the Taylor remainder
    tot = res1.theta + res1.gamma + res1.vega + res1.hedge_err + res1.residual
    assert np.allclose(tot, res1.total)
    assert np.abs(res1.residual).max() < 0.05 * np.abs(res1.gamma).max()
    # theta and gamma nearly cancel for a hedged option at the right vol
    assert np.all(np.abs(res1.total) < 0.3 * (np.abs(res1.theta) + 1e-9))
    assert np.std(res1.total) < np.std(res30.total) < np.std(unh.total)


def test_sticky_moneyness_adds_skew_term():
    p, _ = synthetic_panel()
    skew = [svi.SVIParams(a=0.5**2 * Ti * 0.9, b=0.02, rho=-0.7, m=0.0, s=0.05) for Ti in p.T]
    r = h.hedge_ratios(p, skew)
    i = 10
    k = np.log(p.K / p.F[i])
    sig, dsdk = h.surface_vol_and_slope(skew[i], k, p.T[i])
    g = b.greeks(p.F[i], p.K, p.T[i], sig, p.is_call)
    assert np.allclose(r["sticky_strike"][i], g["delta"])
    assert np.allclose(r["sticky_moneyness"][i], g["delta"] - g["vega"] * dsdk / p.F[i])
    # where the smile slopes down (dsigma/dk < 0) sticky moneyness delta is higher,
    # where it slopes up it is lower
    down, up = dsdk < 0, dsdk > 0
    assert down.any() and up.any()
    assert np.all(r["sticky_moneyness"][i][down] > r["sticky_strike"][i][down])
    assert np.all(r["sticky_moneyness"][i][up] < r["sticky_strike"][i][up])


def test_bootstrap_interval_contains_estimate():
    x = np.random.default_rng(1).normal(0, 2, 400)
    s, lo, hi = h.bootstrap_std(x, n_boot=500)
    assert lo < s < hi and abs(s - 2) < 0.3


def test_otm_iv_replaces_itm_side_only():
    K = np.array([90.0, 90.0, 110.0, 110.0])
    cp = np.array([True, False, True, False])
    F = np.array([100.0, 120.0])
    sig = np.array([[0.9, 0.5, 0.4, 0.8],
                    [0.9, 0.5, 0.4, 0.8]])
    out = h.otm_iv(sig, K, cp, F)
    # F=100: 90C ITM <- 90P, 110P ITM <- 110C
    assert np.allclose(out[0], [0.5, 0.5, 0.4, 0.4])
    # F=120: both calls ITM <- puts
    assert np.allclose(out[1], [0.5, 0.5, 0.8, 0.8])
