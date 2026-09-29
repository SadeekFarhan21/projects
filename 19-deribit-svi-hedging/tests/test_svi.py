import numpy as np

from optvol import svi


def test_svi_recovers_known_parameters_from_noisy_smile():
    rng = np.random.default_rng(11)
    true = svi.SVIParams(a=0.02, b=0.12, rho=-0.35, m=0.03, s=0.15)
    T = 0.25
    k = np.linspace(-0.6, 0.5, 41)
    w = svi.svi_w(k, true)
    vol = np.sqrt(w / T)
    vol_noisy = vol + rng.normal(0, 0.002, k.size)  # 0.2 vol point noise
    fit = svi.fit_svi(k, vol_noisy**2 * T, T=T)
    p = fit.params
    assert abs(p.rho - true.rho) < 0.1
    assert abs(p.m - true.m) < 0.05
    assert abs(p.s - true.s) < 0.05
    assert abs(p.b - true.b) < 0.02
    rmse = np.sqrt(np.mean((svi.svi_vol(k, p, T) - vol) ** 2))
    assert rmse < 0.002  # closer to the truth than the noise level


def test_svi_exact_fit_without_noise():
    true = svi.SVIParams(a=0.001, b=0.05, rho=0.2, m=-0.01, s=0.05)
    k = np.linspace(-0.3, 0.3, 31)
    fit = svi.fit_svi(k, svi.svi_w(k, true), T=0.05)
    assert np.max(np.abs(svi.svi_w(k, fit.params) - svi.svi_w(k, true))) < 1e-8


def test_butterfly_check_flags_vogt_smile():
    # Axel Vogt's example (Gatheral and Jacquier 2014): raw SVI with g < 0 near k = 0.7.
    vogt = svi.SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, s=0.4153)
    grid = np.linspace(-1.5, 1.5, 3001)
    ok, gmin = svi.butterfly_ok(vogt, grid)
    assert not ok and gmin < 0


def test_butterfly_check_passes_sane_smile():
    p = svi.SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, s=0.2)
    ok, gmin = svi.butterfly_ok(p, np.linspace(-2, 2, 4001))
    assert ok and gmin > 0


def test_calendar_check_flags_crossing_slices():
    k = np.linspace(-1, 1, 401)
    short = svi.SVIParams(a=0.01, b=0.20, rho=-0.6, m=0.0, s=0.1)  # steep skew
    long = svi.SVIParams(a=0.03, b=0.05, rho=0.0, m=0.0, s=0.3)  # flatter, higher ATM
    ok, dmin = svi.calendar_ok(svi.svi_w(k, short), svi.svi_w(k, long))
    assert not ok and dmin < 0  # the short slice's left wing pokes above
    ok2, _ = svi.calendar_ok(svi.svi_w(k, long), svi.svi_w(k, long) + 0.01)
    assert ok2


def test_ssvi_slice_equals_raw_svi_mapping():
    p = svi.SSVIParams(rho=-0.4, eta=0.9, gamma=0.4, theta=np.array([0.01, 0.04]))
    k = np.linspace(-1, 1, 101)
    for j, th in enumerate(p.theta):
        w_ssvi = svi.ssvi_w(k, th, p.rho, p.phi(th))
        w_raw = svi.svi_w(k, svi.ssvi_slice_params(p, j))
        assert np.max(np.abs(w_ssvi - w_raw)) < 1e-14


def test_ssvi_fit_is_arbitrage_free_and_recovers_surface():
    rng = np.random.default_rng(5)
    true = svi.SSVIParams(rho=-0.3, eta=1.0, gamma=0.4, theta=np.array([0.005, 0.02, 0.05, 0.1]))
    slices = []
    for th in true.theta:
        k = np.linspace(-0.5, 0.5, 21)
        w = svi.ssvi_w(k, th, true.rho, true.phi(th)) * (1 + rng.normal(0, 0.005, k.size))
        slices.append((k, w, np.ones_like(k)))
    p, _ = svi.fit_ssvi(slices)
    assert abs(p.rho - true.rho) < 0.05
    assert np.allclose(p.theta, true.theta, rtol=0.02)
    assert p.eta * (1 + abs(p.rho)) <= 2 + 1e-12 and 0 < p.gamma <= 0.5
    grid = np.linspace(-1.5, 1.5, 2001)
    prev = None
    for j in range(len(p.theta)):
        sp = svi.ssvi_slice_params(p, j)
        assert svi.butterfly_ok(sp, grid)[0]
        w = svi.svi_w(grid, sp)
        if prev is not None:
            assert svi.calendar_ok(prev, w)[0]
        prev = w
