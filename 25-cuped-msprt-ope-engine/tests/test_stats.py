import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import integrate
from scipy import stats as sps

from expope.stats import (
    CovariateStats,
    MeanStats,
    MSPRTMonitor,
    RatioStats,
    analytic_power_welch,
    benjamini_hochberg,
    cuped_from_stats,
    cuped_theta,
    cuped_ttest,
    delta_method_ratio_arrays,
    holm,
    msprt_ci_halfwidth,
    msprt_log_lr,
    ratio_mean_var,
    srm_check,
    welch_ttest,
)

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
sample = arrays(np.float64, st.integers(3, 60), elements=finite)


def _nondegenerate(x):
    return np.ptp(x) > 1e-3


# ------------------------------------------------------------------ sufficient stats


@given(sample)
def test_mean_stats_variance_matches_numpy(x):
    s = MeanStats.from_array(x)
    assert s.mean == pytest.approx(x.mean(), abs=1e-9)
    assert s.var == pytest.approx(x.var(ddof=1), rel=1e-6, abs=1e-6)


# ------------------------------------------------------------------------- Welch


@settings(max_examples=200)
@given(sample, sample)
def test_welch_matches_scipy(a, b):
    if not (_nondegenerate(a) and _nondegenerate(b)):
        return
    r = welch_ttest(a, b)
    ref = sps.ttest_ind(b, a, equal_var=False)
    assert r.stat == pytest.approx(ref.statistic, rel=1e-6, abs=1e-8)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-6, abs=1e-9)
    ci = ref.confidence_interval(0.95)
    assert r.ci_low == pytest.approx(ci.low, rel=1e-6, abs=1e-6)
    assert r.ci_high == pytest.approx(ci.high, rel=1e-6, abs=1e-6)


def test_welch_zero_variance_is_handled():
    r = welch_ttest(np.ones(10), np.ones(10))
    assert r.p_value == 1.0 and r.estimate == 0.0


# ------------------------------------------------------------------ delta method


@settings(max_examples=100)
@given(st.integers(0, 10_000), st.integers(5, 200))
def test_delta_variance_equals_linearisation(seed, n):
    """Delta-method variance equals Var(x - R y) / (n ybar^2), the linearised form."""
    rng = np.random.default_rng(seed)
    y = rng.poisson(3, n) + 1.0
    x = rng.binomial(y.astype(int), 0.3).astype(float)
    r, v = ratio_mean_var(RatioStats.from_arrays(x, y))
    R = x.sum() / y.sum()
    lin = np.var(x - R * y, ddof=1) / (n * y.mean() ** 2)
    assert r == pytest.approx(R)
    assert v == pytest.approx(lin, rel=1e-9, abs=1e-15)


def test_delta_method_matches_simulated_sampling_variance():
    rng = np.random.default_rng(1)
    n, reps = 400, 3000
    ests = []
    for _ in range(reps):
        y = rng.poisson(4, n) + 1.0
        x = rng.binomial(y.astype(int), 0.2).astype(float)
        ests.append(x.sum() / y.sum())
    y = rng.poisson(4, n) + 1.0
    x = rng.binomial(y.astype(int), 0.2).astype(float)
    _, v = ratio_mean_var(RatioStats.from_arrays(x, y))
    assert v == pytest.approx(np.var(ests), rel=0.1)


def test_delta_method_ratio_test_null():
    rng = np.random.default_rng(2)
    y1, y2 = rng.poisson(3, 5000) + 1.0, rng.poisson(3, 5000) + 1.0
    x1, x2 = rng.binomial(y1.astype(int), 0.3), rng.binomial(y2.astype(int), 0.3)
    r = delta_method_ratio_arrays(x1, y1, x2, y2)
    assert r.p_value > 0.001
    assert r.ci_low < 0 < r.ci_high


# ------------------------------------------------------------------------- CUPED


@settings(max_examples=100)
@given(st.integers(0, 10_000), st.integers(10, 300))
def test_cuped_from_stats_equals_direct_adjustment(seed, n):
    rng = np.random.default_rng(seed)
    xc, xt = rng.normal(5, 2, n), rng.normal(5, 2, n + 7)
    yc, yt = xc + rng.normal(0, 1, n), xt + rng.normal(0.3, 1, n + 7)
    r, theta = cuped_ttest(yc, xc, yt, xt)
    x_all, y_all = np.r_[xc, xt], np.r_[yc, yt]
    th = np.cov(x_all, y_all, ddof=1)[0, 1] / np.var(x_all, ddof=1)
    assert theta == pytest.approx(th, rel=1e-9)
    ref = welch_ttest(yc - th * (xc - x_all.mean()), yt - th * (xt - x_all.mean()))
    assert r.estimate == pytest.approx(ref.estimate, rel=1e-8, abs=1e-10)
    assert r.se == pytest.approx(ref.se, rel=1e-7)
    assert r.p_value == pytest.approx(ref.p_value, rel=1e-6, abs=1e-12)


def test_cuped_theta_zero_is_welch():
    rng = np.random.default_rng(0)
    yc, yt, xc, xt = rng.normal(size=(4, 100))
    r, _ = cuped_from_stats(CovariateStats.from_arrays(yc, xc), CovariateStats.from_arrays(yt, xt), theta=0.0)
    ref = welch_ttest(yc, yt)
    assert r.estimate == pytest.approx(ref.estimate) and r.se == pytest.approx(ref.se)


def test_cuped_reduces_variance_by_rho_squared():
    rng = np.random.default_rng(3)
    n, rho = 20000, 0.7
    x = rng.normal(size=2 * n)
    y = rho * x + math.sqrt(1 - rho**2) * rng.normal(size=2 * n)
    r, _ = cuped_ttest(y[:n], x[:n], y[n:], x[n:])
    base = welch_ttest(y[:n], y[n:])
    assert (r.se / base.se) ** 2 == pytest.approx(1 - rho**2, rel=0.05)


# --------------------------------------------------------------------------- SRM


@given(arrays(np.int64, st.integers(2, 6), elements=st.integers(1, 100_000)))
def test_srm_matches_scipy_chisquare(counts):
    r = srm_check(counts)
    ref = sps.chisquare(counts)
    assert r.chi2 == pytest.approx(ref.statistic, rel=1e-9)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-7, abs=1e-300)


def test_srm_detects_mismatch_with_unequal_design():
    assert srm_check([5000, 5000], [0.5, 0.5]).p_value > 0.5
    assert srm_check([5000, 5400], [0.5, 0.5]).is_mismatch()
    assert not srm_check([2500, 7500], [0.25, 0.75]).is_mismatch()


# ------------------------------------------------------------- multiple testing

pvals = arrays(np.float64, st.integers(1, 40), elements=st.floats(0, 1))


@given(pvals)
def test_bh_matches_scipy(p):
    np.testing.assert_allclose(benjamini_hochberg(p), sps.false_discovery_control(p, method="bh"), rtol=1e-12, atol=1e-15)


@given(pvals)
def test_holm_properties(p):
    adj = holm(p)
    assert np.all(adj >= p - 1e-15) and np.all(adj <= 1)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adj[order]) >= -1e-15)  # monotone in raw p
    assert np.all(adj <= np.minimum(p * len(p), 1) + 1e-12)  # never worse than Bonferroni
    assert np.all(benjamini_hochberg(p) <= adj + 1e-12)  # BH is never more conservative than Holm


def test_holm_known_example():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


# ------------------------------------------------------------------------ mSPRT


@pytest.mark.parametrize("z,s2,tau2", [(0.1, 0.01, 0.04), (-0.5, 0.2, 1.0), (0.0, 1.0, 0.1), (2.0, 0.5, 0.3)])
def test_msprt_lr_matches_numerical_mixture(z, s2, tau2):
    """Closed form equals integral of N(z; theta, s2) / N(z; 0, s2) against N(0, tau2)."""
    f = lambda th: sps.norm.pdf(z, th, math.sqrt(s2)) * sps.norm.pdf(th, 0, math.sqrt(tau2))  # noqa: E731
    num, _ = integrate.quad(f, -50, 50, points=[0, z], limit=500)
    lr = num / sps.norm.pdf(z, 0, math.sqrt(s2))
    assert math.exp(float(msprt_log_lr(np.array(z), np.array(s2), tau2))) == pytest.approx(lr, rel=1e-6)


def test_msprt_ci_boundary_hits_threshold():
    s2, tau2, alpha = 0.03, 0.2, 0.05
    hw = float(msprt_ci_halfwidth(np.array(s2), tau2, alpha))
    llr = float(msprt_log_lr(np.array(hw), np.array(s2), tau2))
    assert llr == pytest.approx(math.log(1 / alpha), rel=1e-9)


def test_msprt_monitor_detects_large_effect_and_ci_nests():
    rng = np.random.default_rng(0)
    mon = MSPRTMonitor(tau2=0.1)
    c, t = [], []
    widths = []
    for _ in range(50):
        c.extend(rng.normal(0, 1, 100))
        t.extend(rng.normal(0.3, 1, 100))
        mon.update(MeanStats.from_array(np.array(c)), MeanStats.from_array(np.array(t)))
        widths.append(mon.ci_high - mon.ci_low)
    assert mon.stopped_at is not None and mon.p_value < 0.05
    assert all(b <= a + 1e-12 for a, b in zip(widths, widths[1:]))
    assert mon.ci_low < 0.3 < mon.ci_high


# ------------------------------------------------------------------------ power


def test_analytic_power_at_zero_effect_is_alpha():
    assert analytic_power_welch(0.0, 1.0, 500) == pytest.approx(0.05, abs=1e-9)


def test_analytic_power_matches_normal_approx():
    n, d = 1000, 0.1
    z = d / math.sqrt(2 / n)
    approx = sps.norm.sf(1.959964 - z) + sps.norm.cdf(-1.959964 - z)
    assert analytic_power_welch(d, 1.0, n) == pytest.approx(approx, abs=2e-3)
