"""From-scratch A/B test statistics.

All effects are reported as treatment minus control. Only scipy.stats distribution
functions (t, norm, chi2 CDFs and quantiles) are used; every estimator, variance and
test statistic is computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps

from .sufficient import CovariateStats, MeanStats, RatioStats


@dataclass(frozen=True)
class TestResult:
    """Result of a two-sample test on the difference treatment minus control."""

    __test__ = False  # keep pytest from collecting this dataclass

    estimate: float
    se: float
    stat: float
    p_value: float
    ci_low: float
    ci_high: float
    df: float = math.inf  # inf means a z-test
    control_mean: float = math.nan
    treatment_mean: float = math.nan

    def covers(self, truth: float) -> bool:
        return self.ci_low <= truth <= self.ci_high


def _finish(est: float, se: float, df: float, alpha: float, cm: float, tm: float) -> TestResult:
    if se <= 0 or not np.isfinite(se):
        # Degenerate case (for example zero variance in both arms).
        p = 1.0 if est == 0 else 0.0
        return TestResult(est, se, math.copysign(math.inf, est) if est else 0.0, p, est, est, df, cm, tm)
    stat = est / se
    if math.isinf(df):
        p = 2.0 * sps.norm.sf(abs(stat))
        q = sps.norm.ppf(1 - alpha / 2)
    else:
        p = 2.0 * sps.t.sf(abs(stat), df)
        q = sps.t.ppf(1 - alpha / 2, df)
    return TestResult(est, se, stat, float(p), est - q * se, est + q * se, df, cm, tm)


# ----------------------------------------------------------------------------- Welch


def welch_from_stats(control: MeanStats, treatment: MeanStats, alpha: float = 0.05) -> TestResult:
    """Welch's unequal-variance t-test with Welch-Satterthwaite degrees of freedom."""
    vc = control.var / control.n
    vt = treatment.var / treatment.n
    se = math.sqrt(vc + vt)
    num = (vc + vt) ** 2
    den = vc * vc / (control.n - 1) + vt * vt / (treatment.n - 1)
    df = num / den if den > 0 else math.inf
    est = treatment.mean - control.mean
    return _finish(est, se, df, alpha, control.mean, treatment.mean)


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> TestResult:
    return welch_from_stats(MeanStats.from_array(control), MeanStats.from_array(treatment), alpha)


# ------------------------------------------------------------------- delta method


def ratio_mean_var(s: RatioStats) -> tuple[float, float]:
    """Point estimate and delta-method variance of R = sum(x) / sum(y) for one arm.

    With per-user pairs (x_i, y_i), R = xbar / ybar. First-order Taylor expansion gives
        Var(R) ~= (1/n) * [ Var(x)/mu_y^2 - 2 mu_x Cov(x,y)/mu_y^3 + mu_x^2 Var(y)/mu_y^4 ].
    Sample moments use ddof=1.
    """
    n = s.n
    mx = s.sum_x / n
    my = s.sum_y / n
    vx = (s.sum_x2 - n * mx * mx) / (n - 1)
    vy = (s.sum_y2 - n * my * my) / (n - 1)
    cxy = (s.sum_xy - n * mx * my) / (n - 1)
    r = mx / my
    var = (vx / my**2 - 2 * mx * cxy / my**3 + mx**2 * vy / my**4) / n
    return r, max(var, 0.0)


def delta_method_ratio(control: RatioStats, treatment: RatioStats, alpha: float = 0.05) -> TestResult:
    """z-test on the difference of two ratio metrics using the delta method."""
    rc, vc = ratio_mean_var(control)
    rt, vt = ratio_mean_var(treatment)
    return _finish(rt - rc, math.sqrt(vc + vt), math.inf, alpha, rc, rt)


def delta_method_ratio_arrays(
    num_c: np.ndarray, den_c: np.ndarray, num_t: np.ndarray, den_t: np.ndarray, alpha: float = 0.05
) -> TestResult:
    return delta_method_ratio(RatioStats.from_arrays(num_c, den_c), RatioStats.from_arrays(num_t, den_t), alpha)


# -------------------------------------------------------------------------- CUPED


def cuped_theta(control: CovariateStats, treatment: CovariateStats) -> float:
    """theta = Cov(Y, X) / Var(X), pooled across both arms (Deng et al. 2013)."""
    n = control.n + treatment.n
    sx = control.sum_x + treatment.sum_x
    sy = control.sum_y + treatment.sum_y
    sxx = control.sum_x2 + treatment.sum_x2
    sxy = control.sum_xy + treatment.sum_xy
    cov = (sxy - sx * sy / n) / (n - 1)
    varx = (sxx - sx * sx / n) / (n - 1)
    return cov / varx if varx > 0 else 0.0


def cuped_from_stats(
    control: CovariateStats, treatment: CovariateStats, alpha: float = 0.05, theta: float | None = None
) -> tuple[TestResult, float]:
    """CUPED-adjusted Welch test. Returns (result, theta).

    Y_adj = Y - theta * (X - mean(X_pooled)). Because theta and the pooled mean are shared by
    both arms, the adjusted difference is (ybar_t - ybar_c) - theta * (xbar_t - xbar_c), and the
    per-arm adjusted variance is Var(Y) - 2 theta Cov(X,Y) + theta^2 Var(X).
    """
    if theta is None:
        theta = cuped_theta(control, treatment)
    xbar = (control.sum_x + treatment.sum_x) / (control.n + treatment.n)

    def adjusted(s: CovariateStats) -> MeanStats:
        n = s.n
        # Sufficient statistics of y_adj = y - theta * (x - xbar).
        sum_adj = s.sum_y - theta * (s.sum_x - n * xbar)
        # sum (y - theta x + theta xbar)^2 expanded.
        c = theta * xbar
        sum_adj2 = (
            s.sum_y2
            + theta * theta * s.sum_x2
            + n * c * c
            - 2 * theta * s.sum_xy
            + 2 * c * s.sum_y
            - 2 * theta * c * s.sum_x
        )
        return MeanStats(n=n, sum_y=sum_adj, sum_y2=sum_adj2)

    return welch_from_stats(adjusted(control), adjusted(treatment), alpha), theta


def cuped_ttest(
    y_c: np.ndarray, x_c: np.ndarray, y_t: np.ndarray, x_t: np.ndarray, alpha: float = 0.05
) -> tuple[TestResult, float]:
    return cuped_from_stats(CovariateStats.from_arrays(y_c, x_c), CovariateStats.from_arrays(y_t, x_t), alpha)


# ---------------------------------------------------------------------------- SRM


@dataclass(frozen=True)
class SRMResult:
    chi2: float
    p_value: float
    observed: tuple[int, ...]
    expected: tuple[float, ...]

    def is_mismatch(self, threshold: float = 0.001) -> bool:
        return self.p_value < threshold


def srm_check(counts: Sequence[int], ratios: Sequence[float] | None = None) -> SRMResult:
    """Pearson chi-square goodness-of-fit test of assignment counts against the design ratios."""
    obs = np.asarray(counts, dtype=float)
    k = obs.size
    r = np.full(k, 1.0 / k) if ratios is None else np.asarray(ratios, dtype=float)
    r = r / r.sum()
    exp = obs.sum() * r
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(sps.chi2.sf(chi2, k - 1))
    return SRMResult(chi2, p, tuple(int(c) for c in obs), tuple(float(e) for e in exp))


# ------------------------------------------------------------ multiple testing


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (controls FWER)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (q-values, controls FDR)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


# ------------------------------------------------------------------------ mSPRT


def msprt_log_lr(est: np.ndarray, var: np.ndarray, tau2: float, theta0: float = 0.0) -> np.ndarray:
    """Log mixture likelihood ratio for a normal estimate with variance var, N(theta0, tau2) mixing.

    With an estimate Z ~ N(theta, s^2) and a N(theta0, tau^2) mixture over the alternative,
        Lambda = sqrt(s^2 / (s^2 + tau^2)) * exp( tau^2 (Z - theta0)^2 / (2 s^2 (s^2 + tau^2)) ).
    For a balanced two-sample test s^2 = 2 sigma^2 / n and this reduces to the Johari et al.
    (2017) form. Vectorised over any array shape.
    """
    est = np.asarray(est, dtype=float)
    s2 = np.asarray(var, dtype=float)
    d = est - theta0
    return 0.5 * np.log(s2 / (s2 + tau2)) + tau2 * d * d / (2.0 * s2 * (s2 + tau2))


def msprt_always_valid_p(est: np.ndarray, var: np.ndarray, tau2: float, theta0: float = 0.0, axis: int = -1) -> np.ndarray:
    """Always-valid p-values p_n = min_{k<=n} min(1, 1/Lambda_k) along ``axis``."""
    llr = msprt_log_lr(est, var, tau2, theta0)
    p = np.minimum(1.0, np.exp(-llr))
    return np.minimum.accumulate(p, axis=axis)


def msprt_ci_halfwidth(var: np.ndarray, tau2: float, alpha: float) -> np.ndarray:
    """Half width of the always-valid (1 - alpha) confidence sequence at each look.

    The set {theta : Lambda(theta) < 1/alpha} is Z +/- s * sqrt((s^2+tau^2)/tau^2 * (log((s^2+tau^2)/s^2) + 2 log(1/alpha))).
    """
    s2 = np.asarray(var, dtype=float)
    return np.sqrt(s2 * (s2 + tau2) / tau2 * (np.log((s2 + tau2) / s2) + 2.0 * np.log(1.0 / alpha)))


@dataclass
class MSPRTMonitor:
    """Streaming two-sample mixture SPRT for a difference in means.

    Call :meth:`update` with the running per-arm MeanStats at each look. Variances are plug-in
    estimates, which is the standard practical choice (Johari et al. 2017, section 4).
    """

    tau2: float
    alpha: float = 0.05
    theta0: float = 0.0
    p_value: float = 1.0
    ci_low: float = -math.inf
    ci_high: float = math.inf
    stopped_at: int | None = None
    looks: int = 0

    def update(self, control: MeanStats, treatment: MeanStats) -> bool:
        self.looks += 1
        est = treatment.mean - control.mean
        var = control.var / control.n + treatment.var / treatment.n
        llr = float(msprt_log_lr(np.array(est), np.array(var), self.tau2, self.theta0))
        self.p_value = min(self.p_value, min(1.0, math.exp(-llr)))
        hw = float(msprt_ci_halfwidth(np.array(var), self.tau2, self.alpha))
        self.ci_low = max(self.ci_low, est - hw)
        self.ci_high = min(self.ci_high, est + hw)
        if self.stopped_at is None and self.p_value <= self.alpha:
            self.stopped_at = self.looks
        return self.p_value <= self.alpha


# ------------------------------------------------------------------------ power


def analytic_power_welch(effect: float, sigma: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """Exact power of the equal-n, equal-variance two-sided t-test via the noncentral t."""
    df = 2 * n_per_arm - 2
    ncp = effect / (sigma * math.sqrt(2.0 / n_per_arm))
    q = sps.t.ppf(1 - alpha / 2, df)
    return float(sps.nct.sf(q, df, ncp) + sps.nct.cdf(-q, df, ncp))


def required_n_per_arm(effect: float, sigma: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Normal-approximation sample size per arm for a two-sided test."""
    za = sps.norm.ppf(1 - alpha / 2)
    zb = sps.norm.ppf(power)
    return int(math.ceil(2 * (za + zb) ** 2 * sigma**2 / effect**2))
