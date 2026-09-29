"""Raw SVI per expiry, SSVI across expiries, and static no-arbitrage checks.

Raw SVI total variance (Gatheral 2004):
    w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + s^2) )
with k = ln(K / F) and w = sigma_imp^2 T.

Constraints used in the fit (enforced by the parameterization or by bounds):
    b >= 0, |rho| < 1, s > 0
    a + b s sqrt(1 - rho^2) >= 0      (minimum total variance is non-negative)
    b (1 + |rho|) <= 2                 (Roger Lee moment bound on wing slopes)

Butterfly condition (Gatheral and Jacquier 2014): the density is non-negative
iff g(k) >= 0 where
    g = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1 / w + 1 / 4) + w'' / 2 .
Calendar condition: w(k, T) is non-decreasing in T for every k.

SSVI (Gatheral and Jacquier 2014) with the power-law phi:
    w(k, theta) = theta / 2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))
    phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))
It is free of static arbitrage when theta is non-decreasing in T,
eta (1 + |rho|) <= 2 and 0 < gamma <= 1/2. The fit enforces all three by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------- raw SVI
@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def as_tuple(self):
        return (self.a, self.b, self.rho, self.m, self.s)


def svi_w(k, p: SVIParams):
    x = np.asarray(k) - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.s * p.s))


def svi_w_derivs(k, p: SVIParams):
    x = np.asarray(k) - p.m
    r = np.sqrt(x * x + p.s * p.s)
    w = p.a + p.b * (p.rho * x + r)
    w1 = p.b * (p.rho + x / r)
    w2 = p.b * p.s * p.s / r**3
    return w, w1, w2


def g_function(w, w1, w2, k):
    """Gatheral-Jacquier butterfly density function. Returns -inf where w <= 0."""
    k = np.asarray(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2
    return np.where(w > 0, g, -np.inf)


def butterfly_ok(p: SVIParams, k_grid, tol: float = -1e-10) -> tuple[bool, float]:
    w, w1, w2 = svi_w_derivs(k_grid, p)
    g = g_function(w, w1, w2, k_grid)
    gmin = float(np.min(g))
    return gmin >= tol, gmin


def calendar_ok(w_short, w_long, tol: float = 1e-12) -> tuple[bool, float]:
    """True if w_long >= w_short on the grid. Returns the worst violation (negative)."""
    d = np.asarray(w_long) - np.asarray(w_short)
    dmin = float(np.min(d))
    return dmin >= -tol, dmin


@dataclass
class SVIFit:
    params: SVIParams
    T: float
    cost: float
    n: int
    success: bool
    rmse_vol: float = np.nan  # RMSE of implied vol, vol points (x100)
    k_range: tuple = field(default_factory=tuple)


def _unpack(x):
    amin, b, rho, m, s = x
    a = amin - b * s * np.sqrt(1 - rho * rho)
    return SVIParams(a, b, rho, m, s)


def fit_svi(k, w_mkt, weights=None, T: float = 1.0, n_starts: int = 6,
            lee_penalty: float = 1e3, x0: SVIParams | None = None,
            tol: float = 1e-10) -> SVIFit:
    """Weighted least squares of raw SVI to total variance.

    ``weights`` multiply squared residuals (we use vega / spread). Parameters are
    (amin, b, rho, m, s) with a = amin - b s sqrt(1 - rho^2), so the bound
    amin >= 0 is exactly the non-negative-minimum constraint. The Lee bound is
    a smooth penalty that is zero when satisfied.
    """
    k = np.asarray(k, float)
    w_mkt = np.asarray(w_mkt, float)
    wt = np.ones_like(k) if weights is None else np.asarray(weights, float)
    wt = wt / wt.mean()
    sw = np.sqrt(wt)
    scale = max(float(np.median(w_mkt)), 1e-12)  # residuals relative to typical w
    krange = float(k.max() - k.min()) if k.size > 1 else 0.1
    krange = max(krange, 1e-3)

    def resid(x):
        p = _unpack(x)
        r = sw * (svi_w(k, p) - w_mkt) / scale
        lee = max(0.0, p.b * (1 + abs(p.rho)) - 2.0)
        return np.append(r, lee_penalty * lee)

    lo = [0.0, 0.0, -0.999, k.min() - krange, 1e-5 * krange]
    hi = [max(2 * w_mkt.max(), 1e-8), 2.0, 0.999, k.max() + krange, 2.0 * krange + 1.0]
    # starts: parabola-ish guesses around the smile minimum
    i0 = int(np.argmin(w_mkt))
    wmin = float(w_mkt[i0])
    # wing slope estimate for b
    left = w_mkt[k < k[i0]]
    right = w_mkt[k > k[i0]]
    slope_r = (right.max() - wmin) / max(k.max() - k[i0], 1e-6) if right.size else 0.1 * wmin / krange
    slope_l = (left.max() - wmin) / max(k[i0] - k.min(), 1e-6) if left.size else 0.1 * wmin / krange
    b0 = np.clip(0.5 * (slope_r + slope_l), 1e-6, 1.9)
    rho0 = np.clip((slope_r - slope_l) / (slope_r + slope_l + 1e-12), -0.9, 0.9)
    starts = []
    for s_frac in (0.1, 0.3, 1.0):
        for rho_s in (rho0, -0.5 * np.sign(rho0 + 1e-9)):
            s0 = s_frac * krange
            starts.append([0.5 * wmin, b0, rho_s, k[i0], s0])
    starts = starts[:n_starts]
    if x0 is not None:  # warm start (e.g. previous minute's fit) goes first
        amin0 = x0.a + x0.b * x0.s * np.sqrt(1 - x0.rho**2)
        starts = [[amin0, x0.b, x0.rho, x0.m, x0.s]] + starts[:max(n_starts - 1, 1)]
    best = None
    for xs in starts:
        xs = np.clip(xs, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
        try:
            r = least_squares(resid, xs, bounds=(lo, hi), method="trf", x_scale="jac",
                              ftol=tol, xtol=tol, gtol=tol, max_nfev=1000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = _unpack(best.x)
    return SVIFit(p, T, float(best.cost), int(k.size), bool(best.success),
                  k_range=(float(k.min()), float(k.max())))


def svi_vol(k, p: SVIParams, T: float):
    return np.sqrt(np.maximum(svi_w(k, p), 0.0) / T)


# ---------------------------------------------------------------- SSVI
@dataclass
class SSVIParams:
    rho: float
    eta: float
    gamma: float
    theta: np.ndarray  # per expiry, non-decreasing in T

    def phi(self, theta):
        return self.eta / (theta**self.gamma * (1 + theta) ** (1 - self.gamma))


def ssvi_w(k, theta, rho, phi):
    pk = phi * k
    return 0.5 * theta * (1 + rho * pk + np.sqrt((pk + rho) ** 2 + 1 - rho * rho))


def _ssvi_unpack(x, n_exp):
    rho = np.tanh(x[0])
    gamma = 0.5 / (1 + np.exp(-x[1]))  # (0, 1/2)
    eta = 2.0 / (1 + abs(rho)) / (1 + np.exp(-x[2]))  # eta (1 + |rho|) < 2
    theta = np.cumsum(np.exp(np.clip(x[3:3 + n_exp], -60.0, 20.0)))  # strictly increasing
    return SSVIParams(rho, eta, gamma, theta)


def fit_ssvi(slices: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[SSVIParams, float]:
    """Global SSVI fit. ``slices`` is a list of (k, w_mkt, weights) sorted by T.
    Returns params and final cost. theta is initialized from each slice's ATM
    total variance (interpolated at k = 0) and made monotone."""
    n = len(slices)
    atm = []
    for k, w, _ in slices:
        o = np.argsort(k)
        atm.append(float(np.interp(0.0, k[o], w[o])))
    atm = np.maximum.accumulate(np.maximum(np.array(atm), 1e-8))
    inc = np.diff(np.concatenate([[0.0], atm]))
    inc = np.maximum(inc, 1e-3 * atm[0])
    x0 = np.concatenate([[np.arctanh(-0.1), 0.0, 0.0], np.log(inc)])
    scales = [max(float(np.median(w)), 1e-12) for _, w, _ in slices]
    sws = [np.sqrt(wt / wt.mean()) / np.sqrt(len(k)) for k, _, wt in slices]

    def resid(x):
        p = _ssvi_unpack(x, n)
        out = []
        for j, (k, w, _) in enumerate(slices):
            th = p.theta[j]
            out.append(sws[j] * (ssvi_w(k, th, p.rho, p.phi(th)) - w) / scales[j])
        return np.concatenate(out)

    r = least_squares(resid, x0, method="trf", x_scale="jac", max_nfev=5000,
                      ftol=1e-12, xtol=1e-12)
    return _ssvi_unpack(r.x, n), float(r.cost)


def ssvi_slice_params(p: SSVIParams, j: int) -> SVIParams:
    """SSVI slice j expressed as raw SVI (GJ 2014 Lemma 3.2), so the same
    butterfly and calendar checks apply."""
    th = p.theta[j]
    ph = p.phi(th)
    rho = p.rho
    a = 0.5 * th * (1 - rho * rho)
    b = 0.5 * th * ph
    m = -rho / ph
    s = np.sqrt(1 - rho * rho) / ph
    return SVIParams(a, b, rho, m, s)
