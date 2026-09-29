"""Vectorized Black-76 implied volatility.

Pipeline per option (all inside one numba kernel, parallel over options):

1. Normalize: c = V / (D F), kappa = K / F. Check no-arbitrage bounds
   intrinsic <= c <= upper (call upper 1, put upper kappa). Prices outside get a
   status code and NaN vol instead of a garbage number.
2. Convert to the out-of-the-money price with put-call parity, so the solver
   never works on a tiny time value buried under a large intrinsic value.
3. Initial guess in total vol v = sigma sqrt(T): Corrado-Miller closed form
   when its discriminant is positive, otherwise the inflection point
   sqrt(2 |ln kappa|) where the price is steepest in v.
4. Safeguarded Newton on log(price) in v. A bracket [lo, hi] is kept; any
   step that leaves it is replaced by bisection (in log space for hi = inf).
5. Options that did not converge in MAX_ITER are re-solved by scipy's Brent
   on the same objective (the fallback almost never triggers; the count is
   reported).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq

from .black76 import _broadcast_flat, norm_cdf, norm_pdf

# status codes
OK = 0
BELOW_INTRINSIC = 1  # price < intrinsic (arbitrage), vol undefined
ABOVE_UPPER = 2  # price > upper bound (call > D F, put > D K)
AT_INTRINSIC = 3  # price == intrinsic within tolerance, vol = 0
BAD_INPUT = 4  # non-finite or non-positive F, K, T
NOT_CONVERGED = 5  # Newton failed, Brent fallback pending
BRENT = 6  # solved by the Brent fallback

MAX_ITER = 40
V_MAX = 20.0  # total vol cap, sigma sqrt(T); far beyond anything quoted


@njit(cache=True)
def _otm_norm(v, kappa, is_call_otm):
    """Normalized OTM Black price (F=1, D=1) at total vol v."""
    d1 = -math.log(kappa) / v + 0.5 * v
    d2 = d1 - v
    if is_call_otm:
        return norm_cdf(d1) - kappa * norm_cdf(d2)
    return kappa * norm_cdf(-d2) - norm_cdf(-d1)


@njit(cache=True)
def _otm_vega(v, kappa):
    d1 = -math.log(kappa) / v + 0.5 * v
    return norm_pdf(d1)  # d c / d v for F = 1


@njit(cache=True)
def _solve_one(price, F, K, T, D, is_call, tol):
    """Return (sigma, status, iterations)."""
    if not (F > 0.0 and K > 0.0 and T > 0.0 and D > 0.0) or not math.isfinite(price):
        return math.nan, BAD_INPUT, 0
    c = price / (D * F)
    kappa = K / F
    intrinsic = max(1.0 - kappa, 0.0) if is_call else max(kappa - 1.0, 0.0)
    upper = 1.0 if is_call else kappa
    eps = 1e-15 * max(1.0, kappa)
    if c < intrinsic - eps:
        return math.nan, BELOW_INTRINSIC, 0
    if c > upper + eps:
        return math.nan, ABOVE_UPPER, 0
    # out-of-the-money equivalent
    call_otm = kappa >= 1.0
    if is_call == call_otm:
        target = c
    elif is_call:  # ITM call -> OTM put via parity: p = c - (1 - kappa)
        target = c - (1.0 - kappa)
    else:  # ITM put -> OTM call: c = p - (kappa - 1)
        target = c - (kappa - 1.0)
    otm_upper = 1.0 if call_otm else kappa
    if target <= eps:
        return 0.0, AT_INTRINSIC, 0
    if target >= otm_upper - eps:
        return math.nan, ABOVE_UPPER, 0

    x = math.log(kappa)
    # Corrado-Miller guess (F = 1, strike kappa)
    h = c - 0.5 * (1.0 - kappa) if is_call else c - 0.5 * (kappa - 1.0)
    disc = h * h - (1.0 - kappa) * (1.0 - kappa) / math.pi
    v_infl = math.sqrt(2.0 * abs(x))
    if disc > 0.0:
        v = math.sqrt(2.0 * math.pi) / (1.0 + kappa) * (h + math.sqrt(disc))
    else:
        v = v_infl
    if not (v > 1e-8) or v > V_MAX:
        v = max(v_infl, 1e-3)

    lo, hi = 0.0, V_MAX
    log_t = math.log(target)
    it = 0
    for it in range(1, MAX_ITER + 1):
        p = _otm_norm(v, kappa, call_otm)
        if p <= 0.0:  # underflow: vol too small
            lo = v
            v = 0.5 * (lo + hi) if hi < V_MAX else 2.0 * v
            continue
        f = math.log(p) - log_t
        if f > 0.0:
            hi = v
        else:
            lo = v
        if f == 0.0:
            return v / math.sqrt(T), OK, it
        vega = _otm_vega(v, kappa)
        step = f * p / vega if vega > 0.0 else math.inf
        if abs(step) <= tol * v:
            return (v - step) / math.sqrt(T), OK, it
        v_new = v - step
        if not (lo <= v_new <= hi) or not math.isfinite(v_new):
            # Newton left the bracket: bisect (geometric while hi is the cap)
            v_new = 0.5 * (lo + hi) if hi < V_MAX else min(2.0 * v, 0.5 * (v + hi))
        v = v_new
    return v / math.sqrt(T), NOT_CONVERGED, it


@njit(cache=True, parallel=True)
def _solve_vec(price, F, K, T, D, is_call, tol, sig, status, iters):
    for i in prange(price.shape[0]):
        s, st, it = _solve_one(price[i], F[i], K[i], T[i], D[i], is_call[i], tol)
        sig[i] = s
        status[i] = st
        iters[i] = it


def _brent_one(price, F, K, T, D, is_call):
    from .black76 import price_scalar

    def f(s):
        return price_scalar(F, K, T, s, D, is_call) - price

    lo, hi = 1e-6 / math.sqrt(T), V_MAX / math.sqrt(T)
    if f(lo) * f(hi) > 0:
        return math.nan
    return brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)


def implied_vol(price, F, K, T, is_call, D=1.0, tol=1e-14, return_info=False):
    """Vectorized implied vol. Returns sigma (and status, iterations if asked)."""
    f8 = np.float64
    (p, F_, K_, T_, D_, c_), shape = _broadcast_flat(
        (f8, f8, f8, f8, f8, np.bool_), price, F, K, T, D, is_call)
    n = p.shape[0]
    sig = np.empty(n)
    status = np.empty(n, np.int8)
    iters = np.empty(n, np.int16)
    _solve_vec(p, F_, K_, T_, D_, c_, tol, sig, status, iters)
    bad = np.flatnonzero(status == NOT_CONVERGED)
    for i in bad:
        s = _brent_one(p[i], F_[i], K_[i], T_[i], D_[i], bool(c_[i]))
        sig[i] = s
        status[i] = BRENT if math.isfinite(s) else NOT_CONVERGED
    sig = sig.reshape(shape)
    if return_info:
        return sig, status.reshape(shape), iters.reshape(shape)
    return sig
