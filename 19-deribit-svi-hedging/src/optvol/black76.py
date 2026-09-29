"""Black-76 pricing on the forward with analytic Greeks.

Conventions
-----------
* ``F`` is the forward (Deribit: the expiry's future or synthetic underlying).
* ``K`` strike, ``T`` year fraction (ACT/365 in seconds), ``sigma`` Black vol.
* ``D`` discount factor to expiry. Deribit coin options are valued with a zero
  rate, so the default is 1.
* ``is_call`` is a boolean array; puts are ``~is_call``.

All prices are in the same currency as ``F`` and ``K`` (USD for Deribit once a
coin premium is multiplied by the underlying). ``coin_to_usd`` and
``usd_to_coin`` do that conversion explicitly.

Greeks are with respect to the forward and the vol, per unit of the option:
delta = dV/dF, gamma = d2V/dF2, vega = dV/dsigma (per 1.00 of vol),
theta = dV/dt = -dV/dT (per year, calendar time), vanna = d2V/dF dsigma,
volga = d2V/dsigma2.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@njit(cache=True, fastmath=False)
def norm_cdf(x: float) -> float:
    # erfc keeps full relative precision in the lower tail, unlike 0.5*(1+erf)
    return 0.5 * math.erfc(-x / SQRT2)


@njit(cache=True, fastmath=False)
def norm_pdf(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True)
def _d1d2(F, K, T, sigma):
    v = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return d1, d1 - v, v


@njit(cache=True)
def price_scalar(F, K, T, sigma, D, is_call) -> float:
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = F - K if is_call else K - F
        return D * max(intrinsic, 0.0)
    d1, d2, _ = _d1d2(F, K, T, sigma)
    if is_call:
        return D * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return D * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


@njit(cache=True)
def greeks_scalar(F, K, T, sigma, D, is_call):
    """Return (price, delta, gamma, vega, theta, vanna, volga)."""
    sqT = math.sqrt(T)
    d1, d2, v = _d1d2(F, K, T, sigma)
    pdf1 = norm_pdf(d1)
    if is_call:
        price = D * (F * norm_cdf(d1) - K * norm_cdf(d2))
        delta = D * norm_cdf(d1)
    else:
        price = D * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
        delta = -D * norm_cdf(-d1)
    gamma = D * pdf1 / (F * v)
    vega = D * F * pdf1 * sqT
    # With D fixed (zero rate, Deribit convention) theta is pure time decay.
    theta = -D * F * pdf1 * sigma / (2.0 * sqT)
    vanna = -D * pdf1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return price, delta, gamma, vega, theta, vanna, volga


@njit(cache=True, parallel=True)
def _price_vec(F, K, T, sigma, D, is_call, out):
    for i in prange(F.shape[0]):
        out[i] = price_scalar(F[i], K[i], T[i], sigma[i], D[i], is_call[i])


@njit(cache=True, parallel=True)
def _greeks_vec(F, K, T, sigma, D, is_call, out):
    for i in prange(F.shape[0]):
        g = greeks_scalar(F[i], K[i], T[i], sigma[i], D[i], is_call[i])
        for j in range(7):
            out[j, i] = g[j]


def _flat(a, shape, dtype):
    """Contiguous 1-D array of ``shape`` (writeable, as numba prefers).
    Inputs already at the target shape are not copied; scalars and smaller
    shapes are expanded with np.full / a copy of a broadcast view."""
    a = np.asarray(a, dtype=dtype)
    if a.shape == shape:
        return np.ascontiguousarray(a).reshape(-1)
    if a.ndim == 0:
        return np.full(int(np.prod(shape)), a, dtype=dtype)
    return np.array(np.broadcast_to(a, shape), dtype=dtype).reshape(-1)


def _broadcast_flat(dtypes, *arrays):
    shape = np.broadcast_shapes(*(np.shape(a) for a in arrays))
    return [_flat(a, shape, d) for a, d in zip(arrays, dtypes)], shape


def _broadcast(F, K, T, sigma, D, is_call):
    f8 = np.float64
    return _broadcast_flat((f8, f8, f8, f8, f8, np.bool_), F, K, T, sigma, D, is_call)


def price(F, K, T, sigma, is_call, D=1.0):
    """Vectorized Black-76 price."""
    (F_, K_, T_, s_, D_, c_), shape = _broadcast(F, K, T, sigma, D, is_call)
    out = np.empty(F_.shape[0])
    _price_vec(F_, K_, T_, s_, D_, c_, out)
    return out.reshape(shape) if shape else out[0]


GREEK_NAMES = ("price", "delta", "gamma", "vega", "theta", "vanna", "volga")


def greeks(F, K, T, sigma, is_call, D=1.0) -> dict[str, np.ndarray]:
    """Vectorized price and Greeks, returned as a dict of arrays."""
    (F_, K_, T_, s_, D_, c_), shape = _broadcast(F, K, T, sigma, D, is_call)
    out = np.empty((7, F_.shape[0]))
    _greeks_vec(F_, K_, T_, s_, D_, c_, out)
    return {n: out[j].reshape(shape) for j, n in enumerate(GREEK_NAMES)}


def coin_to_usd(premium_coin, underlying_price):
    """Deribit inverse options quote premium in coin; USD value = coin * underlying."""
    return np.asarray(premium_coin) * np.asarray(underlying_price)


def usd_to_coin(premium_usd, underlying_price):
    return np.asarray(premium_usd) / np.asarray(underlying_price)
