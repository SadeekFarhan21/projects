"""Feature operators on (T, N) float64 arrays.

Contract: an op is causal if out[t] depends only on inputs[:t+1]. Every op in CAUSAL_OPS
satisfies this; the ops in LEAKY_OPS deliberately do not and exist so the leakage audit
has something to catch (tests and the leakage experiment use them).

Rolling windows use a masked cumulative-sum trick so NaNs (unlisted or missing bars)
do not poison the whole series: a window is valid only if all `w` values are finite.
"""

from __future__ import annotations

import numpy as np

from qrp.panel import simple_returns


def _shift(x: np.ndarray, n: int) -> np.ndarray:
    """out[t] = x[t-n] for n > 0 (lag), x[t+n] for n < 0 (lead). Fills with NaN."""
    out = np.full_like(x, np.nan)
    if n > 0:
        out[n:] = x[:-n]
    elif n < 0:
        out[:n] = x[-n:]
    else:
        out[:] = x
    return out


def _window_sums(x: np.ndarray, w: int):
    """Return (sum, sumsq, count) over trailing windows [t-w+1, t]."""
    finite = np.isfinite(x)
    xz = np.where(finite, x, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    c1 = np.concatenate([pad, np.cumsum(xz, axis=0)])
    c2 = np.concatenate([pad, np.cumsum(xz * xz, axis=0)])
    cn = np.concatenate([pad, np.cumsum(finite, axis=0)])
    T = x.shape[0]
    s = np.full_like(x, np.nan)
    ss = np.full_like(x, np.nan)
    n = np.zeros_like(x)
    if w <= T:
        s[w - 1:] = c1[w:] - c1[:-w]
        ss[w - 1:] = c2[w:] - c2[:-w]
        n[w - 1:] = cn[w:] - cn[:-w]
    return s, ss, n


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    s, _, n = _window_sums(x, w)
    return np.where(n == w, s / w, np.nan)


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    s, ss, n = _window_sums(x, w)
    var = (ss - s * s / w) / (w - 1)
    return np.where(n == w, np.sqrt(np.maximum(var, 0.0)), np.nan)


# ---------------------------------------------------------------- causal ops
def op_returns(close, window: int = 1):
    return close / _shift(close, window) - 1.0


def op_momentum(close, window: int = 20, skip: int = 0):
    lagged = _shift(close, skip)
    return lagged / _shift(close, skip + window) - 1.0


def op_volatility(close, window: int = 20):
    return rolling_std(simple_returns(close), window)


def op_ma_ratio(close, window: int = 20):
    return close / rolling_mean(close, window) - 1.0


def op_volume_z(quote_volume, window: int = 20):
    lv = np.log(np.where(quote_volume > 0, quote_volume, np.nan))
    return (lv - rolling_mean(lv, window)) / rolling_std(lv, window)


def op_zscore_ts(x, window: int = 60):
    return (x - rolling_mean(x, window)) / rolling_std(x, window)


def op_lag(x, n: int = 1):
    return _shift(x, n)


def op_xs_rank(x):
    """Cross-sectional rank per row, scaled to [-0.5, 0.5]. Uses only row t."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        m = np.isfinite(row)
        k = int(m.sum())
        if k == 0:
            continue
        ranks = np.empty(k)
        ranks[np.argsort(row[m], kind="stable")] = np.arange(k)
        out[t, m] = ranks / max(k - 1, 1) - 0.5
    return out


def op_xs_zscore(x):
    mu = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True)
    sd = np.nanstd(np.where(np.isfinite(x), x, np.nan), axis=1, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, np.nan)


# ---------------------------------------------------------------- leaky ops (for tests)
def leaky_forward_return(close, window: int = 5):
    """Blatant: uses the future price."""
    return _shift(close, -window) / close - 1.0


def leaky_centered_mean(close, window: int = 21):
    """Subtle: a centered moving average (pandas rolling(center=True)) sees w//2 bars ahead."""
    half = window // 2
    return close / _shift(rolling_mean(close, window), -half) - 1.0


def leaky_zscore_full(x):
    """Subtle: normalizes each series with its full-sample mean and std."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=0, keepdims=True)
        sd = np.nanstd(x, axis=0, keepdims=True)
    return (x - mu) / sd


# op name -> (function, default input names). Inputs are panel fields or feature names.
CAUSAL_OPS = {
    "returns": (op_returns, ["close"]),
    "momentum": (op_momentum, ["close"]),
    "volatility": (op_volatility, ["close"]),
    "ma_ratio": (op_ma_ratio, ["close"]),
    "volume_z": (op_volume_z, ["quote_volume"]),
    "zscore_ts": (op_zscore_ts, None),
    "lag": (op_lag, None),
    "xs_rank": (op_xs_rank, None),
    "xs_zscore": (op_xs_zscore, None),
}
LEAKY_OPS = {
    "leaky_forward_return": (leaky_forward_return, ["close"]),
    "leaky_centered_mean": (leaky_centered_mean, ["close"]),
    "leaky_zscore_full": (leaky_zscore_full, None),
}
ALL_OPS = {**CAUSAL_OPS, **LEAKY_OPS}
