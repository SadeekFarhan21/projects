"""Evaluation statistics: IC, Newey-West t, Sharpe, PSR, deflated Sharpe."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ANN = 365  # crypto trades every day
EULER_GAMMA = 0.5772156649015329


def daily_ic(forecast: pd.Series, fwd_ret: pd.Series, min_names: int = 10) -> pd.Series:
    """Spearman rank IC per date, over names with a non-missing forward return."""
    df = pd.DataFrame({"f": forecast, "y": fwd_ret}).dropna()
    g = df.groupby(level="date")
    fr = g["f"].rank()
    yr = g["y"].rank()
    df = df.assign(fr=fr, yr=yr)
    out = df.groupby(level="date")[["fr", "yr"]].apply(
        lambda x: x["fr"].corr(x["yr"]) if len(x) >= min_names else np.nan)
    return out.dropna()


def newey_west_t(x: pd.Series | np.ndarray, lags: int = 5) -> float:
    """t-stat of the mean with a Bartlett-kernel HAC standard error."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def sharpe(r: pd.Series, periods: int = ANN) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def psr(r: pd.Series, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey and Lopez de Prado 2012).

    Works on per-period (non-annualized) Sharpe; `sr_star` is per-period too.
    Adjusts for sample length, skew and kurtosis of returns.
    """
    r = pd.Series(r).dropna().to_numpy()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)  # raw kurtosis, 3 for a normal
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials per-period Sharpe estimates under the null
    of zero true Sharpe (the SR* threshold of the deflated Sharpe ratio)."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpes: list[float]) -> dict:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014).

    trial_sharpes: per-period Sharpe of every trial, used for their variance.
    Returns the PSR of the chosen strategy against the expected max of
    n_trials null Sharpes.
    """
    var_sr = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return {"dsr": psr(r, sr_star), "sr_star_ann": sr_star * math.sqrt(ANN),
            "n_trials": n_trials, "var_sr": var_sr}


def summarize(net: pd.Series, gross: pd.Series, turnover: pd.Series,
              ic: pd.Series | None = None) -> dict:
    out = {
        "days": int(net.notna().sum()),
        "ann_return_gross": float(gross.mean() * ANN),
        "ann_return_net": float(net.mean() * ANN),
        "ann_vol": float(net.std(ddof=1) * math.sqrt(ANN)),
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "psr_net": psr(net),
        "max_drawdown_net": max_drawdown(net),
        "avg_turnover": float(turnover.mean()),
    }
    if ic is not None and len(ic):
        out.update({
            "ic_mean": float(ic.mean()),
            "ic_t_nw": newey_west_t(ic, 5),
            "ic_ir_ann": float(ic.mean() / ic.std(ddof=1) * math.sqrt(ANN)),
            "ic_hit_rate": float((ic > 0).mean()),
        })
    return out
