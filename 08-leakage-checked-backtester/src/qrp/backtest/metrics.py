from __future__ import annotations

import numpy as np


def summarize(returns: np.ndarray, turnover: np.ndarray | None = None,
              cost: np.ndarray | None = None, equity: np.ndarray | None = None,
              periods_per_year: int = 365, mask: np.ndarray | None = None) -> dict:
    """Standard performance stats for one daily return series.

    mask selects the evaluation rows (e.g. out-of-sample test rows only).
    Crypto trades every day, so the default annualization is 365.
    """
    r = returns if mask is None else returns[mask]
    n = len(r)
    if n < 2:
        return {"n_days": n}
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    dd = float(np.min(curve / peak - 1))
    out = {
        "n_days": n,
        "ann_return": float(curve[-1] ** (periods_per_year / n) - 1) if curve[-1] > 0 else -1.0,
        "ann_vol": sd * np.sqrt(periods_per_year),
        "sharpe": mu / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0,
        "max_drawdown": dd,
        "hit_rate": float(np.mean(r > 0)),
        "total_return": float(curve[-1] - 1),
    }
    if turnover is not None:
        tv = turnover if mask is None else turnover[mask]
        out["avg_daily_turnover"] = float(np.mean(tv))
    if cost is not None and equity is not None:
        c = cost if mask is None else cost[mask]
        e = equity if mask is None else equity[mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ann_cost_drag"] = float(np.nanmean(np.where(e > 0, c / e, 0.0)) * periods_per_year)
    return out
