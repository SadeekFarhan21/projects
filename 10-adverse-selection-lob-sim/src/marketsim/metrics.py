"""Stylized-fact statistics and market-maker PnL decomposition."""

from __future__ import annotations

import numpy as np

from .simulation import SimResult


def log_returns(prices: np.ndarray, every: int = 1) -> np.ndarray:
    """Log returns of a regularly sampled price series, subsampled by ``every``."""
    p = np.asarray(prices, dtype=float)[::every]
    return np.diff(np.log(p))


def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = np.mean(x * x)
    return float(np.mean(x ** 4) / var ** 2 - 3.0) if var > 0 else float("nan")


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (biased estimator, as usual)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.full(max_lag, np.nan)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def hill_tail_index(x: np.ndarray, tail_frac: float = 0.02) -> float:
    """Hill estimator of the tail exponent of |x| using the top ``tail_frac``.
    Gaussian data gives large values (thin tails); equity returns give ~3."""
    a = np.sort(np.abs(np.asarray(x, dtype=float)))[::-1]
    a = a[a > 0]
    k = max(int(len(a) * tail_frac), 10)
    if len(a) <= k:
        return float("nan")
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def mm_pnl_decomposition(res: SimResult) -> dict[str, float]:
    """Split market-maker PnL (marked at the final mid) into three parts.

    For a fill with sign s (+1 buy), price p, size q, mid m0 just before the
    aggressing order, mid m_h one markout horizon later, and final mid m_T:

        s*q*(m_T - p) = s*q*(m0 - p)        spread capture
                      + s*q*(m_h - m0)      adverse selection (markout)
                      + s*q*(m_T - m_h)     inventory carry

    Summed over fills the left side is exactly cash + inventory * m_T, since
    the MM starts flat with zero cash. The identity is checked in the tests.
    """
    f = res.mm_fills
    if len(f["t"]) == 0:
        return {k: 0.0 for k in ("total", "spread_capture", "adverse_selection", "inventory_carry",
                                 "n_fills", "spread_capture_per_fill", "adverse_selection_per_fill",
                                 "edge_vs_fundamental_per_fill")}
    s, p, q = f["side"], f["price"], f["qty"]
    m0, mh, mT = f["mid_before"], f["mid_after_h"], res.final_mid
    sc = float(np.sum(s * q * (m0 - p)))
    adv = float(np.sum(s * q * (mh - m0)))
    carry = float(np.sum(s * q * (mT - mh)))
    n = float(np.sum(q))
    return {
        "total": sc + adv + carry,
        "spread_capture": sc,
        "adverse_selection": adv,
        "inventory_carry": carry,
        "n_fills": n,
        "spread_capture_per_fill": sc / n,
        "adverse_selection_per_fill": adv / n,
        # Ground truth only a simulator has: edge against the true value at fill.
        "edge_vs_fundamental_per_fill": float(np.sum(s * q * (f["fundamental"] - p)) / n),
    }


def summarize(res: SimResult, return_every: int = 10, max_lag: int = 50) -> dict[str, float]:
    """Headline numbers for one run."""
    smp = res.samples
    r = log_returns(smp["mid"], return_every)
    r = r[np.isfinite(r)]
    ac_abs = acf(np.abs(r), max_lag)
    ac_raw = acf(r, max_lag)
    spread = smp["spread"][np.isfinite(smp["spread"])]
    dec = mm_pnl_decomposition(res)
    c = res.counters
    track_err = smp["mid"] - smp["fundamental"]
    out = {
        "informed_frac": res.config.informed_frac,
        "mm_adaptive": float(res.config.mm_adaptive),
        "seed": res.config.seed,
        "n_returns": len(r),
        "return_std": float(np.std(r)),
        "excess_kurtosis": excess_kurtosis(r),
        "hill_tail_index": hill_tail_index(r),
        "acf_ret_lag1": float(ac_raw[0]),
        "acf_abs_lag1": float(ac_abs[0]),
        "acf_abs_lag5": float(ac_abs[4]),
        "acf_abs_lag20": float(ac_abs[19]) if max_lag >= 20 else float("nan"),
        "mean_spread": float(np.mean(spread)),
        "median_spread": float(np.median(spread)),
        "p95_spread": float(np.percentile(spread, 95)),
        "mean_mm_quote_spread": float(np.nanmean(smp["mm_quote_spread"])),
        "frac_one_sided": float(np.mean(~np.isfinite(smp["spread"]))),
        "mean_abs_mid_minus_V": float(np.mean(np.abs(track_err))),
        "mm_pnl_total": dec["total"],
        "mm_spread_capture": dec["spread_capture"],
        "mm_adverse_selection": dec["adverse_selection"],
        "mm_inventory_carry": dec["inventory_carry"],
        "mm_n_fills": dec["n_fills"],
        "mm_pnl_per_fill": dec["total"] / dec["n_fills"] if dec["n_fills"] else float("nan"),
        "mm_spread_capture_per_fill": dec["spread_capture_per_fill"],
        "mm_adverse_selection_per_fill": dec["adverse_selection_per_fill"],
        "mm_edge_vs_V_per_fill": dec["edge_vs_fundamental_per_fill"],
        "mm_max_abs_inventory": float(np.max(np.abs(smp["mm_inventory"]))),
        "informed_pnl": c["informed_pnl_fundamental"],
        "noise_pnl": c["noise_pnl_fundamental"],
        "noise_lp_pnl": c["noise_lp_pnl_fundamental"],
        "mm_pnl_vs_V": c.get("mm_pnl_fundamental", 0.0),
        "informed_orders": c["informed_orders"],
        "informed_share_of_trades": c["informed_orders"] / max(1, c["informed_orders"] + c["noise_orders"]),
        # Glosten-Milgrom breakeven half-spread: what informed traders take per
        # taker trade, which a zero-profit dealer must recover from every trade.
        # Uses edge at trade time (V_t - price), not end-of-run marks, so the
        # informed traders' unhedged inventory does not add noise.
        "gm_breakeven_half_spread": c["informed_edge_vs_V"] / max(1, c["informed_orders"] + c["noise_orders"]),
        "informed_edge_per_trade": c["informed_edge_vs_V"] / max(1, c["informed_orders"]),
        "informed_edge": c["informed_edge_vs_V"],
        "noise_edge": c["noise_edge_vs_V"],
        "noise_lp_edge": c["noise_lp_edge_vs_V"],
        "mm_edge": c.get("mm_edge_vs_V", 0.0),
        "noise_orders": c["noise_orders"],
        "events": c["events"],
        "events_per_s": c["events_per_s"],
    }
    return out
