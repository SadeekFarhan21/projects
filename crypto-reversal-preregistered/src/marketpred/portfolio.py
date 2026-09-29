"""Forecast -> dollar-neutral weights -> daily PnL net of costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def weights_rank(forecast: pd.Series) -> pd.Series:
    """Weights proportional to the centered rank of the forecast, per date,
    scaled so sum|w| = 1 (0.5 long, 0.5 short)."""
    g = forecast.groupby(level="date")
    r = g.rank(method="average")
    n = g.transform("count")
    c = (r - 1) / (n - 1).where(n > 1) - 0.5
    return c.div(c.abs().groupby(level="date").transform("sum")).fillna(0.0)


def weights_quintile(forecast: pd.Series, q: float = 0.2) -> pd.Series:
    """Equal-weight long the top q, short the bottom q; gross exposure 1."""
    g = forecast.groupby(level="date")
    pct = g.rank(method="first", pct=True)
    long = pct > 1 - q
    short = pct <= q
    nl = long.groupby(level="date").transform("sum")
    ns = short.groupby(level="date").transform("sum")
    w = 0.5 * long / nl.where(nl > 0) - 0.5 * short / ns.where(ns > 0)
    return w.fillna(0.0)


CONSTRUCTIONS = {"rank": weights_rank, "quintile": weights_quintile}


@dataclass
class Backtest:
    daily: pd.DataFrame  # columns: gross, turnover, and net_<bps> per cost level

    def net(self, bps: float) -> pd.Series:
        return self.daily["gross"] - self.daily["turnover"] * bps / 1e4


def backtest(weights: pd.Series, fwd_ret: pd.Series, lag: int = 0) -> Backtest:
    """Daily PnL of holding `weights` (formed at close of t) over t -> t+1.

    fwd_ret: forward return on the same (date, symbol) index. A missing forward
    return (delisting, data gap) counts as 0: the weight was chosen without
    knowing the bar would be missing.
    lag: extra days between signal and trade (robustness check).
    """
    W = weights.unstack("symbol").fillna(0.0).sort_index()
    R = fwd_ret.unstack("symbol").reindex_like(W).fillna(0.0)
    if lag:
        W = W.shift(lag).fillna(0.0)
    gross = (W * R).sum(axis=1)
    # Turnover ignores intraday drift of weights, as pre-registered.
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    return Backtest(pd.DataFrame({"gross": gross, "turnover": turnover}))
