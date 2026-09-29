"""Panel construction: wide price frames -> features, target, universe.

Conventions
- Wide frames are indexed by a complete daily DatetimeIndex (no missing days)
  with one column per symbol episode; NaN means no bar that day.
- Feature at row t uses closes up to and including t.
- `fwd_ret` at row t is close[t+1] / close[t] - 1, the return earned by a
  position formed at the close of t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import eligibility

FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "vol_20d", "dvol_20d",
    "amihud_20d", "dist_high_20d", "beta_btc_60d",
]
RELIST_GAP_DAYS = 7


def split_relistings(panel: pd.DataFrame, gap_days: int = RELIST_GAP_DAYS) -> pd.DataFrame:
    """Give each continuous listing episode its own symbol id.

    Binance reuses tickers (LUNAUSDT was the old Terra until May 2022 and the
    new Terra from June 2022). A gap longer than `gap_days` starts a new
    episode `SYM#2`, so no return is ever computed across two different assets.
    """
    panel = panel.sort_values(["symbol", "date"]).copy()
    gap = panel.groupby("symbol")["date"].diff().dt.days
    episode = (gap > gap_days).groupby(panel["symbol"]).cumsum().astype(int)
    panel["symbol"] = np.where(episode > 0,
                               panel["symbol"] + "#" + (episode + 1).astype(str),
                               panel["symbol"])
    return panel.sort_values(["date", "symbol"], ignore_index=True)


def to_wide(panel: pd.DataFrame, col: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return panel.pivot(index="date", columns="symbol", values=col).reindex(index)


def cs_rank(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Centered cross-sectional rank in [-0.5, 0.5] among mask members.
    Members with a NaN value get 0 (the cross-sectional median)."""
    x = wide.where(mask)
    r = x.rank(axis=1, method="average")
    n = r.count(axis=1)
    out = r.sub(1).div((n - 1).where(n > 1), axis=0) - 0.5
    return out.where(mask).fillna(0.0).where(mask)


@dataclass
class Dataset:
    long: pd.DataFrame          # (date, symbol) rows in universe: ranked features + targets
    raw: dict[str, pd.DataFrame]  # wide raw features, for inspection
    mask: pd.DataFrame          # wide universe membership
    fwd_ret: pd.DataFrame       # wide forward returns (NaN where no next bar)


def build_dataset(panel: pd.DataFrame, end: str | None = None,
                  btc: str = "BTCUSDT") -> Dataset:
    """Build features/targets. `end` truncates the raw data (inclusive) so
    nothing after it can leak into any computation."""
    if end is not None:
        panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel = split_relistings(panel)
    idx = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    close = to_wide(panel, "close", idx)
    high = to_wide(panel, "high", idx)
    dvol = to_wide(panel, "quote_volume", idx)

    ret1 = close / close.shift(1) - 1
    logret = np.log1p(ret1)
    raw = {
        "ret_1d": ret1,
        "ret_5d": close / close.shift(5) - 1,
        "ret_20d": close / close.shift(20) - 1,
        "ret_60d": close / close.shift(60) - 1,
        "vol_20d": logret.rolling(20, min_periods=15).std(),
        "dvol_20d": np.log(dvol.rolling(20, min_periods=15).median().clip(lower=1.0)),
        "amihud_20d": (ret1.abs() / dvol.replace(0, np.nan)).rolling(20, min_periods=15).mean(),
        "dist_high_20d": close / high.rolling(20, min_periods=15).max() - 1,
    }
    b = logret[btc]
    cov = logret.rolling(60, min_periods=40).cov(b)
    var = b.rolling(60, min_periods=40).var()
    raw["beta_btc_60d"] = cov.div(var, axis=0)

    mask = eligibility(close, dvol)
    fwd = close.shift(-1) / close - 1

    ranked = {k: cs_rank(v, mask) for k, v in raw.items()}
    # Training label: centered rank of the forward return among members that
    # have a next bar. Evaluation uses the raw forward return instead.
    has_fwd = mask & fwd.notna()
    y_rank = cs_rank(fwd, has_fwd)

    # pandas 3 stack() keeps NaN, so select member rows explicitly.
    m = mask.stack()
    keep = m[m].index
    long = pd.DataFrame({k: v.stack().reindex(keep) for k, v in ranked.items()})
    long["fwd_ret"] = fwd.stack().reindex(keep)
    long["y_rank"] = y_rank.stack().reindex(keep)
    long.index.names = ["date", "symbol"]
    long = long.sort_index()
    return Dataset(long=long, raw=raw, mask=mask, fwd_ret=fwd)
