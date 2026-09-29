"""Wide (dates x symbols) numpy panel. Every downstream stage works on these arrays.

Long parquet is the storage format; wide float64 arrays are the compute format. A
missing bar is NaN, never forward filled at this layer, so "was it tradeable" stays
recoverable (np.isfinite(close)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


@dataclass
class Panel:
    dates: np.ndarray                 # (T,) datetime64[D], strictly increasing
    symbols: list[str]                # (N,)
    fields: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (T, N) float64

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.dates), len(self.symbols)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]

    def slice_time(self, stop: int) -> "Panel":
        """Rows [0, stop). Used by the truncation leakage test."""
        return Panel(self.dates[:stop], list(self.symbols),
                     {k: v[:stop].copy() for k, v in self.fields.items()})

    def copy(self) -> "Panel":
        return Panel(self.dates.copy(), list(self.symbols),
                     {k: v.copy() for k, v in self.fields.items()})

    @staticmethod
    def from_long(df: pl.DataFrame, fields=FIELDS) -> "Panel":
        dates = np.array(sorted(df["date"].unique().to_list()), dtype="datetime64[D]")
        symbols = sorted(df["symbol"].unique().to_list())
        t_idx = np.searchsorted(dates, df["date"].to_numpy().astype("datetime64[D]"))
        s_map = {s: i for i, s in enumerate(symbols)}
        n_idx = np.array([s_map[s] for s in df["symbol"].to_list()])
        out = {}
        for f in fields:
            a = np.full((len(dates), len(symbols)), np.nan)
            a[t_idx, n_idx] = df[f].to_numpy().astype(np.float64)
            out[f] = a
        return Panel(dates, symbols, out)


def split_relisted(df: pl.DataFrame, max_gap_days: int = 3) -> pl.DataFrame:
    """Treat a symbol that stops trading for more than max_gap_days and comes back as a new
    instrument ("LUNAUSDT" then "LUNAUSDT#2").

    Exchanges reuse tickers (LUNA 2.0 relisted as LUNAUSDT 18 days after the old LUNA
    died at $0.00005) and redenominate after pauses. A forward-filled price across such a
    gap produces fake returns of 1000x or more. Splitting makes the old instrument end at
    its last traded price (a delisting) and starts the new one with no history.
    """
    gap = pl.col("date").diff().over("symbol").dt.total_days()
    seg = (gap > max_gap_days).fill_null(False).cast(pl.Int32).cum_sum().over("symbol")
    return (df.sort("symbol", "date")
              .with_columns(seg.alias("_seg"))
              .with_columns(pl.when(pl.col("_seg") == 0).then(pl.col("symbol"))
                            .otherwise(pl.col("symbol") + "#" + (pl.col("_seg") + 1).cast(pl.Utf8))
                            .alias("symbol"))
              .drop("_seg"))


def simple_returns(close: np.ndarray) -> np.ndarray:
    """r[t] = close[t]/close[t-1] - 1. NaN if either side is missing. r[0] = NaN."""
    r = np.full_like(close, np.nan)
    r[1:] = close[1:] / close[:-1] - 1.0
    return r


def synthetic_panel(T: int = 500, N: int = 50, seed: int = 0, missing: float = 0.0,
                    start: str = "2020-01-01") -> Panel:
    """GBM prices with staggered listings and optional random gaps. For tests and benchmarks."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.01, 0.05, N)
    rets = rng.standard_normal((T, N)) * vol + 0.0002
    close = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    listing = rng.integers(0, T // 4, N)
    listing[: max(1, N // 2)] = 0
    for i in range(N):
        close[: listing[i], i] = np.nan
    if missing > 0:
        close[rng.random((T, N)) < missing] = np.nan
    qv = np.exp(rng.normal(15, 1, (T, N))) * np.isfinite(close)
    qv[~np.isfinite(close)] = np.nan
    dates = np.datetime64(start, "D") + np.arange(T)
    fields = {"close": close, "open": close, "high": close * 1.01, "low": close * 0.99,
              "volume": qv / np.where(np.isfinite(close), close, 1.0), "quote_volume": qv}
    return Panel(dates, [f"S{i:04d}" for i in range(N)], fields)
