"""Loading and cleaning Deribit option chains.

Input is the per-minute file produced by ``tools/minute_snap`` from the
tardis.dev ``options_chain`` dataset: one row per symbol per minute holding
the last ticker state before each minute boundary (``snap_ts``).

Steps
-----
``load_minute_file``   read, keep coin-settled BTC and ETH options, parse
                       symbols, validate the 08:00 UTC expiry.
``snapshot``           as-of view of the whole chain at time t (last row per
                       symbol with snap_ts <= t, at most ``max_age`` old).
``clean``              drop missing, zero-bid, crossed and locked quotes;
                       compute coin and USD mids and exact T.
``fit_forwards``       per-expiry put-call parity regression for F.
``attach_iv``          implied vols of bid, ask, mid on the fitted forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from . import black76, iv

US_PER_YEAR = 365.0 * 86400.0 * 1e6
MIN_PARITY_PAIRS = 5
MAX_INTERCEPT_DEV = 0.01  # |a - 1|; coin premiums imply (C - P) = 1 - K / F
EXPIRY_HOUR_UTC = 8

FLOAT_COLS = [
    "strike_price", "open_interest", "last_price", "bid_price", "bid_amount", "bid_iv",
    "ask_price", "ask_amount", "ask_iv", "mark_price", "mark_iv", "underlying_price",
    "delta", "gamma", "vega", "theta", "rho",
]


def load_minute_file(path: str, coins=("BTC", "ETH")) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        schema_overrides={c: pl.Float64 for c in FLOAT_COLS}
        | {"snap_ts": pl.Int64, "timestamp": pl.Int64, "local_timestamp": pl.Int64,
           "expiration": pl.Int64},
        infer_schema_length=0,
    )
    df = df.with_columns(pl.col("symbol").str.split("-").alias("_parts"))
    df = df.with_columns(
        pl.col("_parts").list.get(0).alias("coin"),
        pl.col("_parts").list.get(1).alias("expiry_code"),
        (pl.col("type") == "call").alias("is_call"),
    ).drop("_parts")
    df = df.filter(pl.col("coin").is_in(list(coins)))
    # Deribit expiries are 08:00 UTC; assert rather than assume.
    hour = (pl.col("expiration") // 1_000_000 % 86400) // 3600
    bad = df.filter(hour != EXPIRY_HOUR_UTC)
    if bad.height:
        raise ValueError(f"{bad.height} rows with expiry not at 08:00 UTC, e.g. {bad['symbol'][0]}")
    return df


def snapshot(df: pl.DataFrame, t_us: int, max_age_s: float = 600.0) -> pl.DataFrame:
    """Chain as of t: last row per symbol with snap_ts <= t, dropping stale quotes
    and expired options. The per-expiry underlying is taken from the freshest row
    in that expiry, so a stale row does not carry a stale underlying."""
    cur = (
        df.filter((pl.col("snap_ts") <= t_us) & (pl.col("expiration") > t_us))
        .sort("snap_ts")
        .group_by("symbol")
        .last()
        .filter(pl.col("timestamp") >= t_us - int(max_age_s * 1e6))
    )
    und = (
        cur.sort("timestamp")
        .group_by(["coin", "expiration"])
        .agg(pl.col("underlying_price").last().alias("F_und"))
    )
    return cur.join(und, on=["coin", "expiration"]).with_columns(pl.lit(t_us).alias("t_us"))


def clean(chain: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Drop unusable quotes and compute mids and T. Returns (clean, drop_counts)."""
    n0 = chain.height
    counts = {}
    c = chain.filter(pl.col("bid_price").is_not_null() & pl.col("ask_price").is_not_null())
    counts["one_sided_or_empty"] = n0 - c.height
    n1 = c.height
    c = c.filter(pl.col("bid_price") > 0)
    counts["zero_bid"] = n1 - c.height
    n2 = c.height
    c = c.filter(pl.col("bid_price") < pl.col("ask_price"))
    counts["crossed_or_locked"] = n2 - c.height
    c = c.with_columns(
        ((pl.col("expiration") - pl.col("t_us")) / US_PER_YEAR).alias("T"),
        ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_coin"),
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread_coin"),
    )
    counts["kept"] = c.height
    return c, counts


@dataclass
class ForwardFit:
    coin: str
    expiration: int
    F: float
    intercept: float  # a in (C - P)_coin = a + b K; 1.0 under the Deribit zero-rate convention
    n_pairs: int
    resid_std: float
    F_und: float


def fit_forward_one(K, cp_diff, w, n_iter: int = 2) -> tuple[float, float, int, float]:
    """Weighted LS of (C - P) in coin on K: y = a + b K, so F = -a / b.
    One robust pass drops pairs more than 4 MAD from the fit."""
    keep = np.ones(K.shape[0], bool)
    a = b = np.nan
    resid = np.zeros_like(K)
    for _ in range(n_iter):
        if keep.sum() < 3:
            return np.nan, np.nan, int(keep.sum()), np.nan
        X = np.column_stack([np.ones(keep.sum()), K[keep]])
        sw = np.sqrt(w[keep])
        coef, *_ = np.linalg.lstsq(X * sw[:, None], cp_diff[keep] * sw, rcond=None)
        a, b = coef
        resid = cp_diff - (a + b * K)
        mad = np.median(np.abs(resid[keep])) + 1e-12
        keep = np.abs(resid) <= 4 * 1.4826 * mad
    if not (b < 0):
        return np.nan, np.nan, int(keep.sum()), np.nan
    return -a / b, a, int(keep.sum()), float(np.std(resid[keep]))


def fit_forwards(c: pl.DataFrame, max_abs_logm: float = 0.25) -> pl.DataFrame:
    """Forward per (coin, expiry) from put-call parity on strikes with both a clean
    call and a clean put near the money."""
    calls = c.filter(pl.col("is_call")).select(
        "coin", "expiration", "strike_price", pl.col("mid_coin").alias("C"),
        pl.col("spread_coin").alias("sC"), "F_und")
    puts = c.filter(~pl.col("is_call")).select(
        "coin", "expiration", "strike_price", pl.col("mid_coin").alias("P"),
        pl.col("spread_coin").alias("sP"))
    pairs = calls.join(puts, on=["coin", "expiration", "strike_price"])
    pairs = pairs.filter(
        (pl.col("strike_price") / pl.col("F_und")).log().abs() <= max_abs_logm)
    rows = []
    for (coin, exp), g in pairs.group_by(["coin", "expiration"]):
        K = g["strike_price"].to_numpy()
        y = (g["C"] - g["P"]).to_numpy()
        w = 1.0 / (g["sC"] + g["sP"]).to_numpy() ** 2
        F, a, n, sd = fit_forward_one(K, y, w)
        ok = bool(n >= MIN_PARITY_PAIRS and np.isfinite(F) and abs(a - 1.0) <= MAX_INTERCEPT_DEV)
        rows.append(dict(coin=coin, expiration=exp, F=F, intercept=a, n_pairs=n,
                         resid_std=sd, F_und=float(g["F_und"][0]), accepted=ok))
    return pl.DataFrame(rows, schema={"coin": pl.Utf8, "expiration": pl.Int64, "F": pl.Float64,
                                      "intercept": pl.Float64, "n_pairs": pl.Int64,
                                      "resid_std": pl.Float64, "F_und": pl.Float64,
                                      "accepted": pl.Boolean})


def attach_iv(c: pl.DataFrame, fwd: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join forwards (falling back to the Deribit underlying when the parity fit is
    not accepted: fewer than 5 pairs or an intercept more than 0.01 from 1) and
    compute bid, ask and mid implied vols plus Greeks at mid IV.
    Coin premiums are converted to USD via the forward: V_usd = V_coin * F."""
    if fwd is None:
        fwd = fit_forwards(c)
    c = c.join(fwd.select("coin", "expiration", pl.col("F").alias("F_pcp"), "accepted"),
               on=["coin", "expiration"], how="left")
    c = c.with_columns(
        pl.when(pl.col("accepted").fill_null(False))
        .then(pl.col("F_pcp")).otherwise(pl.col("F_und")).alias("F"))
    F = c["F"].to_numpy()
    K = c["strike_price"].to_numpy()
    T = c["T"].to_numpy()
    cp = c["is_call"].to_numpy()
    out = {}
    for name, col in (("iv_bid", "bid_price"), ("iv_ask", "ask_price"), ("iv_mid", "mid_coin")):
        prem_usd = black76.coin_to_usd(c[col].to_numpy(), F)
        sig, st, _ = iv.implied_vol(prem_usd, F, K, T, cp, return_info=True)
        out[name] = sig
        out[name + "_status"] = st
    g = black76.greeks(F, K, T, np.nan_to_num(out["iv_mid"], nan=0.5), cp)
    c = c.with_columns(
        *[pl.Series(k, v) for k, v in out.items()],
        pl.Series("vega_usd", g["vega"]),
        pl.Series("delta_b76", g["delta"]),
        (pl.col("strike_price") / pl.col("F")).log().alias("k"),
        (pl.col("mid_coin") * pl.col("F")).alias("mid_usd"),
        (pl.col("spread_coin") * pl.col("F")).alias("spread_usd"),
    )
    return c


def otm_slice(c: pl.DataFrame) -> pl.DataFrame:
    """Out-of-the-money quotes only (puts below F, calls at or above F) with a valid
    mid IV. This is the set the smile fits use."""
    return c.filter(
        (pl.col("is_call") == (pl.col("k") >= 0))
        & pl.col("iv_mid").is_not_nan()
        & pl.col("iv_mid_status").is_in([iv.OK, iv.BRENT])
    )
