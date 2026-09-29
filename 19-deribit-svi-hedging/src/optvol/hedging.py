"""Discrete delta hedging of short options on a per-minute panel, with P&L attribution.

A panel holds, for one (coin, expiry), a minute grid and per-option mid prices
(USD), own mid IVs, and the expiry's underlying F (the hedge instrument, the
Deribit future or synthetic for that expiry). Positions are one short option
per strike, delta hedged with the future every ``every`` minutes.

Hedge ratio methods (all Black-76 on the same F):
  own_iv            delta at the option's own current mid IV
  sticky_strike     delta at the SVI surface vol for this strike; the smile is
                    assumed to stay fixed in strike when F moves
  sticky_moneyness  sticky strike delta plus vega * d sigma / d F, where the
                    smile is fixed in k = ln(K / F), so d sigma / d F = -sigma'(k) / F

Per-minute attribution of the short, hedged position (Greeks at own IV at the
start of the step, dF and d sigma over the step):
  theta     = -theta dt
  gamma     = -0.5 gamma dF^2
  vega      = -vega d sigma_own
  hedge_err = (h - delta_own) dF        stale or model hedge vs own-IV delta
  residual  = total - the four above    vanna, volga, higher order, mid noise
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from . import black76, iv, svi
from .chain import US_PER_YEAR

MIN_US = 60_000_000
SPIKE_MULT = 10.0  # drop quotes whose half spread exceeds 10x the option's median


@dataclass
class Panel:
    coin: str
    expiration: int
    t_us: np.ndarray  # (n_t,) minute grid
    F: np.ndarray  # (n_t,) hedge instrument price
    K: np.ndarray  # (n_o,)
    is_call: np.ndarray  # (n_o,)
    symbols: list
    mid: np.ndarray  # (n_t, n_o) USD mid, NaN where no clean quote
    half_spread: np.ndarray  # (n_t, n_o) USD
    iv: np.ndarray  # (n_t, n_o) own mid IV, NaN where undefined

    @property
    def T(self):
        return (self.expiration - self.t_us) / US_PER_YEAR


def build_panel(df: pl.DataFrame, coin: str, expiration: int, t0: int, t1: int,
                max_age_s: float = 600.0) -> Panel:
    """As-of panel on the minute grid [t0, t1] from the minute_snap day frame."""
    grid = np.arange(t0, t1 + 1, MIN_US, dtype=np.int64)
    sub = df.filter((pl.col("coin") == coin) & (pl.col("expiration") == expiration)
                    & (pl.col("snap_ts") <= t1)).sort("snap_ts")
    syms = sorted(sub["symbol"].unique().to_list())
    g = pl.DataFrame({"snap_ts": np.repeat(grid, len(syms)),
                      "symbol": np.tile(np.array(syms, dtype=object), grid.size).tolist()})
    g = g.sort("snap_ts")
    j = g.join_asof(sub.select("snap_ts", "symbol", "timestamp", "bid_price", "ask_price",
                               "underlying_price", "strike_price", "is_call"),
                    on="snap_ts", by="symbol", strategy="backward", check_sortedness=False)
    fresh = pl.col("timestamp") >= pl.col("snap_ts") - int(max_age_s * 1e6)
    j = j.with_columns(pl.when(fresh).then(pl.col("bid_price")).alias("bid_price"),
                       pl.when(fresh).then(pl.col("ask_price")).alias("ask_price"),
                       pl.when(fresh).then(pl.col("timestamp")).alias("timestamp"))
    # hedge price: underlying from the freshest row at each minute
    Fs = (j.drop_nulls("timestamp").sort("timestamp").group_by("snap_ts")
          .agg(pl.col("underlying_price").last()).sort("snap_ts"))
    Fs = pl.DataFrame({"snap_ts": grid}).join(Fs, on="snap_ts", how="left")
    F = Fs["underlying_price"].fill_null(strategy="forward").to_numpy()
    j = j.sort(["snap_ts", "symbol"])
    shape = (grid.size, len(syms))
    bid = j["bid_price"].to_numpy().astype(float).reshape(shape)
    ask = j["ask_price"].to_numpy().astype(float).reshape(shape)
    ok = (bid > 0) & (ask > bid)
    # spread spikes: around the 08:00 settlement some makers widen for a minute
    # (one ETH put went from a 0.37 to a 57.85 USD half spread). Such quotes are
    # not crossed, so chain.clean keeps them, but a mid there is meaningless.
    hs_coin = np.where(ok, 0.5 * (ask - bid), np.nan)
    with np.errstate(all="ignore"):
        typical = np.nanmedian(hs_coin, axis=0)
    ok &= ~(hs_coin > SPIKE_MULT * typical[None, :])
    mid = np.where(ok, 0.5 * (bid + ask), np.nan) * F[:, None]
    hs = np.where(ok, 0.5 * (ask - bid), np.nan) * F[:, None]
    meta = sub.group_by("symbol").agg(pl.col("strike_price").first(), pl.col("is_call").first()).sort("symbol")
    K = meta["strike_price"].to_numpy()
    cp = meta["is_call"].to_numpy()
    T = (expiration - grid) / US_PER_YEAR
    Fm = np.broadcast_to(F[:, None], shape)
    sig = iv.implied_vol(mid, Fm, np.broadcast_to(K, shape), np.broadcast_to(T[:, None], shape),
                         np.broadcast_to(cp, shape))
    sig = np.where(sig > 0, sig, np.nan)
    sig = otm_iv(sig, K, cp, F)
    return Panel(coin, expiration, grid, F, K, cp, syms, mid, hs, sig)


def otm_iv(sig: np.ndarray, K: np.ndarray, is_call: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Give each in-the-money option the mid IV of the out-of-the-money option
    at the same strike when that twin has one. Parity makes the two equal in
    theory; in practice the ITM mid carries the spread of a much larger premium
    and its IV is badly conditioned (tiny vega per unit of price noise).
    ITM options whose twin has no clean quote get NaN, which hedge_ratios
    fills from the SVI surface."""
    out = sig.copy()
    idx = {(float(k), bool(c)): j for j, (k, c) in enumerate(zip(K, is_call))}
    itm_call = K[None, :] < F[:, None]  # call ITM where K < F
    for j, (k, c) in enumerate(zip(K, is_call)):
        twin = idx.get((float(k), not bool(c)))
        if twin is None:
            continue
        itm = itm_call[:, j] if c else ~itm_call[:, j]
        # ITM: take the twin's IV, or NaN (later filled from the surface)
        out[itm, j] = sig[itm, twin]
    return out


def fit_surface_path(p: Panel, min_points: int = 5):
    """SVI fit of the OTM mid IVs at every minute (warm started from the
    previous minute). Returns a list of SVIParams (None before the first fit)."""
    out = []
    prev = None
    k_all = np.log(p.K[None, :] / p.F[:, None])
    for i in range(p.t_us.size):
        T = p.T[i]
        k = k_all[i]
        otm = (p.is_call == (k >= 0)) & np.isfinite(p.iv[i])
        if otm.sum() >= min_points and T > 0:
            kk, ss = k[otm], p.iv[i, otm]
            vega = black76.greeks(p.F[i], p.K[otm], T, ss, p.is_call[otm])["vega"]
            wt = vega / np.maximum(p.half_spread[i, otm], 1e-12)
            wt = np.minimum(wt, 10 * np.median(wt))
            o = np.argsort(kk)
            f = svi.fit_svi(kk[o], ss[o] ** 2 * T, wt[o], T, n_starts=2 if prev else 6, x0=prev)
            prev = f.params
        out.append(prev)
    return out


def surface_vol_and_slope(params: svi.SVIParams, k, T):
    w, w1, _ = svi.svi_w_derivs(k, params)
    w = np.maximum(w, 1e-12)
    sig = np.sqrt(w / T)
    dsig_dk = w1 / (2.0 * sig * T)
    return sig, dsig_dk


def hedge_ratios(p: Panel, surf) -> dict[str, np.ndarray]:
    """(n_t, n_o) hedge ratios per method. Missing own IVs are carried forward,
    then filled from the surface."""
    n_t, n_o = p.mid.shape
    iv_own = _ffill(p.iv)
    ss = np.full((n_t, n_o), np.nan)
    slope = np.full((n_t, n_o), np.nan)
    for i in range(n_t):
        if surf[i] is None or p.T[i] <= 0:
            continue
        k = np.log(p.K / p.F[i])
        ss[i], slope[i] = surface_vol_and_slope(surf[i], k, p.T[i])
    iv_own = np.where(np.isfinite(iv_own), iv_own, ss)
    F = p.F[:, None]
    T = np.maximum(p.T, 1e-9)[:, None]
    g_own = black76.greeks(F, p.K, T, iv_own, p.is_call)
    g_ss = black76.greeks(F, p.K, T, ss, p.is_call)
    return dict(
        own_iv=g_own["delta"],
        sticky_strike=g_ss["delta"],
        sticky_moneyness=g_ss["delta"] - g_ss["vega"] * slope / F,
        _own_greeks=g_own, _iv_own=iv_own,
    )


def _ffill(a):
    a = a.copy()
    for i in range(1, a.shape[0]):
        m = np.isnan(a[i])
        a[i, m] = a[i - 1, m]
    return a


@dataclass
class HedgeResult:
    total: np.ndarray  # (n_o,) USD P&L of the short, hedged position
    theta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    hedge_err: np.ndarray
    residual: np.ndarray
    tail: np.ndarray  # whole P&L of the last ``tail_steps`` steps, not attributed
    n_rebalances: int


def simulate(p: Panel, ratios: dict, method: str | None, every: int, i0: int, i1: int,
             final_value: np.ndarray | None = None, final_F: float | None = None,
             tail_steps: int = 0) -> HedgeResult:
    """Short one of each option at mid at minute i0, hold to i1, hedge every
    ``every`` minutes with ``method`` (None = unhedged). If ``final_value`` is
    given the position is closed at that value and F at ``final_F`` (expiry
    settlement); otherwise at the mid at i1. The last ``tail_steps`` steps are
    reported only as a lump ``tail``: in the final minutes before expiry the
    Greeks blow up and a Taylor attribution stops meaning anything."""
    g = ratios["_own_greeks"]
    sig = ratios["_iv_own"]
    V = _ffill(p.mid)[i0:i1 + 1].copy()
    F = p.F[i0:i1 + 1].copy()
    if final_value is not None:
        V[-1] = final_value
        F[-1] = final_F
    n = V.shape[0] - 1
    dV = np.diff(V, axis=0)
    dF = np.diff(F)[:, None]
    dt = np.diff(p.T[i0:i1 + 1])[:, None]  # negative: T shrinks
    dsig = np.diff(sig[i0:i1 + 1], axis=0)
    if final_value is not None:
        dsig[-1] = 0.0  # no vol at expiry; that step's repricing lands in residual
    dsig = np.nan_to_num(dsig)
    th, ga, ve, de = (x[i0:i1] for x in (g["theta"], g["gamma"], g["vega"], g["delta"]))
    if method is None:
        h = np.zeros_like(de)
    else:
        hr = ratios[method][i0:i1]
        idx = (np.arange(n) // every) * every  # last rebalance minute
        h = hr[idx]
    step = -dV + h * dF
    theta = -th * (-dt)  # theta is dV/dt in calendar time; dt here is -dT
    gamma = -0.5 * ga * dF**2
    vega = -ve * dsig
    herr = (h - de) * dF
    resid = step - theta - gamma - vega - herr
    head = slice(0, n - tail_steps)
    tail = step[n - tail_steps:].sum(0) if tail_steps else np.zeros(step.shape[1])
    return HedgeResult(step.sum(0), theta[head].sum(0), gamma[head].sum(0), vega[head].sum(0),
                       herr[head].sum(0), resid[head].sum(0), tail, int(np.ceil(n / every)))


def bootstrap_std(x: np.ndarray, clusters: np.ndarray | None = None, n_boot: int = 2000,
                  seed: int = 0, q=(0.025, 0.975)):
    """Std of x with a (cluster) bootstrap percentile interval."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if clusters is None:
        clusters = np.arange(x.size)
    ids = np.unique(clusters)
    groups = [x[clusters == c] for c in ids]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, ids.size, ids.size)
        stats[b] = np.std(np.concatenate([groups[i] for i in pick]), ddof=1)
    lo, hi = np.quantile(stats, q)
    return float(np.std(x, ddof=1)), float(lo), float(hi)
