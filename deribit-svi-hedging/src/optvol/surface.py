"""Fit a whole snapshot: raw SVI per expiry, SSVI across expiries, arbitrage checks."""

from __future__ import annotations

import numpy as np
import polars as pl

from . import svi
from .chain import otm_slice

MIN_POINTS = 5
WEIGHT_CAP = 10.0  # cap vega/spread weights at 10x the slice median
GRID_POINTS = 2001


def slice_arrays(g: pl.DataFrame):
    g = g.sort("k")
    k = g["k"].to_numpy()
    T = float(g["T"][0])
    ivm = g["iv_mid"].to_numpy()
    w = ivm**2 * T
    wt = (g["vega_usd"] / g["spread_usd"]).to_numpy()
    wt = np.minimum(wt, WEIGHT_CAP * np.median(wt))
    hs = (g["iv_ask"] - g["iv_bid"]).to_numpy() / 2
    return k, T, ivm, w, wt, hs, g


def butterfly_grid(k):
    r = max(k.max() - k.min(), 1e-3)
    return np.linspace(k.min() - 0.5 * r, k.max() + 0.5 * r, GRID_POINTS)


def fit_snapshot(c: pl.DataFrame, coin: str, with_ssvi: bool = True):
    """c: cleaned chain with IVs (chain.attach_iv). Returns (slice rows, fits)."""
    o = otm_slice(c).filter(pl.col("coin") == coin)
    groups = sorted(o.group_by("expiration"), key=lambda kv: kv[0][0])
    rows, fits, arrays = [], [], []
    for (exp,), g in groups:
        if g.height < MIN_POINTS:
            continue
        k, T, ivm, w, wt, hs, gs = slice_arrays(g)
        f = svi.fit_svi(k, w, wt, T)
        mv = svi.svi_vol(k, f.params, T)
        err = mv - ivm
        ok_b, gmin = svi.butterfly_ok(f.params, butterfly_grid(k))
        iv_bid = gs["iv_bid"].to_numpy()
        iv_ask = gs["iv_ask"].to_numpy()
        inside = np.where(np.isfinite(iv_bid), (mv >= iv_bid) & (mv <= iv_ask), mv <= iv_ask)
        rows.append(dict(
            coin=coin, expiration=int(exp), T_days=T * 365, n=int(k.size),
            k_min=float(k.min()), k_max=float(k.max()),
            svi_rmse_vp=float(np.sqrt(np.mean(err**2)) * 100),
            svi_wrmse_vp=float(np.sqrt(np.sum(wt * err**2) / np.sum(wt)) * 100),
            half_spread_vp=float(np.nanmedian(hs) * 100),
            inside_spread=float(np.mean(inside)),
            butterfly_ok=bool(ok_b), g_min=gmin,
            a=f.params.a, b=f.params.b, rho=f.params.rho, m=f.params.m, s=f.params.s,
        ))
        fits.append(f)
        arrays.append((k, w, wt, T, ivm))
    # calendar: consecutive slices on the overlap of their data ranges
    n = len(rows)
    cal_ok = [True] * n
    cal_min = [np.nan] * n
    for j in range(n - 1):
        lo = max(rows[j]["k_min"], rows[j + 1]["k_min"])
        hi = min(rows[j]["k_max"], rows[j + 1]["k_max"])
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, GRID_POINTS)
        ok, dmin = svi.calendar_ok(svi.svi_w(grid, fits[j].params), svi.svi_w(grid, fits[j + 1].params))
        for i in (j, j + 1):
            cal_ok[i] = cal_ok[i] and ok
            cal_min[i] = dmin if np.isnan(cal_min[i]) else min(cal_min[i], dmin)
    for i in range(n):
        rows[i]["calendar_ok"] = cal_ok[i]
        rows[i]["calendar_min_dw"] = cal_min[i]
        rows[i]["both_ok"] = rows[i]["butterfly_ok"] and cal_ok[i]
    ssvi_params = None
    if with_ssvi and n >= 2:
        ssvi_params, _ = svi.fit_ssvi([(k, w, wt) for k, w, wt, _, _ in arrays])
        for j, (k, w, wt, T, ivm) in enumerate(arrays):
            sp = svi.ssvi_slice_params(ssvi_params, j)
            mv = svi.svi_vol(k, sp, T)
            rows[j]["ssvi_rmse_vp"] = float(np.sqrt(np.mean((mv - ivm) ** 2)) * 100)
            rows[j]["ssvi_butterfly_ok"] = bool(svi.butterfly_ok(sp, butterfly_grid(k))[0])
            rows[j]["ssvi_theta"] = float(ssvi_params.theta[j])
        for j in range(n):
            rows[j]["ssvi_rho"] = ssvi_params.rho
            rows[j]["ssvi_eta"] = ssvi_params.eta
            rows[j]["ssvi_gamma"] = ssvi_params.gamma
    return rows, fits, arrays, ssvi_params
