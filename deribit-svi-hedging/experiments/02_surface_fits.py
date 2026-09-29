"""Raw SVI per expiry and global SSVI on hourly snapshots of one tardis day.

For every hour 00:00 to 23:00 UTC and each coin: clean the chain, fit forwards,
compute IVs, fit SVI on each expiry with at least 5 OTM quotes, run the
butterfly (dense g grid) and calendar checks, then fit SSVI across expiries.

outputs
  results/02_svi_slices.csv       one row per (hour, coin, expiry)
  results/02_forwards.csv         parity forwards vs Deribit underlying
  results/02_drop_counts.csv      cleaning drop counts per snapshot
  results/02_surface_summary.json headline aggregates
  results/02_smiles_btc.png, 02_smiles_eth.png, 02_svi_vs_ssvi_rmse.png

usage: uv run python experiments/02_surface_fits.py [day]
"""

from __future__ import annotations

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from optvol import data, svi
from optvol.surface import fit_snapshot

DAY = sys.argv[1] if len(sys.argv) > 1 else data.DEFAULT_DAY
PLOT_HOUR = 12


def plot_smiles(coin, arrays, fits, ssvi_p, rows, path):
    idx = [i for i in range(len(rows)) if rows[i]["T_days"] > 0.5]
    pick = [idx[j] for j in np.linspace(0, len(idx) - 1, 4).astype(int)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, i in zip(axes.ravel(), pick):
        k, w, wt, T, ivm = arrays[i]
        g = rows[i]["_g"]
        lo, hi = g["iv_bid"].to_numpy(), g["iv_ask"].to_numpy()
        ax.vlines(k, lo * 100, hi * 100, color="0.65", lw=1.2, label="bid to ask")
        ax.plot(k, ivm * 100, ".", color="0.2", ms=4, label="mid")
        kk = np.linspace(k.min(), k.max(), 300)
        ax.plot(kk, svi.svi_vol(kk, fits[i].params, T) * 100, color="C0", lw=1.6, label="SVI")
        if ssvi_p is not None:
            sp = svi.ssvi_slice_params(ssvi_p, i)
            ax.plot(kk, svi.svi_vol(kk, sp, T) * 100, color="C3", lw=1.2, ls="--", label="SSVI")
        ax.set_title(f"{coin} expiry in {rows[i]['T_days']:.1f} days", fontsize=10)
        ax.set_xlabel("log moneyness ln(K/F)")
        ax.set_ylabel("implied vol (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{coin} smiles on {DAY} at {PLOT_HOUR:02d}00 UTC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    data.RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    all_rows, fwd_rows, drop_rows = [], [], []
    for hour in range(24):
        t_us = data.ts_us(DAY, hour)
        c, fwd, counts = data.clean_snapshot(DAY, t_us)
        if c.height == 0:  # data gap in the sample: every quote is stale
            print(f"hour {hour:02d} empty (no quote younger than 10 min)")
            drop_rows.append(dict(hour=hour, **counts))
            continue
        drop_rows.append(dict(hour=hour, **counts))
        fwd_rows += [dict(hour=hour, **r) for r in fwd.to_dicts()]
        for coin in ("BTC", "ETH"):
            rows, fits, arrays, ssvi_p = fit_snapshot(c, coin)
            if hour == PLOT_HOUR:
                from optvol.chain import otm_slice
                o = otm_slice(c).filter(pl.col("coin") == coin)
                for r in rows:
                    r["_g"] = o.filter(pl.col("expiration") == r["expiration"]).sort("k")
                plot_smiles(coin, arrays, fits, ssvi_p, rows,
                            data.RESULTS / f"02_smiles_{coin.lower()}.png")
                for r in rows:
                    r.pop("_g")
            all_rows += [dict(hour=hour, **r) for r in rows]
        print(f"hour {hour:02d} done, {len(all_rows)} slices, {time.time() - t0:.0f}s", flush=True)
    df = pl.DataFrame(all_rows)
    df.write_csv(data.RESULTS / "02_svi_slices.csv")
    pl.DataFrame(fwd_rows).write_csv(data.RESULTS / "02_forwards.csv")
    pl.DataFrame(drop_rows).write_csv(data.RESULTS / "02_drop_counts.csv")

    fw_all = pl.DataFrame(fwd_rows)
    fw = fw_all.filter(pl.col("accepted"))
    summ = dict(
        day=DAY, n_slices=df.height, n_snapshots=int(df['hour'].n_unique()),
        empty_hours=[r['hour'] for r in drop_rows if r['kept'] == 0],
        svi_rmse_vp_median=float(df["svi_rmse_vp"].median()),
        svi_rmse_vp_p90=float(df["svi_rmse_vp"].quantile(0.9)),
        half_spread_vp_median=float(df["half_spread_vp"].median()),
        share_rmse_below_half_spread=float((df["svi_rmse_vp"] < df["half_spread_vp"]).mean()),
        svi_inside_spread_mean=float(df["inside_spread"].mean()),
        share_butterfly_ok=float(df["butterfly_ok"].mean()),
        share_calendar_ok=float(df["calendar_ok"].mean()),
        share_both_ok=float(df["both_ok"].mean()),
        ssvi_rmse_vp_median=float(df["ssvi_rmse_vp"].median()),
        ssvi_rmse_vp_p90=float(df["ssvi_rmse_vp"].quantile(0.9)),
        ssvi_share_butterfly_ok=float(df["ssvi_butterfly_ok"].mean()),
        by_coin={coin: dict(
            svi_rmse_vp_median=float(g["svi_rmse_vp"].median()),
            ssvi_rmse_vp_median=float(g["ssvi_rmse_vp"].median()),
            half_spread_vp_median=float(g["half_spread_vp"].median()),
            share_both_ok=float(g["both_ok"].mean()), n=g.height)
            for (coin,), g in df.group_by("coin")},
        forward_minus_underlying_bp_median_abs=float(
            ((fw["F"] / fw["F_und"] - 1).abs() * 1e4).median()),
        forward_minus_underlying_bp_max_abs=float(
            ((fw["F"] / fw["F_und"] - 1).abs() * 1e4).max()),
        parity_intercept_median=float(fw["intercept"].median()),
        forward_share_accepted=float(fw_all["accepted"].mean()),
        forward_n_expiry_snapshots=fw_all.height,
        drop_totals={k: int(sum(r[k] for r in drop_rows)) for k in drop_rows[0] if k != "hour"},
        wall_s=time.time() - t0,
    )
    (data.RESULTS / "02_surface_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    # SVI vs SSVI error by expiry bucket
    d = df.with_columns(pl.col("T_days").cut([1, 7, 30, 90], labels=["<1d", "1-7d", "7-30d", "30-90d", ">90d"]).alias("bucket"))
    agg = d.group_by("bucket").agg(pl.col("svi_rmse_vp").median(), pl.col("ssvi_rmse_vp").median(),
                                   pl.col("half_spread_vp").median()).sort("bucket")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(agg.height)
    ax.bar(x - 0.27, agg["svi_rmse_vp"], 0.27, label="SVI per expiry", color="C0")
    ax.bar(x, agg["ssvi_rmse_vp"], 0.27, label="SSVI global", color="C3")
    ax.bar(x + 0.27, agg["half_spread_vp"], 0.27, label="median half spread", color="0.6")
    ax.set_xticks(x, [str(s) for s in agg["bucket"]])
    ax.set_xlabel("time to expiry")
    ax.set_ylabel("vol points")
    ax.set_title(f"Fit RMSE vs half spread, {DAY}, hourly snapshots")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(data.RESULTS / "02_svi_vs_ssvi_rmse.png", dpi=130)


if __name__ == "__main__":
    main()
