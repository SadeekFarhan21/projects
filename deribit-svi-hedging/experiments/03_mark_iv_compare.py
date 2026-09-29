"""Own implied vols vs Deribit mark IV, by moneyness bucket, with a decomposition.

For every clean quote with a valid mid IV in the hourly snapshots:
    diff = iv_mid (own mid, own parity forward) - mark_iv / 100
is split exactly into three parts
    price part    = iv_mid - iv(mark_price, own F)       mid vs Deribit mark price
    forward part  = iv(mark, own F) - iv(mark, Deribit underlying)
    residual      = iv(mark, Deribit underlying) - mark_iv / 100
The residual is what is left once both inputs match Deribit's: rounding of
mark_price to 4 decimals of coin plus any model difference.

outputs
  results/03_mark_iv_buckets.csv    stats per standardized moneyness bucket
  results/03_mark_iv_tenor.csv      stats per time to expiry bucket
  results/03_mark_iv_outliers.csv   the 20 largest absolute differences
  results/03_mark_iv_summary.json
  results/03_mark_iv_diff.png

usage: uv run python experiments/03_mark_iv_compare.py [day]
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from optvol import black76, data, iv

DAY = sys.argv[1] if len(sys.argv) > 1 else data.DEFAULT_DAY
Z_EDGES = [-2.0, -1.0, -0.25, 0.25, 1.0, 2.0]
Z_LABELS = ["z<-2", "-2..-1", "-1..-0.25", "atm", "0.25..1", "1..2", "z>2"]


def iv_of_mark(c: pl.DataFrame, fcol: str) -> np.ndarray:
    F = c[fcol].to_numpy()
    usd = black76.coin_to_usd(c["mark_price"].to_numpy(), F)
    return iv.implied_vol(usd, F, c["strike_price"].to_numpy(), c["T"].to_numpy(),
                          c["is_call"].to_numpy())


def stats(g: pl.DataFrame) -> dict:
    d = g["diff_vp"]
    return dict(n=g.height, median_diff_vp=d.median(), mean_diff_vp=d.mean(),
                mad_vp=(d - d.median()).abs().median(), p95_abs_vp=d.abs().quantile(0.95),
                share_within_half_spread=float((d.abs() <= g["half_spread_vp"]).mean()),
                n_no_bid_iv=int(g["half_spread_vp"].is_null().sum()),
                share_mid_equals_mark=float(g["mid_equals_mark"].mean()),
                median_price_part_vp=g["price_part_vp"].median(),
                median_forward_part_vp=g["forward_part_vp"].median(),
                median_residual_vp=g["residual_vp"].median(),
                median_half_spread_vp=g["half_spread_vp"].median())


def main():
    frames = []
    for hour in range(24):
        c, _, _ = data.clean_snapshot(DAY, data.ts_us(DAY, hour))
        if c.height == 0:
            continue
        c = c.filter(pl.col("iv_mid_status").is_in([iv.OK, iv.BRENT]) & pl.col("mark_iv").is_not_null())
        c = c.with_columns(pl.Series("iv_mark_ownF", iv_of_mark(c, "F")),
                           pl.Series("iv_mark_undF", iv_of_mark(c, "underlying_price")),
                           pl.lit(hour).alias("hour"))
        frames.append(c)
    c = pl.concat(frames, how="diagonal_relaxed")
    c = c.with_columns(
        ((pl.col("iv_mid") - pl.col("mark_iv") / 100) * 100).alias("diff_vp"),
        ((pl.col("iv_mid") - pl.col("iv_mark_ownF")) * 100).alias("price_part_vp"),
        ((pl.col("iv_mark_ownF") - pl.col("iv_mark_undF")) * 100).alias("forward_part_vp"),
        ((pl.col("iv_mark_undF") - pl.col("mark_iv") / 100) * 100).alias("residual_vp"),
        ((pl.col("iv_ask") - pl.col("iv_bid")) * 50).fill_nan(None).alias("half_spread_vp"),
        (pl.col("mid_coin") - pl.when(pl.col("is_call")).then(1 - pl.col("strike_price") / pl.col("F"))
         .otherwise(pl.col("strike_price") / pl.col("F") - 1).clip(lower_bound=0)).alias("time_value_coin"),
        (pl.col("mid_coin") == pl.col("mark_price")).alias("mid_equals_mark"),
        (pl.col("k") / (pl.col("iv_mid") * pl.col("T").sqrt())).alias("z"),
        (pl.col("is_call") == (pl.col("k") >= 0)).alias("otm"),
        (pl.col("T") * 365).alias("T_days"),
    ).filter(pl.col("iv_mark_ownF").is_not_nan() & pl.col("iv_mark_undF").is_not_nan())
    c = c.with_columns(pl.col("z").cut(Z_EDGES, labels=Z_LABELS).alias("z_bucket"),
                       pl.col("T_days").cut([1, 7, 30, 90], labels=["<1d", "1-7d", "7-30d", "30-90d", ">90d"]).alias("t_bucket"))

    rows = [dict(z_bucket=b, **stats(g)) for b in Z_LABELS
            if (g := c.filter(pl.col("z_bucket") == b)).height]
    pl.DataFrame(rows).write_csv(data.RESULTS / "03_mark_iv_buckets.csv")
    trows = [dict(t_bucket=b, **stats(g)) for b in ["<1d", "1-7d", "7-30d", "30-90d", ">90d"]
             if (g := c.filter(pl.col("t_bucket") == b)).height]
    pl.DataFrame(trows).write_csv(data.RESULTS / "03_mark_iv_tenor.csv")
    out = c.sort(pl.col("diff_vp").abs(), descending=True).head(20).select(
        "hour", "symbol", "T_days", "k", "z", "otm", "iv_mid", "mark_iv", "diff_vp",
        "price_part_vp", "forward_part_vp", "residual_vp", "half_spread_vp",
        "time_value_coin", "vega_usd", "bid_price", "ask_price", "mark_price")
    out.write_csv(data.RESULTS / "03_mark_iv_outliers.csv")
    otm = c.filter(pl.col("otm"))
    summ = dict(day=DAY, all=stats(c), otm_only=stats(otm),
                share_abs_diff_below_0p5vp=float((c["diff_vp"].abs() < 0.5).mean()),
                corr_diff_vs_price_part=float(np.corrcoef(c["diff_vp"], c["price_part_vp"])[0, 1]))
    (data.RESULTS / "03_mark_iv_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    print(pl.DataFrame(rows))
    print(out.head(10))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    parts = [("price_part_vp", "mid vs mark price", "C0"),
             ("forward_part_vp", "forward choice", "C2"),
             ("residual_vp", "residual", "0.5")]
    x = np.arange(len(rows))
    for j, (col, lab, colr) in enumerate(parts):
        ax.bar(x + (j - 1) * 0.26, [r[f"median_{col.replace('_vp', '')}_vp"] for r in rows], 0.26,
               label=lab, color=colr)
    ax.plot(x, [r["median_diff_vp"] for r in rows], "k_", ms=22, mew=2, label="total median")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [r["z_bucket"] for r in rows])
    ax.set_xlabel("standardized moneyness z = ln(K/F) / (vol sqrt T)")
    ax.set_ylabel("own mid IV minus mark IV (vol points)")
    ax.set_title(f"Own IV vs Deribit mark IV by moneyness, {DAY}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(data.RESULTS / "03_mark_iv_diff.png", dpi=130)


if __name__ == "__main__":
    main()
