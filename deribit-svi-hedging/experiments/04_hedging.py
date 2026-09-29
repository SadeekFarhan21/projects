"""Delta hedging short options with the future: method x rebalance interval.

Only out-of-the-money options (at entry) are traded: an ITM option is its OTM
twin plus a forward, the forward is hedged away, so ITM positions would repeat
the OTM ones while adding their much wider, noisier quotes.

Experiment A (to expiry): at 00:05 UTC sell one of every OTM BTC and ETH option
expiring that day at 08:00 with a clean quote and |delta| in [0.05, 0.95]; hold to
expiry. The option pays off against a settlement proxy, the mean of the
expiry's underlying over 07:30 to 07:59 (Deribit settles on a 30 minute index
TWAP), and the future settles to the same value.

Experiment B (rolling windows): on the next day's expiry, open at the top of
every hour 08:00 to 14:00, sell every OTM option with |delta| in [0.10, 0.90], hold
4 hours and close at mid. Windows stop before the 18:52 to 21:39 data gap.

Every position is hedged every 1, 5 and 30 minutes with three hedge ratios
(own IV, sticky strike SVI, sticky moneyness SVI), plus an unhedged baseline.
P&L is in basis points of the entry forward (1 option on 1 coin).

outputs
  results/04_hedge_pnl.csv          one row per position x method x interval
  results/04_hedge_summary.csv      std of P&L with 95% bootstrap intervals, attribution stds
  results/04_hedge_context.json     realized vs implied vol, counts, settlement proxy
  results/04_hedge_std.png, results/04_attribution.png

usage: uv run python experiments/04_hedging.py [day]
"""

from __future__ import annotations

import json
import sys
import time

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from optvol import data, hedging as H

DAY = sys.argv[1] if len(sys.argv) > 1 else data.DEFAULT_DAY
EVERY = (1, 5, 30)
METHODS = ("own_iv", "sticky_strike", "sticky_moneyness")
COMPONENTS = ("theta", "gamma", "vega", "hedge_err", "residual", "tail")
TAIL_A = 30  # steps from 07:30 to settlement, reported as one lump


def rows_from(res, p, sel, method, every, exp_name, window, F0):
    out = []
    for j, o in enumerate(sel):
        r = dict(experiment=exp_name, coin=p.coin, symbol=p.symbols[o], window=window,
                 method=method or "unhedged", every=every if method else 0,
                 is_call=bool(p.is_call[o]), K=float(p.K[o]))
        for name in ("total",) + COMPONENTS:
            r[name + "_bp"] = float(getattr(res, name)[j] / F0 * 1e4)
        out.append(r)
    return out


def run_positions(p, ratios, i0, i1, sel, exp_name, window, final=None, tail=0):
    rows = []
    sub = H.Panel(p.coin, p.expiration, p.t_us, p.F, p.K[sel], p.is_call[sel],
                  [p.symbols[o] for o in sel], p.mid[:, sel], p.half_spread[:, sel], p.iv[:, sel])
    r_sub = {k: (v[:, sel] if isinstance(v, np.ndarray) else {g: a[:, sel] for g, a in v.items()})
             for k, v in ratios.items()}
    fv, fF = (None, None) if final is None else final
    if fv is not None:
        fv = fv[sel]
    F0 = p.F[i0]
    idx = np.arange(len(sel))
    res = H.simulate(sub, r_sub, None, 1, i0, i1, fv, fF, tail)
    rows += rows_from(res, sub, idx, None, 0, exp_name, window, F0)
    for m in METHODS:
        for e in EVERY:
            res = H.simulate(sub, r_sub, m, e, i0, i1, fv, fF, tail)
            rows += rows_from(res, sub, idx, m, e, exp_name, window, F0)
    return rows


def realized_vol(F):
    r = np.diff(np.log(F))
    return float(np.sqrt(np.sum(r**2) * 365 * 1440 / r.size))


def main():
    t_start = time.time()
    df = data.load_day(DAY)
    exps = sorted(df["expiration"].unique().to_list())
    day0 = data.ts_us(DAY, 0)
    exp_a = [e for e in exps if day0 < e <= day0 + 86400 * 10**6][0]
    exp_b = [e for e in exps if e > exp_a][0]
    rows, ctx = [], dict(day=DAY, experiment_A={}, experiment_B={})

    # ---------------- A: to expiry
    for coin in ("BTC", "ETH"):
        t0, t1 = data.ts_us(DAY, 0, 5), exp_a - H.MIN_US
        p = H.build_panel(df, coin, exp_a, t0, t1)
        surf = H.fit_surface_path(p)
        ratios = H.hedge_ratios(p, surf)
        i0, i1 = 0, p.t_us.size - 1
        tw = p.t_us >= exp_a - 30 * H.MIN_US
        S = float(np.mean(p.F[tw]))
        payoff = np.where(p.is_call, np.maximum(S - p.K, 0), np.maximum(p.K - S, 0))
        d0 = ratios["own_iv"][i0]
        otm0 = p.is_call == (p.K >= p.F[i0])
        sel = np.flatnonzero(otm0 & np.isfinite(p.mid[i0]) & (np.abs(d0) >= 0.05) & (np.abs(d0) <= 0.95))
        # the settlement step: hold hedges from 07:59 into settlement
        panel_ext = extend_to_settlement(p, ratios, exp_a)
        pe, re_ = panel_ext
        rows += run_positions(pe, re_, i0, pe.t_us.size - 1, sel, "A_to_expiry", 0,
                              final=(payoff, S), tail=TAIL_A)
        atm = np.nanmedian(p.iv[i0][np.abs(np.log(p.K / p.F[i0])) < 0.02])
        ctx["experiment_A"][coin] = dict(
            n_options=int(sel.size), F_entry=float(p.F[i0]), settlement_proxy=S,
            F_last=float(p.F[-1]), realized_vol=realized_vol(p.F), atm_iv_entry=float(atm),
            minutes=int(p.t_us.size), svi_fits=int(sum(s is not None for s in surf)))
        print(f"A {coin}: {sel.size} options, {time.time() - t_start:.0f}s", flush=True)

    # ---------------- B: rolling 4h windows on the next expiry
    for coin in ("BTC", "ETH"):
        t0, t1 = data.ts_us(DAY, 8), data.ts_us(DAY, 18)
        p = H.build_panel(df, coin, exp_b, t0, t1)
        surf = H.fit_surface_path(p)
        ratios = H.hedge_ratios(p, surf)
        wins = []
        for w, hour in enumerate(range(8, 15)):
            i0 = (hour - 8) * 60
            i1 = i0 + 240
            d0 = ratios["own_iv"][i0]
            otm0 = p.is_call == (p.K >= p.F[i0])
            sel = np.flatnonzero(otm0 & np.isfinite(p.mid[i0]) & np.isfinite(p.mid[i1])
                                 & (np.abs(d0) >= 0.10) & (np.abs(d0) <= 0.90))
            rows += run_positions(p, ratios, i0, i1, sel, "B_rolling_4h", w)
            atm = np.nanmedian(p.iv[i0][np.abs(np.log(p.K / p.F[i0])) < 0.03])
            wins.append(dict(entry_hour=hour, n_options=int(sel.size),
                             realized_vol=realized_vol(p.F[i0:i1 + 1]), atm_iv_entry=float(atm)))
        ctx["experiment_B"][coin] = wins
        print(f"B {coin}: {time.time() - t_start:.0f}s", flush=True)

    pnl = pl.DataFrame(rows)
    pnl.write_csv(data.RESULTS / "04_hedge_pnl.csv")

    # ---------------- summary with bootstrap intervals
    summ = []
    for (exp_name, method, every), g in pnl.group_by(["experiment", "method", "every"]):
        x = g["total_bp"].to_numpy()
        if exp_name.startswith("B"):
            clusters = (g["coin"] + "_" + g["window"].cast(pl.Utf8)).to_numpy()
        else:
            clusters = None  # one path per coin: resample options
        sd, lo, hi = H.bootstrap_std(x, clusters)
        r = dict(experiment=exp_name, method=method, every=every, n=g.height,
                 mean_bp=float(x.mean()), std_bp=sd, std_lo=lo, std_hi=hi)
        for c in COMPONENTS:
            r[f"mean_{c}_bp"] = float(g[f"{c}_bp"].mean())
            r[f"std_{c}_bp"] = float(g[f"{c}_bp"].std())
        summ.append(r)
    summ = pl.DataFrame(summ).sort(["experiment", "method", "every"])
    summ.write_csv(data.RESULTS / "04_hedge_summary.csv")
    ctx["wall_s"] = time.time() - t_start
    ctx["n_positions"] = {k: int(v) for k, v in
                          pnl.filter(pl.col("method") == "unhedged").group_by("experiment").len().iter_rows()}
    (data.RESULTS / "04_hedge_context.json").write_text(json.dumps(ctx, indent=1))
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=200):
        print(summ.select("experiment", "method", "every", "n", "mean_bp", "std_bp", "std_lo", "std_hi",
                          "std_gamma_bp", "std_vega_bp", "std_hedge_err_bp", "std_residual_bp"))
    print(json.dumps(ctx, indent=1))
    plots(summ)


def extend_to_settlement(p, ratios, exp):
    """Append the settlement instant as a final grid point so the last step runs
    from 07:59 to 08:00. Greeks there are never used (no step starts at it)."""
    t = np.append(p.t_us, exp)
    pad = lambda a: np.vstack([a, a[-1:]])  # noqa: E731
    pe = H.Panel(p.coin, p.expiration, t, np.append(p.F, p.F[-1]), p.K, p.is_call, p.symbols,
                 pad(p.mid), pad(p.half_spread), pad(p.iv))
    re_ = {k: (pad(v) if isinstance(v, np.ndarray) else {g: pad(a) for g, a in v.items()})
           for k, v in ratios.items()}
    return pe, re_


def plots(summ):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = {"A_to_expiry": "A, same day options held to expiry",
             "B_rolling_4h": "B, next day expiry, 4 hour windows"}
    colors = {"own_iv": "C0", "sticky_strike": "C1", "sticky_moneyness": "C2"}
    for ax, (exp_name, title) in zip(axes, names.items()):
        s = summ.filter(pl.col("experiment") == exp_name)
        for j, m in enumerate(METHODS):
            g = s.filter(pl.col("method") == m).sort("every")
            x = np.arange(len(EVERY)) + (j - 1) * 0.22
            y = g["std_bp"].to_numpy()
            ax.errorbar(x, y, yerr=[y - g["std_lo"].to_numpy(), g["std_hi"].to_numpy() - y],
                        fmt="o", color=colors[m], capsize=3, label=m.replace("_", " "))
        u = s.filter(pl.col("method") == "unhedged")
        ax.axhline(u["std_bp"][0], color="0.5", ls="--", lw=1, label="unhedged")
        ax.set_xticks(range(len(EVERY)), [f"{e} min" for e in EVERY])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel("rebalance interval")
        ax.set_ylabel("std of P&L (bp of forward)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Delta hedged short options on {DAY} with 95% bootstrap intervals")
    fig.tight_layout()
    fig.savefig(data.RESULTS / "04_hedge_std.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (exp_name, title) in zip(axes, names.items()):
        s = summ.filter((pl.col("experiment") == exp_name) & (pl.col("method") == "own_iv")).sort("every")
        x = np.arange(len(EVERY))
        for j, c in enumerate(COMPONENTS):
            ax.bar(x + (j - 2.5) * 0.14, s[f"std_{c}_bp"], 0.14, label=c.replace("_", " "))
        ax.plot(x, s["std_bp"], "k_", ms=30, mew=2, label="total")
        ax.set_xticks(x, [f"{e} min" for e in EVERY])
        ax.set_xlabel("rebalance interval")
        ax.set_ylabel("std across positions (bp of forward)")
        ax.set_title(title + ", own IV delta", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Where hedged P&L dispersion comes from")
    fig.tight_layout()
    fig.savefig(data.RESULTS / "04_attribution.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
