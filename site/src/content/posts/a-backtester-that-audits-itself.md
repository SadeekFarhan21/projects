---
title: "a backtester that checks itself for look-ahead bias"
description: "A crypto research platform with a point-in-time store, purged cross-validation and a look-ahead audit that flags leaky features before they reach a backtest."
slug: a-backtester-that-audits-itself
tags:
  - quant
  - backtesting
  - research
draft: true
category: projects
---

I built qrp, a small research platform for daily crypto bars whose job is to produce numbers I would actually believe. It ingests 300 Binance USDT spot pairs from 2022-01-01 to 2026-08-31, stores them in an append-only parquet store with a point-in-time read path, computes features from a declarative spec, audits those features for look-ahead before every run, fits a model with purged walk-forward cross-validation, and backtests the resulting target weights with drift, fees, slippage and optional square-root impact. Every run lands in a local tracker you can diff from the command line.

The honest baseline, a cross-sectional ridge on seven ranked price and volume features, earns an out of sample **Sharpe of 0.49** over 1,338 days at 15 bps one way. Add a single feature that peeks five days ahead and switch off the audit's fail switch, and the same pipeline reports a **Sharpe of 19.40** with a max drawdown of 2.5%. The point of the project is the machinery that makes the second number impossible to produce by accident, and the tests that break when any piece of that machinery is wrong.

The platform is about 1,650 lines of Python with 73 tests. The compiled backtest kernel runs about 80 million asset-days per second on one thread of an Apple M4 Pro that was heavily loaded by other jobs, so every throughput number here is a lower bound.

*Reading note.* The argument is that most great-looking backtests are wrong in one of five boring ways, and that each one can be turned into a written rule with a test attached. The problems section is where the rules earned their keep. Skip to [results](#results) if you only want the numbers.

## table of contents

- [what i wanted to build](#what-i-wanted-to-build)
- [theory](#theory)
- [architecture](#architecture)
- [implementation](#implementation)
- [problems](#problems)
- [experiments](#experiments)
- [results](#results)
- [what i would change](#what-i-would-change)
- [reproducibility](#reproducibility)

## what i wanted to build

A backtest that looks great is usually broken, and the ways it breaks are few and dull. A feature quietly uses tomorrow's data. The model trains on labels whose returns overlap the test period. The simulator trades at the same close the signal just observed. Turnover is computed as the change in target weights, ignoring that weights drift with prices, so costs come out too low. Or the data itself contains a ticker that died and came back as a different asset, and the backtester sees a 177,000x return across the gap.

None of these need a clever strategy to show up. They show up in the plumbing. So the goal for v0 was not a strategy at all. It was the smallest pipeline in which each of those five failure modes has a written contract and a test that fails when the contract is broken, fast enough that later projects can sweep thousands of variants. I left out minute bars, and US equities for a mundane reason covered in [problems](#problems).

## theory

Five ideas carry the whole design. Each one is small, and each one is violated by default in a naive research loop.

### one timing convention, used everywhere

Row $t$ means "after the close of day $t$". Every stage is written against that single sentence.

1. A feature at row $t$ may use bars at rows $\le t$ only.
2. A target weight decided at row $t$ executes at the close of row $t + d$, where $d$ is the execution delay (default 1).
3. The training label for row $t$ is the return from close $t + d$ to close $t + d + h$, for horizon $h$. Its information interval is $[t,\ t + d + h]$.
4. The universe at row $t$ (top 100 by trailing dollar volume, at least 60 bars of history, has a bar today) is computed from rows $\le t$.

Delay 0 is allowed but optimistic, since it trades at the very close the signal just read.

### look-ahead as an empirical property

A feature is causal if its value at row $t$ is a function of rows $\le t$ only. You can try to enforce that by reading code or by labelling operators as safe, but both rely on someone noticing. I test it instead, by changing the future and checking that the past does not move.

Pick a cut row $c$. The **truncation probe** recomputes every feature on the panel cut to rows $0..c$ and requires rows $\le c$ to be bit-for-bit unchanged. A z-score normalized with the full-sample mean and standard deviation fails this at every row, because shortening the sample changes the mean. The **perturbation probe** multiplies every field after $c$ by random positive noise, recomputes, and requires the same thing. A `shift(-1)` fails it at exactly row $c$. A centered moving average fails both probes in the last half window before the cut.

Neither probe needs to know what an operator does.

### labels overlap, so splits must be purged

A 5 day forward return label at row $t$ contains the returns of days $t+2$ through $t+6$ (with delay 1). If a training row sits just before a test block, its label contains returns that are also inside the test block, and the model is partly graded on what it trained on. Purging, from chapter 7 of Lopez de Prado's *Advances in Financial Machine Learning*, drops every training row whose label interval overlaps a test label interval. Embargo drops a further band of rows after each test block, because features computed right after the block are serially correlated with the test labels.

In a walk-forward split, training data is always before the test block, so only purging matters. In k-fold, training data sits on both sides, so both matter.

### a backtest is a recursion, not a matrix product

The common vectorized backtest computes turnover as $\sum_i |W_{t,i} - W_{t-1,i}|$. That is wrong whenever prices move. Between rebalances, weights drift with returns, so the trade needed at day $t$ is the difference between today's target and the drifted weights, not between today's and yesterday's targets. And because costs reduce equity, which changes tomorrow's drifted weights, day $t$ depends on day $t-1$ through the cost term. The loop over time cannot be vectorized away.

Per strategy and day, with equity in units of starting capital,

$$
E^-_t = A_{t-1} + \sum_i D_{t-1,i}\, r_{t,i}, \qquad
p_{t,i} = \frac{D_{t-1,i}\,(1 + r_{t,i})}{E^-_t}
$$

$$
h_t = W_{t-d}, \qquad
k_t = \sum_i |h_{t,i} - p_{t,i}|\, c_{t,i}, \qquad
A_t = E^-_t\,(1 - k_t), \qquad
D_t = h_t\, E^-_t
$$

$E^-$ is pre-trade equity after marking to market, $p$ the drifted weights, $h$ the executed targets, $c$ the per-unit cost, $A$ post-trade equity and $D$ dollar positions. Where an asset has no bar, $h = p$, so it is held and not traded.

The convention that matters is that targets are fractions of *pre-trade* equity and the cost is paid from cash. Under that convention an explicit ledger of share counts and a cash balance produces exactly the same numbers as the weight recursion, to floating point. That is what lets a slow, obviously correct ledger serve as the oracle for the fast kernels. It also guarantees three identities that the tests check directly.

$$
A_t = A_{t-1} + \text{gross}_t - \text{cost}_t, \qquad
\text{gross}_t = \sum_i D_{t-1,i}\, r_{t,i}, \qquad
A_t = \text{cash}_t + \sum_i D_{t,i}
$$

### costs and capacity

The per-unit cost is a linear term (exchange fee plus half spread, 10 plus 5 bps by default) and an optional square-root impact term,

$$
c_{t,i} = \ell + \eta\, \sigma_{t,i} \sqrt{\frac{|h_{t,i} - p_{t,i}|\, E^-_t\, \text{AUM}}{\text{ADV}_{t,i}}}
$$

where $\sigma$ is trailing daily volatility and ADV trailing dollar volume, both causal. Impact makes capacity visible, since the same weights lose more per trade as the book grows.

### point-in-time data

Every stored bar carries two timestamps, `available_at` (the bar's close plus 1 ms, the earliest moment its values could be known) and `ingested_at` (when the store learned them). A read as of instant $X$ keeps, for each (symbol, date), the latest version with both timestamps $\le X$. That is enough to reproduce any past read exactly, including across restatements, and it means a run can record precisely which data it saw.

## architecture

The pipeline is a straight line of stages, each with one data structure in and one out.

```
            data.binance.vision (public S3 archive, one zip per symbol-month)
                                   |
  qrp ingest --> data/binance.py   list pairs, keep 300, fetch in parallel,
                                   fix ms/us timestamps, stamp available_at
                                   |
                 data/store.py     BarStore, append only
                                     bars/freq=1d/year=YYYY/<ingest_id>.parquet
                                     meta/ingests.parquet
                                   load(as_of) keeps the latest visible version
                                   |  long polars frame
                 panel.py          split_relisted, then wide (T, N) float64 arrays
                                   |
        +--------------------------+---------------------------+
        |                          |                           |
  universe_mask             features/spec.py              forward_label
  (trailing $vol,           FeatureSpec DAG               y[t] = close[t+d+h]
   history, has bar)        + audit_causality                  / close[t+d] - 1
        |                   (fails the run on leak)            |
        +--------------------------+---------------------------+
                                   |
                 cv.py             purged walk-forward or k-fold splits
                 research.py       ridge per split, out of sample scores,
                                   dollar-neutral rank weights W (T, N)
                                   |
                 backtest/engine   run_backtest(W, close, costs, delay)
                                   reference | numpy | numba kernels
                                   |
                 backtest/metrics  Sharpe, IC, drawdown, turnover, cost drag
                 tracker.py        runs/<id>/{config,meta,metrics}.json + artifacts
                                   |
                 qrp runs list | show | compare
```

The one deliberate split is between storage and compute. Storage is long parquet through polars, which is good at partition pruning and the point-in-time group-by. Compute is wide numpy arrays of shape (dates, symbols) with NaN meaning "no bar", which is good at rolling and cross-sectional operations over a whole block. Nothing is forward filled at the panel layer, so `np.isfinite(close)` stays exactly "this asset traded today". Partitions are by year, not symbol, which gives five files per ingest instead of 1,500 tiny ones.

## implementation

### the store

`BarStore.ingest` validates (no duplicate keys, high at least low, positive close), stamps `ingested_at` and an ingest id, and writes one parquet file per year partition plus a row in an ingest log. Nothing is ever overwritten.

```python
if as_of is not None:
    lf = lf.filter((pl.col("ingested_at") <= as_of) & (pl.col("available_at") <= as_of))
# Latest known version of each bar wins (restatements).
lf = (lf.sort("ingested_at")
        .group_by(["symbol", "date"]).last()
        .drop("year")
        .sort(["symbol", "date"]))
```

`snapshot_id(as_of)` hashes the ingest ids visible at that instant, and every run records it. All experiments below ran on snapshot `417cbe2508ff`, which is one ingest of 466,274 bars across 300 symbols, 83 of which stopped trading before the end of the window.

### features without pandas

Rolling windows are the core of almost every feature, and the usual pandas approach works one column at a time and needs care around missing values. I used masked cumulative sums over the whole (T, N) block instead. Replace NaN with zero, cumulatively sum the values, their squares and a finite mask, and difference at lag $w$.

```python
finite = np.isfinite(x)
xz = np.where(finite, x, 0.0)
c1 = np.concatenate([pad, np.cumsum(xz, axis=0)])
cn = np.concatenate([pad, np.cumsum(finite, axis=0)])
s[w - 1:] = c1[w:] - c1[:-w]
n[w - 1:] = cn[w:] - cn[:-w]
# a window is valid only if all w values were finite
mean = np.where(n == w, s / w, np.nan)
```

A window with any missing bar is NaN rather than silently averaged over fewer values. Cross-sectional rank is a per-row argsort scaled to $[-0.5, 0.5]$. Specs come from TOML tables, inputs can be panel fields or other features, and the specs are topologically sorted so declaration order does not matter. There are nine causal operators and three deliberately leaky ones (a forward return, a centered moving average, a full-sample z-score) in a separate registry, so the audit has something to catch in tests and in the demo config.

### the leakage audit

The audit runs both probes at four cut points, treating NaN as equal to NaN.

```python
for c in cuts:
    trunc = compute_features(panel.slice_time(c + 1), specs)
    pert_panel = panel.copy()
    for k, v in pert_panel.fields.items():
        v[c + 1:] *= np.exp(rng.normal(0, 0.5, v[c + 1:].shape))
    pert = compute_features(pert_panel, specs)
    for s in specs:
        record(s.name, _same(trunc[s.name], base[s.name][: c + 1]).all(axis=1), c, "truncation")
        record(s.name, _same(pert[s.name][: c + 1], base[s.name][: c + 1]).all(axis=1), c,
               "perturbation")
```

It costs nine feature computations per run (the base plus two probes at four cuts). `run_research` calls it first, on the exact spec the run uses, and raises `LookAheadError` if anything leaks. The failed run is still written to the tracker with status "failed", so a refused experiment leaves a trace.

### three engines for one formula

The backtester has three kernels implementing the recursion above. The reference kernel is plain Python holding a cash balance and a share count per asset. It is slow on purpose. The numpy kernel loops over days and vectorizes over (strategies, assets). The numba kernel is a compiled triple loop, parallel over strategies.

```python
for s in numba.prange(S):
    A = 1.0
    D = np.zeros(N)
    for t in range(T):
        gpnl = 0.0
        for i in range(N):
            gpnl += D[i] * r[t, i]
        Em = A + gpnl
        ...
        for i in range(N):
            p = D[i] * (1.0 + r[t, i]) / Es
            hi = H[s, t, i] if tradeable[t, i] else p
            dw = abs(hi - p)
            unit = lin
            if impact > 0.0:
                unit = lin + impact * sigma[t, i] * np.sqrt(dw * Es * aum / adv[t, i])
            k += dw * unit
```

The test holding this together runs all three engines on a synthetic panel with 3% missing bars, four random strategies, three cost models (zero, linear, linear plus impact) and delays 0, 1 and 2, and requires equity, gross PnL, cost, turnover, exposures and full dollar positions to agree with the reference to a relative tolerance of 1e-10. Missing bars earn a zero return (price carried forward) and cannot be traded. If pre-trade equity reaches zero, the strategy is marked bankrupt and holds nothing afterwards.

Execution delay is a shift of the target array before the kernel, and a negative delay raises, since it would mean trading before the signal exists. A separate test changes the targets from some row $c$ onward and checks that equity before row $c + d$ is unchanged. That is the backtester's own look-ahead test.

### purged splits

The purge and embargo logic is a handful of boolean masks over row indices.

```python
if purge:
    # Rows before the block whose labels reach into it.
    train_mask &= ~((rows < a) & (rows + label_horizon >= a))
    # Rows after the block that start inside the test labels' span.
    train_mask &= ~((rows > b) & (rows <= b + label_horizon))
if embargo > 0:
    h = label_horizon if purge else 0
    train_mask &= ~((rows > b) & (rows <= b + h + embargo))
```

Here `label_horizon` is $d + h$. A separate `check_split` verifies any split against the contract independently of how it was built, and the tests run it over both split kinds with several horizons and embargoes.

### model, portfolio and tracker

The model is closed-form ridge with an unpenalized intercept, fitted on pooled (date, symbol) rows of cross-sectionally ranked features, predicting the cross-sectional rank of the forward return. Out of sample scores become dollar-neutral weights proportional to the demeaned score rank, scaled to gross exposure 1, and optionally held for $k$ days. It is a deliberately plain model. A plain model makes every downstream effect easy to attribute.

A tracked run is a directory named by timestamp, run name and config hash, holding the fully resolved config, metadata (git sha, library versions, data snapshot, duration, status), metrics and artifacts. `qrp runs compare a b` prints metrics side by side and only the config keys that differ, which is the view I actually wanted every time I compared two runs.

## problems

These are the real bugs and surprises, in the order I hit them. Two of them were in my own tests, and the most instructive one was in the data.

### the obvious data sources were blocked

`api.binance.com` returned HTTP 451, the US geo-block, and Yahoo's chart API returned HTTP 429 on the first request. That is why v0 has no US equities. The free equity sources also lack delisted names, which would have made survivorship bias unavoidable. I switched to `data.binance.vision`, Binance's public S3 bulk archive, which serves one zip per symbol and month, has no geo-block, and includes pairs that were later delisted. The ingest lists every USDT pair through the paginated S3 XML listing, drops leveraged tokens and stablecoin pairs, and keeps the 300 with the most archived months in the window.

### non-ASCII tickers crashed the first ingest

The S3 listing contains a few symbols with Chinese names, and `urllib` raised a `UnicodeEncodeError` when building their URLs. I now keep only ASCII alphanumeric symbols, which are the only ones in scope for this universe anyway.

### binance changed its timestamp unit

Spot archive files from 2025-01-01 onward use microseconds instead of milliseconds. Read naively, every 2025 and 2026 bar would land somewhere around the year 57000, and the panel would have silently lost its last 20 months. The fix is a threshold, since no millisecond epoch before the year 5000 exceeds $10^{14}$.

```python
def _to_millis(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(np.int64)
    return np.where(ts > 10**14, ts // 1000, ts)
```

A test pins the conversion for a 2025-01-01 timestamp in both units.

### my drift test was wrong and the engine was right

I wrote a test expecting that targets equal to the buy-and-hold drift weights would need zero turnover after the initial purchase. Without costs, that held. With 15 bps costs, day 2 still traded 0.15% of equity, and I assumed an engine bug.

It was not. The entry cost is paid from cash, so after the first trade the book holds assets worth 100% of pre-trade equity against slightly less than 100% of post-trade equity. That is a small leverage, equal to the fee. A target of 100% invested therefore has to sell a little the next day to get back to 100%, and that sale pays its own fee, which requires a smaller correction the day after, decaying geometrically. The reference ledger and the vectorized kernels agreed on this exactly, which is what convinced me the test was wrong rather than both engines. The test now asserts zero turnover without costs, about 15 bps on day 2 with costs, and a total that stays below twice that.

A second hand-built test had the position on a halted day as 0.55 when it is 0.525, because day 1 had already rebalanced back to 50% of an equity of 1.05. Writing expected values by hand for tiny cases was still worth it, since that is where the delevering effect surfaced.

### the luna relisting blew up a backtest

The first purged k-fold run went to exactly zero equity on 2022-05-31, with a daily turnover above 2, which should be impossible for a book with gross exposure 1.

The data explains it. LUNAUSDT closed at \$1.08 on 2022-05-11, \$0.00032 on 2022-05-12 and \$0.00005 on 2022-05-13. It then had no bars for 18 days, and came back on 2022-05-31 at \$8.87 as LUNA 2.0, a different asset under the same ticker. The backtester forward-fills prices through missing bars (correctly, for a halted asset), so a short LUNA position that was stuck because the asset was untradeable saw a one-day return of about 177,000x. The walk-forward runs never noticed, because their test periods start in 2023.

Eight symbols in the data have gaps longer than three days.

| Symbol | Gap in days | Resumes |
| --- | --- | --- |
| FTTUSDT | 311 | 2023-09-22 |
| CVCUSDT | 154 | 2023-05-12 |
| KEYUSDT | 28 | 2023-03-10 |
| LUNAUSDT | 18 | 2022-05-31 |
| VIDTUSDT | 9 | 2022-11-09 |
| STRAXUSDT | 8 | 2024-03-28 |
| BNXUSDT | 6 | 2023-02-22 |
| QUICKUSDT | 4 | 2023-07-21 |

The v0 fix is `split_relisted`, which makes any gap over three days start a new instrument.

```python
gap = pl.col("date").diff().over("symbol").dt.total_days()
seg = (gap > max_gap_days).fill_null(False).cast(pl.Int32).cum_sum().over("symbol")
```

The old LUNAUSDT is now delisted at its last real price and `LUNAUSDT#2` starts with no history. That took the panel from 300 to 308 instruments. A regression test builds a tiny LUNA-shaped series, shows a held short going bankrupt on the raw data, and shows the same short surviving after the split. It is a blunt rule. A redenomination without a trading pause would still produce a fake return, and a real corporate actions table is on the list below.

### a high IC with a low Sharpe looked suspicious

The baseline ridge had an out of sample information coefficient (mean cross-sectional rank correlation between score and realized return) of 0.125. That is high for daily crypto. Meanwhile every single feature I would have guessed as the driver had an IC near zero or negative on its own. Shorting yesterday's winners had an IC of 0.000, and 30 day momentum had an IC of -0.049.

After the LUNA bug I did not trust any number better than I expected, so I added a random-score negative control. It came back with an IC of 0.0005 and a Sharpe of -1.34, the signature of trading noise through a cost model, so the IC calculation itself was not inflated. Then I computed single-feature ICs by hand. The driver is 30 day volatility, with a clearly negative IC, meaning high-volatility coins underperformed. I did not save that single-feature number to a results file, so I do not quote it here, and saving it is on the list below. The Sharpe is much lower than the IC suggests because a long-short crypto book carries large idiosyncratic volatility and a weekly holding period uses only part of a 5 day signal.

### the machine was shared

Ten or more other jobs shared the 14-core machine for the whole session, with the load average between about 220 and 400. The experiment suite hit a 10 minute tool timeout on its first attempt and was rerun in the background, where it took 891 seconds for 17 runs. The benchmark recorded a load average of 360, 351 and 311 before it started and 231, 286 and 292 after. Every throughput number below is a lower bound, and the multithreaded ones are noisy enough that at some sizes 14 threads were slower than one.

### the forward fill cost more than the kernel

A quick profile of one strategy on 300 assets, which I did not save to the results directory, showed `prepare_market` (forward fill plus returns) taking about 18 ms of a 23 ms call under load. The compiled kernel was the cheap part. Rather than micro-optimize the forward fill, I made `run_backtest` accept a precomputed market, which is the right shape for sweeping many strategies over one panel anyway, and added a test that the result is identical. The benchmark rows labelled "precomputed" measure that path.

## experiments

All experiments ran on snapshot `417cbe2508ff`, a panel of 1,704 daily rows from 2022-01-01 to 2026-08-31 by 308 instruments (300 symbols before the relisting split, 308 instruments after it). The panel loads in 2.88 s.

### the base configuration

- Universe. The top 100 instruments by trailing 30 day mean dollar volume with at least 60 bars of history, recomputed every day.
- Features. Cross-sectional ranks of the 1 day return, 7, 30 and 90 day momentum, 30 day volatility, 20 day moving average ratio and 20 day volume z-score.
- Model. Ridge with alpha 10, predicting the rank of the 5 day forward return.
- CV. Walk-forward with 8 test blocks after a 365 day minimum training period, purged on the label interval, 5 row embargo.
- Portfolio. Dollar neutral, weights proportional to the demeaned score rank, gross 1, rebalanced every 5 days, executed with a 1 day delay.
- Costs. 10 bps fee plus 5 bps slippage per unit traded.

Metrics use only out of sample days, 1,338 of them from 2023-01-02 (the first test row, 2023-01-01, plus the 1 day delay).

### the variants

Each variant changes one thing through `--set` overrides and is its own tracked run.

1. Costs. 0, 5, 15 (baseline), 25 and 50 bps one way, plus square-root impact with coefficient 0.5 at a \$50M book.
2. Rebalance frequency. Daily instead of every 5 days.
3. Execution timing. Delay 0 against delay 1.
4. Controls. A random score, and two single-signal baselines with no model (short 1 day winners, long 30 day momentum).
5. Leakage. One leaky feature added to the model with the audit's fail switch turned off, either a 5 day forward return or an 11 day centered moving average ratio. Separately, `qrp audit` on a demo config with three honest and three leaky features, on the real data.
6. Purging. A 20 day label horizon, so that overlap is large, under walk-forward and k-fold, each with purge plus a 10 row embargo against neither.
7. Throughput. Synthetic panels with T = 1,825 days, N in {100, 300, 1000} assets and S in {1, 16, 128} strategies per call, for the numpy kernel and the numba kernel on 1 and 14 threads, plus the reference engine at a small size, the numba kernel with a precomputed market, and the real 1,704 by 300 panel. Timings are best of 5 (best of 1 for the largest numpy cases).

## results

An independent reviewer later reran the full experiment suite from a clean download and matched all 17 rows of `results/experiments.csv` exactly, and reran the leakage audit with identical output. The benchmark was not rerun, so the throughput numbers rest on the original measurement only. Two corrections from that review are applied here (the out of sample start date, and the direction of the 300 to 308 instrument split).

One caveat applies to every Sharpe below. All 17 experiments share the same 2023 to 2026-08 out of sample period, and I looked at it while debugging the IC. These are comparisons between variants on one platform, not one-shot holdout results.

### leakage

The audit on the demo config flagged exactly the three leaky features on real data and passed the three honest ones. The forward return and the centered average were both first caught at row 422 against a cut at row 426, which is the correct answer, since a 5 day lead and a 5 bar half window both reach row 427 from row 422. The full-sample z-score was caught at row 30, the first row where its input exists, because a full-sample statistic contaminates every row. The command exits 1.

What the audit prevents is visible when it is switched off.

<figure data-figure="chart:projects/quant-research-platform/leakage-sharpe"></figure>

The 5 day forward return turns a Sharpe of 0.49 into 19.40, with an IC of 0.724, an annual return of 9,161% and a max drawdown of 2.5%. The centered average is the more instructive case. It is not an obvious bug. It is what `rolling(center=True)` does in pandas, it only sees five bars ahead through a smoothing window, and it still produces a Sharpe of 13.33, an IC of 0.438 and a max drawdown of 3.4%, ending at about 21,900 times its starting equity. Nobody would believe 19. Somebody might believe a subtler leak that added 0.3 to a Sharpe of 0.5. The audit would flag that one with the same two probes, as long as the leak lives in a feature and not in the labels or the backtester, which have their own tests.

### baseline

Out of sample, the baseline earned a Sharpe of 0.490, an annual return of 7.3% at 17.6% annual volatility, a max drawdown of 29.7% and an IC of 0.125. Average daily turnover was 15.9% of equity and the annual cost drag was 8.7%. Over 1,338 days the equity grew 29.5%.

### costs

<figure data-figure="chart:projects/quant-research-platform/cost-sensitivity"></figure>

Turnover is identical across the cost runs, so Sharpe falls almost linearly with cost, from 0.988 at zero to 0.822 at 5 bps, 0.490 at 15, 0.159 at 25 and -0.659 at 50. Interpolating between the last two points puts break-even near 30 bps one way. At zero cost the same signal has a Sharpe of 0.988 and a max drawdown of 20.9% instead of 29.7%.

Adding square-root impact at a \$50M book gives a Sharpe of -1.078 and a cost drag of 37.0% a year, so this signal has very little capacity. Rebalancing daily instead of weekly doubled turnover from 15.9% to 31.6% a day and cut the Sharpe from 0.490 to 0.122. The fresher signal did not earn back the extra trading.

### timing

Trading at the same close the signal observed (delay 0) raised the Sharpe from 0.490 to 0.692 and the IC from 0.125 to 0.133. One day of execution optimism overstates the Sharpe by about 41%. That is much harder to spot than a leaky feature, because 0.69 is a believable number.

### controls

The random score had an IC of 0.0005 and a Sharpe of -1.339, with a cost drag of 16.8% a year from 30.6% daily turnover. Shorting 1 day winners alone had an IC of 0.000 and a Sharpe of -0.848. Long 30 day momentum alone had an IC of -0.049 and a Sharpe of -0.324. Neither textbook signal works in this universe and period, which is why the baseline's edge needed explaining.

### purging

<figure data-figure="chart:projects/quant-research-platform/purging"></figure>

This is a real negative result. With a 20 day horizon, walk-forward gave a Sharpe of 0.842 purged against 0.847 unpurged, and k-fold gave 0.850 against 0.844. The ICs differ by less than 0.002, and the sign of the difference flips between the two split kinds.

The explanation is capacity. A seven-coefficient ridge trained on tens of thousands of pooled rows cannot memorize the 21 rows at each block boundary whose labels overlap the test block, so the leak that purging removes is too small to register. Purging stays on by default, because project 09 runs higher-capacity models on this pipeline. But this experiment does not show purging doing useful work, only that it is correct by test.

### throughput

<figure data-figure="chart:projects/quant-research-platform/throughput"></figure>

Measured on an Apple M4 Pro with 14 cores, Python 3.12.13, numpy 2.5.3 and numba 0.67.0, with the load average between 230 and 360 during the run.

The pure-Python reference engine ran 1,386 strategy-days per second at N = 100, or 139 thousand asset-days per second. With the market precomputed, the single-threaded numba kernel ran 772 thousand strategy-days per second at N = 100, 270 thousand at N = 300 and 79 thousand at N = 1000. In asset-days that is a flat 77 to 81 million per second across a tenfold change in width, which says the kernel does constant work per cell, and it is 557 to 585 times the reference engine per asset-day. That is the price of an oracle, and it only runs in tests.

The best batched number was 2.12 million strategy-days per second, on 14 threads with S = 16 and N = 100. On the real 1,704 by 300 panel, one strategy took 10.8 ms and 64 random strategies took 0.74 s in one call, 147 thousand strategy-days per second. numpy was between 1.2 times (S = 1, N = 1000) and 20 times (S = 1, N = 100) slower than single-threaded numba, since its per-day overhead only amortizes over wide rows.

Throughput falls at the largest size. At S = 128 and N = 1000, single-threaded numba dropped to 43 thousand strategy-days per second, and the 14-thread run was slower still at 26.5 thousand. The (S, T, N) target array there is about 1.9 GB, and `run_backtest` copies it twice before the kernel starts (to lag it and to clean NaNs), so the call is dominated by memory traffic. With other jobs also contending for memory bandwidth, extra threads made it worse.

### tests

All 73 tests pass, on synthetic data. They cover look-ahead detection (including a one-bar peek and a leak passed through the DAG), accounting identities, cross-engine agreement, closed-form cases, purging and embargo, point-in-time reads across restatements, the relisting split, and the pipeline end to end.

## what i would change

### lag and clean the targets inside the kernel

Materializing two copies of the (S, T, N) array is the reason throughput collapses at S = 128 and N = 1000. The kernel can read `W[s, t - d, i]` directly and treat NaN as zero on the fly.

### rerun the benchmark on an idle machine

Everything here was measured at a load average between 220 and 400. I would not draw conclusions from the 14-thread rows at all.

### replace the gap rule with real reference data

A symbol mapping and corporate actions table would handle ticker reuse and redenominations properly, and delisted positions should be closed at a delisting price instead of being held at the last price forever.

### build the universe point in time

The 300 pairs were chosen as the ones with the most archived months in the window, which uses today's knowledge. The daily universe filter inside that set is causal, but the set itself carries mild selection bias. Point-in-time exchange listings would remove it.

### show purging mattering

A gradient-boosted model or a much smaller sample should make the effect visible, which is the experiment to add once project 09 brings nonlinear models.

### save single-feature ICs with every run

They explained the baseline faster than anything else, and saving them would have let me quote the volatility IC here.

### make the audit cheaper

It recomputes every feature nine times, which is fine for 14 features on daily bars and will not be for minute bars. Caching the unchanged prefix would remove most of the truncation probe's work.

## reproducibility

Everything runs from the project directory with [uv](https://docs.astral.sh/uv/) and Python 3.12. The data is about 13 MB of parquet from roughly 15,400 small zip files, and is gitignored.

```bash
cd projects/08-leakage-checked-backtester
uv sync
./scripts/download_data.sh                          # same as: uv run qrp ingest --root data
uv run qrp info                                     # ingest log and symbol metadata
uv run pytest                                       # 73 tests, synthetic data only

uv run qrp audit configs/leaky_demo.toml            # flags the 3 leaky features, exits 1
uv run qrp run configs/xs_ridge.toml                # tracked baseline run
uv run qrp run configs/xs_ridge.toml --set costs.fee_bps=0 --set portfolio.rebalance_every=1
uv run qrp runs list --sort sharpe
uv run qrp runs compare <run_id_a> <run_id_b>

uv run python scripts/run_experiments.py            # results/experiments.{csv,json}, plots
uv run python scripts/bench_backtest.py --real      # results/bench_backtest.{csv,json,png}
```

A fresh download gets a new ingest id and therefore a different snapshot id from `417cbe2508ff` (the reviewer's was `f2322f2cf2d1`), but the same 466,274 rows, and the experiment table reproduced exactly. Throughput will differ on any other machine, and should be higher on an idle one.

Code is in `projects/08-leakage-checked-backtester`. The raw outputs behind every number in this post are in its `results/` directory, `DESIGN.md` lists each invariant with the test that enforces it, and `DEVLOG.md` has the build log.
