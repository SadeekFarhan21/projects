# Design

## Goal

A small, auditable pipeline where the dangerous mistakes of backtesting
(look-ahead, survivorship, silent re-use of the holdout, uncounted trials) are
either impossible by construction or caught by a test.

## Architecture

```
 data.binance.vision (public S3)
          |
          |  scripts/download_data.py  (27k monthly zips, threaded, cached in data/raw)
          v
 data.py: parse_kline_csv -> data/processed/klines_1d.parquet
          |   long panel: one row per (date, symbol), OHLC + quote volume
          |   load_panel()  <-- the only entry point; project 08 replaces it later
          v
 features.build_dataset(panel, end=...)       hard truncation at `end` first
          |   split_relistings   SYM -> SYM#2 after a gap > 7 days
          |   to_wide            date x symbol frames on a complete daily index
          |   raw features       9 wide frames, each using rows <= t
          |   universe.eligibility   point-in-time mask (history, liquidity, exclusions, top 50)
          |   cs_rank            centered ranks in [-0.5, 0.5] among members
          v
 Dataset.long: (date, symbol) member rows x [9 ranked features, fwd_ret, y_rank]
          |
          |  walkforward.walk_forward(model_factory, long, folds)
          |     for each fold: fit on date <= train_end, predict test window
          v
 forecast Series (date, symbol)
          |
          +--> metrics.daily_ic (Spearman per day) --> IC mean, NW t, IR
          |
          +--> portfolio.weights_rank / weights_quintile  (dollar neutral, gross 1)
                    |
                    v
               portfolio.backtest(weights, fwd_ret, lag)
                    |  gross = sum w * r, turnover = sum |dw|, net = gross - turnover * cost
                    v
               metrics.summarize  --> Sharpe, PSR, max DD, cost grid
                    |
     scripts/run_dev.py            scripts/run_holdout.py
     all 18 trials on 9 folds  --> selection.json --> one refit, one evaluation
     append trials.jsonl           LOCK.json written before any number
     deflated Sharpe of best       H1 / H2 verdicts
```

## Key data structures

- Long panel (`load_panel`): columns `date, symbol, open, high, low, close,
  volume, quote_volume`. Dates are UTC days. `quote_volume` is USDT traded.
- Wide frames: `DataFrame[date x symbol]` on a gap-free daily index. NaN means
  no bar. All rolling features are computed here because vectorized rolling on
  wide frames is simple and fast (the full panel is 2373 x 522 in dev).
- `Dataset.long`: `DataFrame` indexed by `(date, symbol)`, only universe
  members. Columns are the 9 ranked features, `fwd_ret` (raw next-day return,
  NaN if the next bar is missing) and `y_rank` (the training label).
- Forecasts and weights are `Series` on the same `(date, symbol)` index.
- `Backtest.daily`: `DataFrame[date]` with `gross` and `turnover`; `net(bps)`
  derives net returns for any cost level without recomputation.
- `results/dev/trials.jsonl`: append-only log of every trial ever evaluated in
  development. Its line count is the N in the deflated Sharpe ratio.
- `results/holdout/LOCK.json`: timestamp, SHA-256 of `PREREGISTRATION.md`, the
  frozen selection, and the verdicts.

## Invariants

1. A feature or universe decision at date t uses only bars dated <= t.
   Tested by perturbing prices after T and asserting nothing before T changes
   (`test_features_do_not_look_ahead`, `test_no_lookahead_in_universe`).
2. `fwd_ret` at t is `close[t+1] / close[t] - 1`, and is the only column that
   looks forward. It is never an input to a model. (`test_forward_return_alignment`)
3. Weights never depend on whether the next bar exists. A missing next bar
   earns 0. (`test_backtest_pnl_turnover_and_missing_fwd`)
4. A training row dated d satisfies `d <= test_start - 6 days`, so the last
   label is at least 5 days before the first test day.
   (`test_dev_folds`, `test_walk_forward_respects_train_end`)
5. Every portfolio is dollar neutral with gross exposure 1 on every day it
   trades. (`test_rank_weights_neutral_and_gross_one`, `test_quintile_weights`)
6. `run_dev.py` truncates the raw panel at 2024-06-30 before building
   anything, so no holdout bar can reach the development results.
7. `run_holdout.py` writes the lock before computing and refuses to start if
   the lock exists.
8. No return is computed across two different assets sharing a ticker
   (relisting split). (`test_split_relistings`)

## Trade-offs chosen and rejected

### Crypto instead of US equities
Chosen: Binance's archive is free, includes delisted pairs, has exact daily
quote volume, and trades 24/7 so "signal at close, trade at next open" is the
same instant. Rejected: Yahoo daily equities, because a free survivorship-free
constituent list is hard to get, and a current S&P 500 list would bake in
survivorship bias.

### Point-in-time top 50 by trailing liquidity, from all USDT pairs
Chosen over a hand-picked list of "top coins", which would leak today's
knowledge of which coins survived. The cost is downloading every USDT pair
(27,721 files) instead of ~50.

### Rank everything
Features and labels are converted to cross-sectional ranks. Crypto daily
returns have tails like +392% (DOGE on 2021-01-28, `results/dev/data_report.json`);
raw regression targets would be dominated by a handful of days. The cost:
the model optimizes rank IC, which turned out not to be the same as PnL.

### Dense wide frames instead of a groupby on the long panel
Rolling windows on a date x symbol frame are one vectorized call each. The
long panel is only built at the end, for member rows. Rejected: per-symbol
groupby rolling, which was simpler to read but has to be careful about gaps.

### Rolling windows by row, not by trading-calendar logic
Because the index is a complete daily calendar and crypto trades every day,
a row is a day. Short maintenance gaps produce NaN returns that propagate
into rolling windows through `min_periods`, which is acceptable.

### Single-threaded sklearn
`HistGradientBoostingRegressor` hung in its OpenMP pool on this machine (see
DEVLOG). All sklearn calls run under `threadpool_limits(1)`. A depth 3 fit on
75k rows takes about 3 s, so the loss is small.

### Selection by dev net Sharpe
Pre-registered because Sharpe net of costs is what one would trade on. In
hindsight it is a noisy criterion (see DEVLOG "What I would change"); IC or a
blend would have been more stable. It was not changed after seeing results.

### No short-borrow or funding model
Rejected for v0 because free historical funding data needs a separate
downloader and the result was already negative before these costs. Stated as
a limitation next to every H2 number.
