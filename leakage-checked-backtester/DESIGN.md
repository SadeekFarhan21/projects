# qrp design

qrp is a daily-bar research platform in about 1,650 lines of Python (plus about 550 lines of tests). The design goal is that a result it prints is one you can trust: every stage has a written timing contract, and each contract has a test that fails when it is broken.

## Architecture

```
                 data.binance.vision (public S3 archive, monthly zips)
                                  |
                                  v
  qrp ingest ---> data/binance.py  list symbols, pick up to 300, fetch zips in parallel,
                                   fix ms/us timestamps, stamp available_at = bar close
                                  |
                                  v
                 data/store.py  BarStore (append only)
                   bars/freq=1d/year=YYYY/<ingest_id>.parquet   rows carry ingested_at
                   meta/ingests.parquet                          ingest log
                   meta/symbols/<ingest_id>.parquet              symbol records
                   load(as_of) -> latest version with ingested_at <= as_of
                                              and available_at <= as_of
                                  |
                                  v   long polars frame
                 panel.py  split_relisted (ticker reuse -> new instrument)
                           Panel.from_long -> dates (T), symbols (N), fields {name: (T,N)}
                                  |
          +-----------------------+------------------------+
          |                       |                        |
          v                       v                        v
  research.universe_mask   features/spec.py          research.forward_label
  (top K by trailing $vol, FeatureSpec DAG ->        y[t] = close[t+d+h]/close[t+d] - 1
   min history, has bar)   features/ops.py kernels   (labels only, never features)
          |                audit_causality  <---- fails the run on look-ahead
          |                       |                        |
          +-----------+-----------+------------------------+
                      v
               cv.py  purged_splits (walk forward or k-fold, purge + embargo
                      on the label interval [t, t + d + h])
                      |
                      v
               research.fit_ridge per split -> out-of-sample scores
               research.rank_weights -> dollar neutral targets W (T,N)
                      |
                      v
               backtest/engine.py  run_backtest(W, close, CostModel, delay)
                 kernels: reference (share/cash oracle) | numpy | numba (parallel over S)
                      |
                      v
               backtest/metrics.py  summarize on out-of-sample rows
                      |
                      v
               tracker.py  runs/<id>/{config,meta,metrics}.json + artifacts/
                      |
                      v
               qrp runs list | show | compare
```

## Key data structures

### Bar rows (storage)

`symbol, date, open, high, low, close, volume, quote_volume, trades, available_at, ingested_at, ingest_id`. `date` is the UTC day the bar opened. `available_at` is the bar close plus 1 ms, the earliest instant its values could be known. `ingested_at` is when the store learned them.

### Panel (compute)

`Panel(dates: datetime64[D] (T,), symbols: list[str] (N,), fields: dict[str, float64 (T, N)])`. A missing bar is NaN. Nothing is forward filled at this layer, so `np.isfinite(close)` is exactly "this asset traded today".

### FeatureSpec

`FeatureSpec(name, op, inputs, params)`, loaded from TOML `[[features]]` tables. Inputs are panel fields or other feature names, so specs form a DAG that is topologically sorted (declaration order does not matter, cycles and unknown inputs are errors). Ops are plain functions from (T, N) arrays to a (T, N) array.

### Split

`Split(train: row indices, test: contiguous row indices)` over panel rows. All symbols share rows, so a split is a set of dates.

### BacktestResult

`(S, T)` arrays: `equity` (post-trade), `returns`, `gross_pnl`, `cost`, `turnover`, `gross_exposure`, `net_exposure`, and optional `(S, T, N)` dollar `positions`. Single-strategy inputs return squeezed `(T,)` arrays.

### Run directory

`runs/<timestamp>-<name>-<config hash>/` with `config.json` (fully resolved), `meta.json` (git sha, versions, data snapshot id, duration, status), `metrics.json`, `artifacts/equity.parquet`, `artifacts/leakage_audit.json`.

## Timing contract

Row t means "after the close of day t". This one convention is used everywhere.

1. A feature at row t may use bars at rows <= t only.
2. A target weight decided at row t is executed at the close of row t + delay (default delay = 1). delay = 0 is allowed but optimistic, since it trades at the same close the signal just observed.
3. The training label for row t is the return from close t + delay to close t + delay + horizon. Its information interval is [t, t + delay + horizon], and CV purges on that interval.
4. The universe at row t is computed from rows <= t (trailing dollar volume, history count, has a bar today).

## Invariants and the tests that enforce them

| Invariant | Test |
|---|---|
| Every shipped feature op is causal | `test_every_causal_op_passes_audit` (truncation and perturbation probes) |
| A peeking feature is rejected, including a one-bar lead and leaks passed through the DAG | `test_every_leaky_op_fails_audit`, `test_one_bar_peek_is_caught`, `test_leak_propagates_through_dag` |
| A run with a leaky feature refuses to produce results | `test_pipeline_refuses_leaky_config` |
| Changing targets from row c on cannot change equity before row c + delay | `test_backtester_does_not_peek_at_future_weights` |
| `A[t] = A[t-1] + gross_pnl[t] - cost[t]` and returns compound to equity | `test_equity_identity` |
| `gross_pnl[t] = sum_i D[t-1, i] * r[t, i]` | `test_gross_pnl_equals_sum_of_asset_pnl` |
| Vectorized kernels equal the share/cash oracle to 1e-10 (with costs, impact, gaps, delays 0 to 2) | `test_engines_agree_with_reference` |
| Buy and hold, equal weight rebalance and drift-matching targets give their closed form answers | `test_single_asset_buy_and_hold`, `test_equal_weight_daily_rebalance_is_cross_sectional_mean`, `test_drifting_targets_have_zero_turnover` |
| No training row's label interval overlaps a test label interval; embargo gap respected | `test_purged_splits_are_clean`, `test_kfold_embargo_gap` |
| A read at as_of reproduces what was known then (restatements, bars not yet closed) | `test_restatement_is_point_in_time`, `test_bar_not_visible_before_it_closes` |
| A reused ticker cannot produce a return across the gap | `test_relisted_ticker_becomes_new_instrument` |

## The leakage audit

`audit_causality(panel, specs)` is empirical: it does not trust op metadata. For a few cut points c it runs two probes and requires rows <= c to be bit-for-bit unchanged (NaN equal to NaN).

1. Truncation. Recompute every feature on `panel[:c+1]`. This catches statistics over the whole sample, such as a z-score with full-sample mean and std, because their value at early rows changes when the sample gets shorter.
2. Perturbation. Multiply every field after c by random positive noise and recompute. This catches leads (`shift(-k)`) and centered windows even if they are written in a way that survives truncation.

The audit runs by default at the start of every research run, on the exact spec the run uses, and the run fails with `LookAheadError` if anything leaks. It costs one feature computation per probe (9 in total with 4 cut points).

## Backtest accounting

Per strategy and day, with equity in units of starting capital:

```
E-[t] = A[t-1] + sum_i D[t-1,i] r[t,i]            mark to market
p[t]  = D[t-1] (1 + r[t]) / E-[t]                 pre-trade (drifted) weights
h[t]  = W[t-delay], except h = p where no bar      targets, untradeable assets are held
k[t]  = sum_i |h-p|_i (lin + impact sigma_i sqrt(|h-p|_i E- aum / adv_i))
A[t]  = E-[t] (1 - k[t])                          cost paid from cash
D[t]  = h[t] E-[t]                                dollar positions
```

Targets are fractions of pre-trade equity and costs come out of cash. That choice makes the share/cash reference engine and the vectorized kernels agree exactly, not just to first order. A consequence found by the tests is that a fully invested book ends up levered by the fee after each trade, so a 100% invested target triggers a small delevering trade the next day.

Missing bars earn a zero return (price carried forward) and cannot be traded. A delisted asset stays at its last price forever. If pre-trade equity reaches zero the strategy is marked bankrupt and holds nothing afterwards.

## Trade-offs

### Chosen

- Wide numpy panels for compute, long parquet for storage. Features and backtests are array math over (T, N), which is simpler and faster in numpy than group-by-symbol dataframe code. Polars handles I/O, partition pruning and the PIT dedup.
- Append-only store with two timestamps instead of overwriting files. Restatements and reproducible reads come for free, and the cost is a group-by at read time (0.9 to 2.9 seconds for all 466k rows on the loaded machine).
- Year partitions, not symbol partitions. 300 symbols by 5 years gives 5 files per ingest instead of 1,500 small ones, and research reads the whole cross-section anyway.
- A sequential time loop in the backtester. Exact drift and cost accounting makes day t depend on day t-1, so the loop over T stays and vectorization happens over strategies and assets. numba compiles the loop and parallelizes over strategies.
- Three engines for one formula. The slow pure-Python engine keeps shares and cash explicitly and serves as the test oracle. It is roughly 560 times slower per asset-day and that is fine.
- An empirical leakage audit instead of a whitelist. New ops get checked without anyone having to classify them.
- A plain-file tracker. JSON and parquet in a directory, readable with `cat` and `jq`, no server.
- Splitting instruments on gaps longer than 3 days. It is a blunt rule, but it removes the LUNA-style fake returns without a corporate actions database.

### Rejected

- Turnover as `sum |W[t] - W[t-1]|` with no drift. Fully vectorized and common, but it ignores that weights move with prices, so it misstates costs and breaks the equity identity. Measured cost is part of the result, so exactness won.
- pandas rolling windows. They are fine, but a masked cumulative sum in numpy does the same trailing windows on the whole (T, N) block, and makes NaN handling explicit (a window is valid only if all its values are finite).
- An event-driven backtester as the main engine. It is the right tool for intraday order logic, but for daily target-weight research it is orders of magnitude slower. It survives as the reference engine.
- MLflow or a database for tracking. Too heavy for v0, and harder to diff.
- Hive-partitioning by symbol. Too many tiny files for daily bars.
- yfinance for US equities. Yahoo returned HTTP 429 from this machine, and the free equity sources have no delisted names. Binance's archive is public, fast, and includes dead pairs.
