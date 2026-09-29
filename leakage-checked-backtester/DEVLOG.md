# DEVLOG, project 08, a quant research platform

### What I wanted to build

I wanted the smallest research platform whose numbers I would actually believe. Most backtests that look great are wrong in one of a few boring ways. A feature quietly uses tomorrow's data, the model is trained on labels that overlap the test period, the backtester trades at the same close the signal just saw, turnover is computed without drift so costs are understated, or a data quirk creates a fake 1000x return. So the goal for v0 was not a clever strategy. It was a pipeline where each of those failure modes has a written rule and a test that breaks when the rule is broken, fast enough that later projects (project 09 runs on this) can sweep many strategies.

Concretely that meant free daily data for a few hundred instruments in partitioned parquet with a point-in-time read path, a declarative feature spec, a vectorized backtester with costs and slippage, walk-forward CV with purging and embargo, a local run tracker with a compare command, leakage and accounting tests, and a throughput benchmark.

### Theory

Look-ahead bias. A feature value at row t must be a function of data at rows <= t. I test this empirically rather than by reading code. If you cut the data at row c, or scramble everything after c, feature values at rows <= c must not change at all. A shift(-1) fails the scramble probe at exactly row c. A z-score with full-sample mean and std fails the truncation probe everywhere, because shortening the sample changes the mean. A centered moving average fails both at the last half window before the cut.

Labels and purging. The label for a decision at row t is the forward return from close t + delay to close t + delay + horizon. Its information interval is [t, t + delay + horizon]. If a training row's interval overlaps a test row's interval, the training label contains returns that are also in the test period, so the model is partly graded on what it was trained on. Purging (Lopez de Prado, Advances in Financial Machine Learning, chapter 7) drops those training rows. Embargo drops a further band of rows after each test block, because features computed right after the block are serially correlated with the test labels. In a walk-forward split training data is always before the test block, so only purging matters. In k-fold, training data sits on both sides, so both matter.

Backtest accounting. A target-weight backtester is a recursion. Weights drift with prices between rebalances, so the trade needed at day t is the difference between the target and the drifted weights, not between today's and yesterday's targets. Costs reduce equity, which changes tomorrow's drifted weights, so day t depends on day t-1 through the cost term. I chose the convention that targets are fractions of pre-trade equity and costs are paid from cash, because under that convention an explicit share and cash ledger gives exactly the same numbers as the weight recursion. That lets a slow, obviously correct ledger act as the oracle for the fast kernels. The identities are A[t] = A[t-1] + gross_pnl[t] - cost[t], gross_pnl[t] = sum over assets of yesterday's dollar position times today's return, and equity = cash + positions.

Costs. A linear cost per unit traded (fee plus half spread) and an optional square-root impact term, cost per unit = impact * sigma * sqrt(trade dollars / average daily dollar volume). Impact makes capacity visible, since the same strategy loses more per trade as AUM grows.

Point in time data. Every stored bar has two timestamps, when the bar closed (available_at) and when the store learned it (ingested_at). A read as of time X keeps the latest version of each bar with both timestamps <= X. That is enough to reproduce any past read exactly, including across restatements.

### Architecture

The pipeline is a straight line of stages, each with one data structure in and one out.

```
binance archive -> BarStore (parquet, year partitions, append only)
  -> load(as_of) -> split_relisted -> Panel (T x N float64 arrays)
  -> universe mask, FeatureSpec DAG (+ leakage audit), forward labels
  -> purged splits -> ridge per split -> out-of-sample scores -> rank weights
  -> run_backtest (reference | numpy | numba) -> metrics on OOS rows
  -> tracker run directory -> qrp runs list | show | compare
```

Storage is long parquet via polars. Compute is wide numpy arrays of shape (dates, symbols), with NaN for "no bar". I kept those two worlds separate on purpose: polars is good at partition pruning and the point-in-time group-by, numpy is good at rolling windows and cross-sectional ops over a whole block. DESIGN.md has the full diagram and the list of invariants with the test that enforces each one.

### Implementation

Data. api.binance.com returns HTTP 451 from the US and Yahoo returned HTTP 429, but data.binance.vision is a public S3 bucket with one zip per symbol and month. The ingest command lists all USDT pairs through the S3 XML listing (paginated), drops leveraged tokens and stablecoin pairs, lists each symbol's archived months, keeps the 300 with the most months between 2022-01 and 2026-08, and fetches the zips with 32 threads. 15,400 files, 466,274 bars, 136 seconds, 13 MB of parquet.

Store. `BarStore.ingest` validates (no duplicate keys, high >= low, positive close), stamps `ingested_at` and an ingest id, writes one parquet file per year partition, writes a symbol metadata file for that ingest, and appends to the ingest log. `load` scans with hive partitioning, prunes years, applies the as_of filter, and keeps the last version per (symbol, date). `snapshot_id(as_of)` hashes the visible ingest ids, and every run records it, so a run says exactly which data it saw.

Features. Rolling windows use masked cumulative sums. I cumsum the values with NaNs replaced by zero and separately cumsum a finite mask, and a window is valid only when its finite count equals the window length. That gives trailing mean and std over the whole (T, N) block in a few vectorized passes with no NaN poisoning. Cross-sectional rank is a per-row argsort. Specs come from TOML, get topologically sorted, and each op is a plain function. The three leaky ops live in a separate registry so tests and the demo config can use them.

Leakage audit. Four cut points spread across the sample, two probes each, bit-for-bit comparison of rows <= cut with NaN equal to NaN. The research pipeline runs the audit on the exact spec of each run before doing anything else and raises `LookAheadError` if a feature fails. The failed run is still written to the tracker with status "failed".

Backtester. Three kernels implement one recursion. The reference kernel is plain Python with a cash balance and share counts per asset. The numpy kernel loops over days and vectorizes over (strategies, assets). The numba kernel is a compiled triple loop with `prange` over strategies. Missing bars earn zero (forward-filled price) and cannot be traded, so the target for that asset is replaced by its drifted weight. Execution delay is a shift of the target array. Negative delay raises.

CV and model. `purged_splits` produces walk-forward (expanding or rolling) or k-fold splits, with purge on the label interval and an embargo band after each test block. `check_split` verifies a split against the contract and is used by the tests. The model is closed-form ridge on pooled (date, symbol) rows of cross-sectionally ranked features, predicting the cross-sectional rank of the forward return. Scores become dollar-neutral weights proportional to the demeaned score rank, gross exposure 1, optionally held for k days.

Tracker. A run is a directory named timestamp, name and config hash, holding the resolved config, metadata (git sha, versions, data snapshot, duration, status), metrics and artifacts. `qrp runs compare` prints metrics side by side and only the config keys that differ, which is the view I actually want when comparing two runs.

### Problems

These are the actual bugs and surprises from this session, in the order I hit them.

1. The obvious data sources were blocked. api.binance.com returned HTTP 451 (the US geo-block) and Yahoo's chart API returned HTTP 429 on the first request. I switched to data.binance.vision, the public S3 bulk archive, which serves monthly zips and has no geo-block.

2. Non-ASCII tickers crashed the first ingest. The S3 listing contains a few symbols with Chinese names, and urllib raised `UnicodeEncodeError: 'ascii' codec can't encode characters` when it built the URL. I now keep only ASCII alphanumeric symbols, since those are out of scope for a v0 universe anyway.

3. Binance changed its timestamp unit. Spot archive files from 2025-01-01 onward use microseconds instead of milliseconds. Read naively, 2025 bars would land in the year 57000. `_to_millis` treats any epoch above 1e14 as microseconds, and a test pins the conversion for a 2025-01-01 timestamp in both units.

4. My own drift test was wrong, and the engine was right. I expected that targets equal to buy-and-hold drift weights would need zero turnover after the first trade. With a 15 bps cost, day 2 still traded 0.15% of equity. The reason is that the entry fee is paid from cash, so after the first trade the book holds 100% of pre-trade equity in assets against slightly less than 100% of post-trade equity, which is a small leverage. A 100% invested target must sell a little to get back to 100%, and each later correction is the fee on the previous one, so it decays geometrically. The test now asserts zero turnover without costs, and turnover of about 15 bps on day 2 with costs.

5. I got the arithmetic of a hand-built test wrong (the position on a halted day is 0.525, not 0.55, because day 1 had already rebalanced back to 50% of equity 1.05). Writing down the expected values by hand for tiny cases was still worth it, since that is where the delever effect above showed up.

6. The LUNA relisting blew up a backtest. The first purged k-fold run went to exactly zero equity on 2022-05-31 with a daily turnover above 2, which should be impossible for a gross-1 book. LUNAUSDT closed at 0.00005 on 2022-05-13, had no bars for 18 days, and came back at 8.87 on 2022-05-31 as LUNA 2.0 under the same ticker. The backtester forward fills prices through missing bars, so a short LUNA position that was stuck (the asset was untradeable) saw a return of about 177,000x in one day. The walk-forward runs never noticed because their test periods start in 2023. Eight symbols in the data have gaps over 3 days (FTT 311 days, CVC 154, KEY 28, LUNA 18, VIDT 9, STRAX 8, BNX 6, QUICK 4). The v0 fix is `split_relisted`, which makes any gap over 3 days start a new instrument (`LUNAUSDT#2`), so the old one is delisted at its last real price. That took the panel from 300 to 308 instruments, and a regression test shows the same short position going bankrupt on the raw data and surviving after the split.

7. A high IC with a low Sharpe looked suspicious. The baseline ridge had an out of sample IC of 0.125, while every single feature I would have guessed as the driver (1 day reversal, 30 day momentum) had an IC near zero or negative. Before trusting it I added a random-score negative control (IC 0.0005, Sharpe -1.34, as expected) and computed single-feature ICs. The driver is 30 day volatility, with a rank IC of -0.13: high-volatility coins underperformed. The Sharpe is much lower than the IC suggests because the long-short book on crypto has large idiosyncratic volatility and the weekly holding period only uses part of the signal.

8. The machine was shared. Ten or more other agents were building projects at the same time and the load average was between 220 and 400 on 14 cores for the whole session. A single research run took about a minute of wall time for about 6 seconds of CPU time, and the experiment suite hit the 10 minute tool timeout on its first attempt, so I reran it in the background. Throughput numbers below are therefore lower bounds, and the multi-threaded numba numbers in particular are noisy (at some sizes the multi-threaded run was slower than the single-threaded one). I recorded the load average before and after in `results/bench_backtest.json`.

9. Profiling showed the forward fill cost more than the kernel. For one strategy on 300 assets, `prepare_market` (forward fill plus returns) took about 18 ms of a 23 ms call under load. Its pieces measured 2.3 ms for `np.maximum.accumulate` and 3.8 ms for `take_along_axis` when timed alone. Rather than micro-optimize, `run_backtest` now accepts a precomputed market, which is the right shape for sweeps over one panel anyway, and a test checks the result is identical.

### Experiments

All experiments ran on the ingested snapshot `417cbe2508ff`: 1,704 daily rows from 2022-01-01 to 2026-08-31 and 308 instruments (300 symbols before the relisting split). The base config is `configs/xs_ridge.toml`.

Base setup. The universe is the top 100 instruments by trailing 30 day mean dollar volume with at least 60 bars of history, recomputed every day. The model is ridge (alpha 10) on the cross-sectional ranks of 7 features (1 day return, 7, 30 and 90 day momentum, 30 day volatility, 20 day moving average ratio, 20 day volume z-score), predicting the rank of the 5 day forward return. CV is walk-forward with 8 test blocks after a 365 day minimum training period, purge on the label interval and a 5 row embargo. The portfolio is dollar neutral with weights proportional to the demeaned score rank, gross 1, rebalanced every 5 days, executed with a 1 day delay, 10 bps fee plus 5 bps slippage. Metrics use only out of sample days (1,338 days from 2023-01-02).

Each variant changes one thing through `--set` style overrides, and each is a tracked run. The script is `scripts/run_experiments.py`, the log is `results/experiments_log.txt`, the table is `results/experiments.csv` (with run ids), and `results/experiments.json` has the same rows plus the snapshot id.

1. Costs. 0, 5, 15 (baseline), 25 and 50 bps one way, plus square-root impact with coefficient 0.5 at 50M AUM.
2. Rebalance frequency. Daily instead of every 5 days.
3. Execution timing. delay 0 (trade at the close the signal observed) against delay 1.
4. Controls. A random score (negative control), and two single-signal baselines with no model (short 1 day winners, long 30 day momentum).
5. Leakage. Adding one leaky feature to the model with the audit's fail switch turned off, either a 5 day forward return or an 11 day centered moving average ratio. Separately, `qrp audit configs/leaky_demo.toml` on the real data (`results/leakage_audit.txt`).
6. Purging. A 20 day label horizon (so overlap is large) under walk-forward and k-fold, each with purge plus a 10 row embargo against no purge and no embargo.
7. Throughput. `scripts/bench_backtest.py --real` on synthetic panels with T = 1,825 days, N in {100, 300, 1000} and S in {1, 16, 128} strategies per call, for the numpy and numba kernels (1 thread and 14 threads), plus the reference engine at a small size, the numba kernel with a precomputed market, and the real 1,704 x 300 panel. Timings are best of 5. Output is `results/bench_backtest.csv`, `.json`, `.png` and `results/bench_log.txt`.

### Results

Leakage (from `results/experiments.csv` and `results/leakage_audit.txt`). The audit flagged exactly the 3 leaky features in `leaky_demo.toml` on real data and passed the 3 honest ones. With the fail switch off, the 5 day forward return feature turned an out of sample Sharpe of 0.49 into 19.40 (IC 0.724, annual return 9,161%), and the centered moving average, which only sees 5 days ahead through a smoothing window, gave a Sharpe of 13.33 and an IC of 0.438. Both have a max drawdown under 3.5%. That is the signature of a leak, and it is why the pipeline refuses to run on a spec that fails the audit.

Baseline (from `results/experiments.csv`). Out of sample Sharpe 0.49, annual return 7.3%, annual volatility 17.6%, max drawdown -29.7%, IC 0.125, average daily turnover 15.9%, annual cost drag 8.7%.

Costs (from `results/experiments.csv`, plotted in `results/cost_sensitivity.png`). Sharpe falls almost linearly with cost: 0.99 at 0 bps, 0.82 at 5, 0.49 at 15, 0.16 at 25 and -0.66 at 50. Break-even is around 30 bps one way. Adding square-root impact at 50M AUM gave -1.08, so this signal has little capacity. Rebalancing daily instead of weekly doubled turnover (15.9% to 31.6% per day) and cut the Sharpe from 0.49 to 0.12, since the extra trades cost more than the fresher signal earned.

Timing (from `results/experiments.csv`). Trading at the same close the signal observed (delay 0) raised the Sharpe from 0.49 to 0.69 and the IC from 0.125 to 0.133. That is a 40% overstatement from one day of execution optimism.

Controls (from `results/experiments.csv`). The random score had IC 0.0005 and Sharpe -1.34, which is the cost of trading noise. Shorting 1 day winners alone had IC 0.000 and Sharpe -0.85, and 30 day momentum alone had IC -0.049 and Sharpe -0.32. The model's edge comes from 30 day volatility (single-feature IC -0.13, measured while debugging problem 7 and not saved to a results file).

Purging (from `results/experiments.csv`). With a 20 day horizon, purging changed almost nothing: walk-forward Sharpe 0.842 purged against 0.847 unpurged, k-fold 0.850 against 0.844, ICs within 0.002. This is a real negative result. A 7 coefficient ridge trained on roughly 35,000 to 150,000 pooled rows (365 to 1,500 training days times 100 names) cannot memorize the roughly 20 overlapping rows per boundary, so the leak purging removes is too small to show up. Purging is still on by default, because the same pipeline will run higher-capacity models in project 09, where overlap does matter.

Throughput (from `results/bench_backtest.csv` and `results/bench_log.txt`, load average about 230 to 360 during the run). The pure-Python reference engine ran 1,386 strategy-days per second at N = 100 (139 thousand asset-days per second). With the market precomputed, the single-threaded numba kernel ran 772 thousand strategy-days per second at N = 100, 270 thousand at N = 300 and 79 thousand at N = 1000, which is a steady 77 to 81 million asset-days per second, about 560 to 585 times the reference engine per asset-day. The best batched number was 2.12 million strategy-days per second (14 threads, S = 16, N = 100). On the real 1,704 x 300 panel, 64 random strategies took 0.74 s, 147 thousand strategy-days per second, and one strategy took 10.8 ms. numpy was between 1.2 times (S = 1, N = 1000) and 20 times (S = 1, N = 100) slower than single-threaded numba. At N = 1000 and S = 128, throughput dropped to 43 thousand strategy-days per second single threaded, because the call copies the 1.9 GB (S, T, N) target array twice before the kernel runs (lagging and NaN cleaning).

Tests. 73 tests pass (`uv run pytest`).

### What I would change

- Lag and clean the targets inside the kernel instead of materializing two copies of the (S, T, N) array. At S = 128 and N = 1000 those copies dominate, and they are why throughput falls at the largest sizes.
- Rerun the benchmark on an idle machine. Everything here was measured at a load average between 220 and 400, so the multi-threaded scaling numbers in particular are not trustworthy.
- Replace the gap rule for relisted tickers with a real symbol mapping and corporate actions table, and close delisted positions at a delisting price rather than holding them at the last price forever.
- Build the universe from point-in-time exchange listings instead of "the 300 pairs with the most months in the archive", which is chosen with today's knowledge.
- Show purging mattering. The ridge was too low-capacity for label overlap to change anything. A gradient-boosted model or a small sample would make the effect visible, and that is the experiment worth adding when project 09 brings nonlinear models.
- Save single-feature ICs as a standard artifact of every run. I computed them by hand while debugging, and they explained the baseline faster than anything else.
- Make the leakage audit cheaper for big specs by caching the unchanged prefix. It recomputes every feature 9 times, which is fine for 14 features on daily data but will not be for minute bars.

### Verification

Independent verification on 2026-09-26 (evening, US Eastern), by a reviewer who did not build the project. Threads were capped at 2 (OMP, OpenBLAS, vecLib and numba) because the machine was shared.

What I reproduced:

- Clean state: removed `__pycache__` and `.pytest_cache`, kept the uv environment, ran `uv sync` and `uv run pytest`. 73 passed in about 2 seconds.
- Data: `./scripts/download_data.sh` worked as written. It fetched 15,400 files with none missing and wrote 466,274 rows for 300 symbols in 111 seconds, the same row count as the original ingest. The snapshot id differs (`f2322f2cf2d1` against `417cbe2508ff`) because it hashes the new ingest id, which is expected.
- Experiments: I reran `scripts/run_experiments.py` in its full configuration with the output directory redirected to a scratch folder, so the committed `results/` files were not overwritten. All 17 rows matched `results/experiments.csv` exactly (maximum absolute difference 0.0 for Sharpe, annual return, volatility, drawdown, IC, turnover, cost drag and day count). Baseline Sharpe 0.490, IC 0.1253, zero cost Sharpe 0.988, delay 0 Sharpe 0.692, random control Sharpe -1.339, leaky forward return Sharpe 19.40, purged against unpurged walk-forward 0.842 against 0.847. `results/runs_compare.txt` also matched apart from the run id timestamps.
- `uv run qrp audit configs/leaky_demo.toml` printed the same six lines as `results/leakage_audit.txt` and exited 1.
- The README research commands (`qrp info`, `qrp info --as-of 2023-06-01`, `qrp run`, `qrp run --set ...`, `qrp runs list --sort sharpe`, `qrp runs compare`, `qrp runs show`) all worked. The `--as-of 2023-06-01` read correctly shows 0 visible symbols, because the data was ingested in 2026.
- Every number in Experiments and Results is present in `results/experiments.csv`, `results/experiments.json`, `results/leakage_audit.txt` or `results/bench_backtest.csv`, except the single-feature IC of -0.13 for 30 day volatility, which the text already says was not saved.

What I deferred:

- Throughput. The load average was between 7 and 60 during verification, so I did not rerun the full benchmark. `scripts/bench_backtest.py --quick` ran cleanly into a scratch folder, which shows the script works and times a real call after JIT warmup. Checking the reported strategy-days per second belongs on a quiet machine.

Discrepancies and fixes:

- Fixed: Experiments said the OOS period started on 2023-01-05. The tracked equity artifact starts on 2023-01-02 (first test row 2023-01-01 plus the 1 day delay), which is also what 1,338 days to 2026-08-31 implies.
- Fixed: Experiments said "308 instruments (300 symbols after the relisting split)". It is the other way round, 300 symbols before the split and 308 instruments after.

Review notes (not fixed, recorded for later):

- The walk-forward OOS period (2023 to 2026-08) is reused by all 17 experiments and was inspected while debugging (problem 7). That is fine for a platform comparison, but none of these Sharpe ratios is a one-shot holdout result and they should not be quoted as one.
- The known selection biases stand as documented: the 300 pairs were chosen with today's knowledge of archive history, and delisted assets are held at their last price instead of being closed.
- The accounting and leakage tests are genuine. The vectorized engines are compared against an independent share and cash ledger, and the leaky ops really fail the audit, including a one bar lead hidden in `lag`.
