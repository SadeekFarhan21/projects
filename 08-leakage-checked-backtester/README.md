# qrp, a quant research platform (project 08)

qrp is a small daily-bar research platform built for correctness first. It downloads free daily bars for 300 Binance USDT spot pairs, stores them as partitioned parquet with a point-in-time read path, computes features from a declarative spec, audits those features for look-ahead, fits models with purged walk-forward cross-validation, backtests target weights with drift, costs and impact, and records every run in a local tracker you can query from the command line.

Project 09 builds on this platform.

## What is in v0

- Ingestion from the public Binance archive (data.binance.vision), 300 USDT pairs, daily bars from 2022-01-01 to 2026-08-31, including 83 pairs that were delisted inside that window.
- An append-only parquet store partitioned by year, with `available_at` and `ingested_at` stamps on every row. `load(as_of=...)` returns exactly what was known at that instant, including restated bars.
- A feature library of 9 causal ops (returns, momentum, volatility, moving average ratio, volume z-score, rolling z-score, lag, cross-sectional rank and z-score) declared in TOML as a DAG, plus 3 deliberately leaky ops that exist so the leakage audit has something to catch.
- An empirical leakage audit (truncation and perturbation probes) that runs before every research run and aborts it if any feature peeks.
- A target-weight backtester with exact drift and cost accounting, a linear fee plus slippage model, optional square-root impact, execution delay, and three engines (pure-Python share/cash oracle, numpy, numba) that agree to 1e-10.
- Walk-forward and k-fold splits with purging and embargo on the label interval.
- A run tracker (`runs/<id>/config.json, meta.json, metrics.json, artifacts/`) with `qrp runs list | show | compare`.
- A throughput benchmark and an experiment suite whose raw outputs are in `results/`.

See [DESIGN.md](DESIGN.md) for the architecture, invariants and trade-offs, and [DEVLOG.md](DEVLOG.md) for how it was built and what went wrong.

## Build

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.12.

```bash
cd projects/08-leakage-checked-backtester
uv sync
```

## Get the data

About 13 MB, 15,400 small zip files, roughly two to three minutes. The data directory is gitignored.

```bash
./scripts/download_data.sh          # same as: uv run qrp ingest --root data
uv run qrp info                     # ingest log and symbol metadata
uv run qrp info --as-of 2023-06-01  # the same, point in time
```

## Test

```bash
uv run pytest
```

The suite (73 tests, a few seconds once numba has cached its kernels) covers look-ahead detection, backtester accounting identities, cross-engine agreement, purging and embargo, the point-in-time store, and the pipeline end to end. Tests use synthetic data and do not need the download.

## Run research

```bash
uv run qrp audit configs/leaky_demo.toml           # flags the 3 leaky features, exits 1
uv run qrp run configs/xs_ridge.toml               # tracked run, prints OOS metrics
uv run qrp run configs/xs_ridge.toml --set costs.fee_bps=0 --set portfolio.rebalance_every=1
uv run qrp runs list --sort sharpe
uv run qrp runs compare <run_id_a> <run_id_b>      # metrics side by side + differing config keys
```

## Benchmark and experiments

```bash
uv run python scripts/bench_backtest.py --real     # results/bench_backtest.{csv,json,png}
uv run python scripts/run_experiments.py           # results/experiments.{csv,json}, plots, runs_compare.txt
```

## Headline results

All numbers come from runs in this session on an Apple M4 Pro (14 cores) that was shared with other heavy jobs, so throughput numbers are lower bounds. Sources are in `results/`.

| Result | Value | Source |
|---|---|---|
| Tests | 73 passed | `uv run pytest` |
| Backtest throughput, numba kernel, 1 thread, market precomputed | 77 to 81 million asset-days/s (270k strategy-days/s at N = 300) | `results/bench_backtest.csv` |
| Best batched throughput (14 threads, S = 16, N = 100) | 2.12 million strategy-days/s | `results/bench_backtest.csv` |
| Real panel (1,704 days x 300 pairs), 64 strategies in one call | 0.74 s, 147k strategy-days/s | `results/bench_backtest.csv` |
| Pure-Python reference engine (the oracle) | 1,386 strategy-days/s at N = 100 | `results/bench_backtest.csv` |
| Baseline ridge, out of sample 2023-01 to 2026-08, 15 bps costs | Sharpe 0.49, IC 0.125, max drawdown -29.7% | `results/experiments.csv` |
| Same with zero costs | Sharpe 0.99 | `results/experiments.csv` |
| Same with delay 0 (trading at the signal's own close) | Sharpe 0.69 | `results/experiments.csv` |
| Random score control | Sharpe -1.34, IC 0.0005 | `results/experiments.csv` |
| One leaky feature added, audit switched off | Sharpe 19.40 (forward return), 13.33 (centered average) | `results/experiments.csv` |
| Leakage audit on `configs/leaky_demo.toml` | 3 of 3 leaky features flagged, 3 of 3 honest ones pass | `results/leakage_audit.txt` |
| Purged vs unpurged CV, 20 day horizon | Sharpe 0.842 vs 0.847 walk-forward, 0.850 vs 0.844 k-fold (no measurable effect for this model) | `results/experiments.csv` |

Plots are `results/bench_backtest.png`, `results/experiments_equity.png` and `results/cost_sensitivity.png`. `results/runs_compare.txt` is verbatim `qrp runs compare` output.

## Layout

```
src/qrp/
  data/binance.py      archive listing, parallel download, timestamp fixes
  data/store.py        BarStore, append-only parquet, point-in-time reads
  panel.py             wide (T, N) arrays, relisting split, synthetic data
  features/ops.py      causal and leaky feature kernels
  features/spec.py     FeatureSpec DAG, leakage audit
  cv.py                purged walk-forward and k-fold splits
  backtest/engine.py   reference, numpy and numba kernels
  backtest/metrics.py  Sharpe, drawdown, turnover, cost drag
  research.py          config -> universe -> features -> model -> backtest -> run
  tracker.py           run directories, list and compare tables
  cli.py               qrp command
configs/               research configs (TOML)
scripts/               download, benchmark, experiment suite
tests/                 pytest suite
results/               raw outputs of the benchmark and experiments
```

## Milestones

| Milestone | Status |
|---|---|
| v0: ingestion, PIT parquet store, feature DAG, leakage audit, vectorized backtester, purged CV, run tracker, benchmark | done |
| Survivorship-free universes (point-in-time exchange listings, delisting returns instead of last price) | planned |
| Corporate actions and symbol mapping (redenominations, ticker reuse without the gap heuristic) | planned |
| Intraday data (minute klines, session-aware features, intraday execution) | planned |
| Distributed parameter sweeps (process pool or Ray over configs, sweep-level tracking) | planned |
| Notebook-friendly API (fluent Python API over the same stages, cached panels) | planned |

## Cut from v0

- Minute bars. The task listed them as optional. The store and panel are frequency-agnostic (`freq=` partition), but the ingest command only fetches `1d`, and features assume one row per day.
- US equities. Yahoo returned HTTP 429 from this machine and api.binance.com returns 451 from the US, so v0 uses the public Binance archive only.

## Known limitations

- Throughput falls at the largest sizes (S = 128, N = 1000) because `run_backtest` copies the (S, T, N) target array twice before the kernel. Benchmarks were run at a load average of 230 to 360 on a shared machine, so multi-threaded scaling numbers are noisy.

- The universe is the 300 pairs with the most archived months in the window, chosen with today's knowledge. The point-in-time universe filter inside that set is causal, but the set itself has mild selection bias. Survivorship-free universes are the next milestone.
- Delisted assets are held at their last price with zero return, not closed at a delisting price.
- Ticker reuse is handled by a gap rule (more than 3 days without a bar starts a new instrument). A redenomination without a trading pause would still produce a fake return.
- Daily returns above 100% remain in the data (50 of them), some are real pumps and some may be data artifacts.
