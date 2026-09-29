# optvol: options pricing, volatility surfaces and hedging on Deribit

Black-76 pricing and Greeks, a vectorized implied vol solver, Deribit chain cleaning with put-call parity forwards, raw SVI and SSVI surface fits with static no-arbitrage checks, a comparison against Deribit's mark IV, and a minute-level delta hedging experiment with P&L attribution. Everything runs on one real day of Deribit BTC and ETH option quotes (the tardis.dev free sample for 2026-09-01).

### Headline numbers (2026-09-01, all from files under results/)

| What | Result | File |
|---|---|---|
| IV solver throughput, 2M synthetic options | 2.07M options/s on 1 thread, 4.07M on 2 | `01_iv_benchmark.json` |
| IV round trip, max price error | 1.1e-15 of F (target 1e-10) | `01_iv_benchmark.json` |
| Greeks vs central differences | pass at 1e-6 relative, 300 hypothesis cases | `tests/test_black76.py` |
| SVI fit RMSE, median over 501 slices | 0.29 vol points vs 0.68 median half spread | `02_surface_summary.json` |
| Slices with SVI RMSE below their half spread | 95.2% | `02_surface_summary.json` |
| Slices passing butterfly and calendar checks | 96.4% | `02_surface_summary.json` |
| SSVI global fit RMSE, median | 1.89 vol points, 100% butterfly free | `02_surface_summary.json` |
| Own mid IV minus Deribit mark IV, OTM quotes | median 0.012 vp, MAD 0.091 vp, 99.1% inside half spread | `03_mark_iv_summary.json` |
| Hedged P&L std, same day options to expiry | 2.5 bp (1 min) vs 10.2 bp unhedged | `04_hedge_summary.csv` |
| Hedged P&L std, next day expiry, 4 h windows | 3.9 bp (1 min), 4.8 bp (30 min) vs 14.9 bp unhedged | `04_hedge_summary.csv` |

### Layout

```
src/optvol/
  black76.py    Black-76 price and Greeks (numba, parallel), coin/USD conversion
  iv.py         vectorized implied vol: bounds, OTM transform, rational guess, safeguarded Newton, Brent fallback
  chain.py      load minute files, snapshot as of t, clean, parity forwards, attach IVs
  svi.py        raw SVI fit, SSVI fit, Gatheral-Jacquier butterfly g(k), calendar check
  surface.py    fit a whole snapshot (all expiries of a coin) and run the checks
  hedging.py    per-minute panels, SVI path, hedge ratios, hedging simulation, attribution, bootstrap
  data.py       paths and cached day loading
tools/minute_snap/   C++20 stream filter: 100 GB/day tardis CSV to one row per symbol per minute
scripts/download_tardis.sh   stream one sample day through minute_snap into data/raw/
scripts/run_all.sh           tests plus all four experiments, thread capped
experiments/01..04           the scripts behind every number in DEVLOG.md
tests/                       pytest plus hypothesis, 25 tests
results/                     raw outputs (CSV, JSON, PNG, logs)
```

### Setup

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), CMake 3.20+, Ninja, a C++20 compiler (Apple clang works), curl.

```
cd projects/19-deribit-svi-hedging
uv sync                      # creates .venv with Python 3.12 (pinned in .python-version)
```

### Data

```
scripts/download_tardis.sh 2026-09-01
```

This builds `tools/minute_snap` with CMake and Ninja if needed, then streams `https://datasets.tardis.dev/v1/deribit/options_chain/2026/09/01/OPTIONS.csv.gz` through `gunzip | minute_snap BTC- ETH- | gzip` into `data/raw/deribit_options_chain_20260901_1m.csv.gz` (147 MB). The raw file is never written to disk: it is 236M rows for the day, of which 142M are BTC and ETH, reduced to 2.3M minute rows over 1948 symbols. The download took about 20 minutes here. tardis.dev serves the first day of each month without an API key; availability was verified before building on it, so the public REST snapshot fallback was not needed.

### Test

```
uv run pytest -q             # 25 passed
```

### Run the experiments

```
scripts/run_all.sh           # tests, then experiments 01 to 04 in order, about 2 minutes
```

or one at a time, for example `uv run python experiments/04_hedging.py 2026-09-01`. The run script caps numba, BLAS and polars at 2 threads because the machine was shared; raise `NUMBA_NUM_THREADS` for the full throughput number.

### Benchmark

```
NUMBA_NUM_THREADS=2 uv run python experiments/01_iv_benchmark.py
```

prices, computes Greeks for, and inverts 2M options over log moneyness in [-1, 1], one hour to three years, and 2% to 300% vol, on 1 thread and on `NUMBA_NUM_THREADS` threads.

### Milestones

| Milestone | Status |
|---|---|
| Data pipeline: tardis.dev sample, C++ minute reducer, loader, 08:00 UTC expiry check | done |
| Black-76 pricer with delta, gamma, vega, theta, vanna, volga; coin premium conversion | done |
| Vectorized IV solver with bounds handling and Brent fallback, above 1M/s | done (2.07M/s single thread) |
| Chain cleaning, parity regression forwards, exact time to expiry | done |
| Raw SVI per expiry with constraints, butterfly and calendar checks, SSVI global fit | done |
| Own IV vs Deribit mark IV by moneyness with outlier decomposition | done |
| Delta hedging experiment: 3 hedge ratios x 3 intervals, attribution, bootstrap intervals | done (one day) |
| Tests: parity, IV round trip, Greeks vs finite differences, SVI recovery, arbitrage flags | done |
| Multi-day hedging study over many monthly sample days (more independent paths) | next |
| Options market-making simulator quoting around the surface | later |
| Pre-registered variance risk premium study | later |
| Heston via the COS method and Dupire local vol, compared with SVI | later |
| American options: binomial tree, Longstaff-Schwartz | later |
| C++ SIMD pricer and IV solver | later |
| Plug into project 06 as an options book and project 10 as an options market maker | later |

### What was cut from v0

- One day only. The hedging study has one price path per coin, so its bootstrap intervals describe dispersion across options and windows on that day, not across days. A second day (2026-08-01) download was started and stopped when the machine was being unloaded; `scripts/download_tardis.sh 2026-08-01 2026-07-01 ...` fetches more days and every experiment takes the day as an argument.
- Hedging is done in USD on a linear future. Deribit BTC and ETH options and futures are coin-margined (inverse), which adds a small convexity term; it is ignored here.
- Settlement uses a proxy: the mean of the expiry's underlying over 07:30 to 07:59, standing in for Deribit's 30 minute index TWAP.
- No transaction costs in the hedging study. Entries and exits are at mid; futures are hedged at the underlying price.
- The SVI fit is unconstrained in the butterfly sense (it only enforces the parameter bounds). The check flags violations but the fit does not repair them.
