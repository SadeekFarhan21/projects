# Can We Actually Predict Markets?

Project 09 in the Trading thread. A pre-registered, end-to-end alpha research
study on free daily crypto data: does cross-sectional short-term reversal exist
among the 50 most liquid Binance USDT pairs, and can any simple model turn it
into money after trading costs?

The short answer from the one-shot holdout (2024-07 to 2026-08):

| Question | Verdict | Key number | Source |
|---|---|---|---|
| H1: does 1 day reversal predict next day cross-sectional ranks? | supported | mean rank IC 0.019, Newey-West t 2.24 | `results/holdout/holdout_report.json` |
| H2: does the dev-selected portfolio make money after 15 bps costs? | not supported | net Sharpe 0.08, PSR 0.55 | `results/holdout/holdout_report.json` |
| Does the reversal signal itself make money? | no | rev_1d gross Sharpe -0.78, net -4.06 | `results/holdout/holdout_report.json` |

The reversal is real in ranks and useless in dollars. The walk-forward
selection rule also picked the wrong model: the ridge and gradient boosting
models, which had far higher IC, did better in the holdout, but that is a
post-hoc observation and is not claimed as a result. See `DEVLOG.md`.

### What is in here

```
PREREGISTRATION.md      hypothesis, universe, periods, trial list, frozen before any data
DESIGN.md               architecture, data structures, invariants, trade-offs
DEVLOG.md               case-study notes: theory, problems hit, every number with its source
src/marketpred/
  data.py               Binance archive downloader + load_panel() (thin, replaced by project 08 later)
  universe.py           point-in-time top-50 liquid universe, exclusion list
  features.py           relisting split, 9 features, cross-sectional ranks, targets
  models.py             single factors, ridge, HistGradientBoosting; the pre-registered registry
  portfolio.py          rank and quintile dollar-neutral weights, backtest with costs
  metrics.py            rank IC, Newey-West t, Sharpe, PSR, deflated Sharpe
  walkforward.py        half-year folds with a 5 day embargo
  study.py              evaluate one model under both constructions and all cost levels
scripts/
  download_data.py      fetch about 27k monthly zips (~25 MB parquet after parsing)
  run_dev.py            development walk-forward over all 18 trials, freezes the selection
  run_holdout.py        one-shot holdout, refuses to run twice (results/holdout/LOCK.json)
  plot_posthoc.py       dev vs holdout Sharpe scatter from saved CSVs
  benchmark.py          timing of each pipeline stage
tests/                  25 pytest tests, synthetic data only (no network)
results/                every number quoted in the docs, as JSON/CSV/PNG
```

### Build

Requires [uv](https://docs.astral.sh/uv/). The project is pinned to Python 3.12.

```
uv sync
```

### Test

```
uv run pytest -q
```

Tests use synthetic panels, so they run without downloading anything. They
cover timestamp parsing (ms and us), exclusion rules, point-in-time universe,
feature look-ahead, forward return alignment, portfolio weights and PnL
accounting, the Newey-West/PSR/DSR math (including a Monte Carlo check of the
expected max Sharpe), fold construction and embargo, and an end-to-end check
that a planted reversal is recovered while pure noise is not.

### Run the study

```
uv run python scripts/download_data.py     # about 7 minutes, writes data/processed/klines_1d.parquet
uv run python scripts/run_dev.py           # about 4 minutes, writes results/dev/
uv run python scripts/run_holdout.py       # one shot; refuses if results/holdout/LOCK.json exists
uv run python scripts/plot_posthoc.py
```

The holdout has already been evaluated and the lock file is committed. Running
`run_holdout.py` again exits with an error, which is the point.

### Benchmark

```
uv run python scripts/benchmark.py         # writes results/benchmark.json
```

Median of 3 runs on an M4 Pro (shared with other jobs, so treat as rough):
building the full dev feature panel takes 11.1 s, a ridge fit on 75k rows takes
0.01 to 0.19 s, a depth 3 gradient boosting fit takes 3.0 s, and a backtest of
one strategy takes 0.12 s. The full 18 trial walk-forward is dominated by
gradient boosting (76 s and 96 s for the two configs, `results/dev/timings.json`).

### Status

| Milestone | Status |
|---|---|
| v0: pre-registered daily cross-sectional study, walk-forward, costs, DSR, one-shot holdout | done |
| v0.1: fresh holdout for the ridge/GBM family (needs new data after 2026-08, or a new asset class) | planned |
| Model borrow/funding costs for the short leg (perpetual funding rates) | planned |
| Intraday signals (hourly klines, same pipeline at higher frequency) | planned |
| Alternative data (funding rates, open interest, on-chain flows) | planned |
| Regime analysis (bull/bear, BTC volatility regimes, IC conditional on regime) | planned |
| Port data layer onto project 08 (replace `data.load_panel`) | planned |

### What was cut from v0

- Short-side costs. Spot shorting is not possible on Binance spot, so a real
  implementation would use perpetual futures and pay or receive funding. This
  is not modeled. It would make the short leg more expensive in bull markets.
- Weight drift. Turnover is computed from target weights day to day, ignoring
  intraday drift, as pre-registered. This slightly understates turnover.
- Probability of backtest overfitting (CSCV). The deflated Sharpe ratio is
  implemented; PBO was left for later.

### Known issues

- In `holdout_report.json`, the `*_lag1` blocks report the IC of the unlagged
  forecast (the IC fields are copied from the lag 0 evaluation). Their Sharpe
  and return fields are correctly lagged. Fixing this would require rerunning
  the holdout, which the protocol forbids, so it stays documented instead.
- `results/dev/trials.jsonl` has 32 lines, not 18: the first development run
  hung inside gradient boosting after logging 14 trials (see DEVLOG). The
  deflated Sharpe uses 32 as the trial count, which is conservative.
- Survivorship is reduced (delisted pairs are in the archive) but not removed:
  pairs that never made it onto Binance are absent by construction.
