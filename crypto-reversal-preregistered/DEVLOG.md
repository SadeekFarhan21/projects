# DEVLOG: Can we actually predict markets?

### What I wanted to build

A research pipeline I could defend in an interview line by line, applied to one
falsifiable question: among the 50 most liquid crypto pairs on Binance, does
yesterday's loser beat yesterday's winner tomorrow (short-term reversal), and
if so, is any of that left after paying to trade it?

I cared more about the process than the answer. The process I wanted:
write the hypothesis, universe, period split, trial list, costs and decision
rule into `PREREGISTRATION.md` before downloading a single price; do all model
selection in a walk-forward development period; count every trial; then open
a two year holdout exactly once and report whatever came out. A negative
answer was an acceptable outcome, and it is roughly what I got.

### Theory

Cross-sectional prediction. Each day I rank the coins in the universe by a
forecast and ask whether that ranking lines up with the ranking of next-day
returns. The daily Spearman correlation between the two is the information
coefficient (IC). A mean IC of 0.02 sounds tiny, but by Grinold's fundamental
law the information ratio scales like IC times the square root of breadth, and
with 50 names a day for 365 days a year the breadth is large. The catch is
that the fundamental law ignores costs.

Why reversal might exist. Short-term reversal in equities is usually explained
by liquidity provision: someone who needs to trade pushes the price, and the
counterparty who absorbs the flow is paid by the rebound. It is also partly
bid-ask bounce, which should be small here because Binance daily closes on
liquid pairs are last trades with spreads of a few basis points.

Costs. A daily-rebalanced long-short book with turnover T per day and a one-way
cost c loses T times c times 365 per year. At the pre-registered 15 bps and a
turnover of 1.3 (typical for a 1 day reversal signal) that is 71% per year,
which any gross edge must beat.

Statistics. Daily ICs are autocorrelated when signals persist (a 20 day
momentum signal barely changes day to day), so I use a Newey-West t-statistic
with 5 lags instead of the naive one. For portfolio returns I use the
probabilistic Sharpe ratio (PSR, Bailey and Lopez de Prado 2012): the
probability that the true Sharpe exceeds a threshold given the sample length,
skew and kurtosis. For the selection step I use the deflated Sharpe ratio
(DSR, 2014), which sets the threshold to the expected maximum Sharpe of N
worthless strategies:

    SR* = sqrt(Var[SR_trials]) * ((1 - gamma) * z(1 - 1/N) + gamma * z(1 - 1/(N e)))

where gamma is the Euler-Mascheroni constant. The more variants you try and
the more their Sharpes disagree, the higher the bar. I checked this formula
against a Monte Carlo of the max of 50 normals in `tests/test_metrics.py`.

### Architecture

A long panel of daily bars comes from Binance's public archive through a thin
data layer (`data.load_panel`, the one function project 08 will replace). The
panel is pivoted to wide date x symbol frames, split into listing episodes,
and turned into 9 features and a point-in-time universe mask. Features and the
training label are converted to cross-sectional ranks and stacked back into a
long `(date, symbol)` frame of universe members only. A walk-forward loop fits
one model per half-year fold with a 5 day embargo. Forecasts become
dollar-neutral weights (rank-weighted or top/bottom quintile), and a vectorized
backtest turns weights and next-day returns into gross PnL and turnover, from
which net PnL at any cost is one subtraction. `DESIGN.md` has the diagram and
the invariants, each tied to a test.

Two scripts carry the protocol. `run_dev.py` truncates the data at 2024-06-30
before anything is computed, evaluates all 18 pre-registered trials, appends
each to `results/dev/trials.jsonl`, and freezes the selection. `run_holdout.py`
writes `results/holdout/LOCK.json` (with the SHA-256 of the pre-registration)
before computing anything and refuses to run if the lock exists.

### Implementation

About 600 lines of library code, numpy/pandas/scikit-learn/scipy only.

- Data. 735 USDT symbols are listed in the archive; I download every monthly
  daily-kline zip from 2018-01 to 2026-08 (27,721 files, 400 s with 32
  threads, `results/download_log.txt`) and parse them into a 25 MB parquet.
  Delisted pairs are included, which is the main defense against
  survivorship bias.
- Universe. On each day: at least 60 bars of history, not a stablecoin, fiat,
  wrapped/staked duplicate, gold token or leveraged token, trailing 30 day
  median quote volume at least 1M USDT, and in the top 50 by that volume.
  Days with fewer than 20 names are skipped.
- Features. 1/5/20/60 day returns, 20 day realized volatility, log median
  dollar volume, Amihud illiquidity, distance from the 20 day high, 60 day
  beta to BTC. All become centered ranks in [-0.5, 0.5]; a member with a
  missing value gets 0.
- Models. Four single factors with no fitting (rev_1d, rev_5d, mom_20d,
  mom_60d), ridge at three alphas, and HistGradientBoosting at depth 3 and 6.
  Fitted models predict the rank of the next-day return.
- Backtest. Weights at the close of t, return from close t to close t+1,
  turnover as the sum of absolute weight changes, a missing next bar earns 0.
- Metrics. Everything in `metrics.py` is written out by hand (Newey-West,
  PSR, DSR) so I could test it directly.

### Problems

1. The Binance REST API returns HTTP 451 from the US. I switched to the static
   archive at data.binance.vision, which also turned out to be better for
   research because it keeps delisted pairs.
2. The archive changed its timestamp unit from milliseconds to microseconds
   in 2025. I knew about this before parsing, so `parse_kline_csv` checks the
   magnitude row by row; a test feeds it one row of each. The parsed panel
   ends on 2026-08-31 rather than somewhere in the year 50,000, which is the
   end-to-end check.
3. Ticker reuse. LUNAUSDT is the old Terra until May 2022 and the new Terra
   from late May 2022. A 60 day return computed across that gap would compare
   two unrelated assets. I split any gap longer than 7 days into a new
   episode (`LUNAUSDT#2`). In the development data this turned 507 symbols
   into 522 episodes (`results/dev/data_report.json`).
4. My first leveraged-token filter (base ends in UP/DOWN/BULL/BEAR) would
   have thrown out JUP. The rule now only fires when the stripped stem is
   itself a listed base, and the test includes JUP and SUPER.
5. pandas 3 changed `DataFrame.stack()` to keep NaN rows. My first version
   of the long frame would have included every (date, symbol) cell of the
   wide frames. I now select member rows explicitly from the mask.
6. The big one: the first development run hung. It logged 14 trials (all
   single factors and ridge) and then sat at 0.2% CPU for more than 10
   minutes. A standalone repro with `faulthandler` showed it stuck in
   `HistGradientBoostingRegressor` `_initialize_root` on 80k random rows. With
   `OMP_NUM_THREADS=1` the same fit finished in 3.4 s; with 4 or 14 threads
   it never finished within 60 s. sklearn ships its own libomp, and this
   machine was also running 14 other agents' jobs; I did not find the root
   cause. The fix is `threadpool_limits(1)` around every sklearn fit and
   predict. The cost of the hang is on the record: `trials.jsonl` has 32
   lines instead of 18, and the deflated Sharpe uses N = 32.
7. After the holdout ran I noticed that the lag 1 robustness blocks in
   `holdout_report.json` reuse the unlagged IC. Their Sharpe numbers are
   correctly lagged. I left the file as it is, because fixing it means
   rerunning the holdout.
8. The surprise that shapes the whole result: high IC did not mean high PnL,
   and the pre-registered selection rule picked the only trial family with a
   negative IC. Details below.

### Experiments

All runs are on the data described above. Development is 2020-01-01 to
2024-06-30 in 9 half-year walk-forward folds; the holdout is 2024-07-01 to
2026-08-31 (792 days). Costs are 15 bps per unit of turnover unless stated.
Sharpe ratios are annualized with 365 days.

Data checks (`results/dev/data_report.json`). The development universe had a
median of 50 names (minimum 26) on 1,643 days, drawn from 279 distinct
episodes. Only 58 of 79,190 member rows had no next-day bar. The largest
daily move in the universe was DOGE +392% on 2021-01-28, which is why every
feature and label is a rank.

Development walk-forward, all 18 trials (`results/dev/summary.csv`,
`results/dev/fold_ic.csv`):

| trial | mean IC | NW t | gross Sharpe | net Sharpe | turnover/day |
|---|---|---|---|---|---|
| rev_1d, rank | +0.036 | 6.98 | 0.15 | -2.86 | 1.33 |
| rev_5d, rank | +0.033 | 6.03 | -1.13 | -2.34 | 0.60 |
| mom_20d, quintile | -0.029 | -5.18 | 0.91 | 0.33 | 0.39 |
| mom_60d, rank | -0.049 | -8.51 | -0.03 | -0.42 | 0.19 |
| ridge alpha 10, rank | +0.097 | 17.29 | 0.24 | -0.92 | 0.56 |
| ridge alpha 10, quintile | +0.097 | 17.29 | 0.45 | -0.75 | 0.74 |
| gbm depth 3, rank | +0.108 | 21.38 | 1.49 | -0.17 | 0.72 |
| gbm depth 3, quintile | +0.108 | 21.38 | 1.64 | 0.06 | 0.89 |
| gbm depth 6, quintile | +0.099 | 20.77 | 1.77 | -0.08 | 0.99 |

Reversal ICs were positive in every one of the 9 folds for rev_1d and rev_5d,
and momentum ICs were negative in 8 of 9 (mom_20d) and 9 of 9 (mom_60d)
(`results/dev/fold_ic.csv`, plotted in `results/dev/dev_ic_by_fold.png`). The
ridge models put their largest weight on low volatility (vol_20d coefficient
about -0.08 in every fold), then on 1 day reversal (about -0.04)
(`results/dev/ridge_coefs.csv`).

Selection and multiple testing (`results/dev/selection.json`). The best dev
net Sharpe was mom_20d with quintile weights, 0.33, with PSR 0.75. Against
N = 32 logged trials and the observed spread of trial Sharpes, the deflated
Sharpe threshold is an annualized Sharpe of 2.01 and the DSR is 0.00018. In
plain words: the best development result is exactly what you would expect
from the luckiest of 32 useless strategies.

Holdout, run once (`results/holdout/holdout_report.json`, lock in
`results/holdout/LOCK.json`). Models refit on data up to 2024-06-25.

Benchmark (`results/benchmark.json`, median of 3, machine shared with other
jobs): building the dev dataset 11.1 s for 84,350 member rows; ridge fit
0.01 to 0.19 s on 75,250 rows; gbm depth 3 fit 3.0 s, depth 6 4.6 s;
daily IC over the whole dev set 2.5 s; one backtest 0.12 s. In the dev run the
two gbm walk-forwards took 76 s and 96 s of the roughly 4 minute total
(`results/dev/timings.json`).

### Results

H1, reversal as a signal: supported. The holdout mean rank IC of rev_1d is
+0.019 with a Newey-West t of 2.24 and a daily hit rate of 53.1%
(`results/holdout/holdout_report.json`). That clears the pre-registered bar of
t > 2, barely, and it is about half the development IC of 0.036
(`results/dev/summary.csv`).

Reversal as a strategy: dead. The same signal traded with rank weights has a
holdout gross Sharpe of -0.78 and a net Sharpe of -4.06, with turnover of 1.31
per day and a max drawdown of -86% (`results/holdout/holdout_report.json`).
Even at zero cost it lost money. The rank ordering is right slightly more
often than not, but the dollars go the other way. My working explanation,
which I have not tested: rank IC weights every coin equally, while PnL is
dominated by the few coins that move 20% in a day, and the coins that just
fell hard are also the ones most likely to keep falling hard.

H2, tradability of the selected strategy: not supported. mom_20d with quintile
weights earned a holdout gross Sharpe of 0.77 and a net Sharpe of 0.08, PSR
0.55, max drawdown -29% (`results/holdout/holdout_report.json`). Its holdout
IC was -0.004 (t -0.52), so it had no measurable rank skill; its gross PnL is
the right tail of a few trending coins. Net Sharpe falls from 0.77 at 0 bps
to 0.54 at 5 bps and -0.37 at 25 bps, and delaying execution by one day takes
it to -0.43 at 15 bps (`results/holdout/holdout_report.json`,
`results/holdout/holdout_cost_sensitivity.png`).

Post-hoc, and not a claim. Because the holdout script also evaluated every
other trial for context, I can see that the selection rule picked badly. The
gbm depth 3 quintile portfolio had a holdout net Sharpe of 0.71 and the three
ridge quintile portfolios about 0.55, with ICs near 0.10
(`results/holdout/all_trials_posthoc.csv`). The rank correlation between dev
and holdout net Sharpe across the 18 trials is 0.46
(`results/holdout/holdout_report.json`, plotted in
`results/holdout/dev_vs_holdout_sharpe.png`). It is tempting to say "the ML
model works". I cannot: I have now looked at the holdout, and picking the
winner after the fact is exactly the multiple-testing error the protocol
exists to prevent. The honest status is "a hypothesis for the next fresh
holdout", and even that would have to survive short-side funding costs,
which are not modeled at all.

So: can we actually predict markets? At the level of daily cross-sectional
ranks in liquid crypto, yes, a little, and the ICs are stable across nine
half-years and a separate two-year holdout. Turning that into money after 15
bps per trade did not happen in the pre-registered test.

### What I would change

- Select on IC, or on a cost-aware score that uses IC and turnover, instead of
  dev net Sharpe. The IC ranking of trials was stable from dev to holdout; the
  net Sharpe ranking was noisy, and the rule chose a strategy whose only merit
  was a lucky right tail.
- Reduce turnover inside the portfolio, not only in the evaluation: trade
  toward the target weights with a no-trade band, or smooth the forecast over
  a few days. The gbm models had gross Sharpes near 1.5 to 1.8 in development
  and lost most of it to costs.
- Keep an untouched second holdout (or wait for new data) so a post-hoc
  finding like the gbm result can be tested properly.
- Model funding for the short leg with perpetual futures data, and weight
  drift in the turnover calculation.
- Replace the dollar volume top 50 with a smaller, stricter universe; the
  names ranked 30 to 50 are where both the tails and the costs live.
- Fix the lag 1 IC bookkeeping before the next holdout, and add a test that
  the lag option changes every lag-dependent field.

### Verification

Independent verification on 2026-09-26 (evening, US Eastern), by a reviewer who did not build the project. Threads were capped at 2 (OMP, OpenBLAS, vecLib) because the machine was shared.

What I reproduced:

- Clean state: removed `__pycache__` and `.pytest_cache`, kept the uv environment, ran `uv sync` and `uv run pytest -q`. 25 passed.
- Data: `uv run python scripts/download_data.py --workers 8` reused the cached zips in `data/raw`, relisted the archive (735 USDT symbols) and rewrote `data/processed/klines_1d.parquet`. The new file is identical to the old one (829,342 rows, 734 symbols, equal after sorting).
- Development study: I reran `scripts/run_dev.py` in its full configuration with the output directory redirected to a scratch folder, seeded with the first 14 lines of `results/dev/trials.jsonl` so the trial count reproduces the original 32. `summary.csv`, `fold_ic.csv`, `ridge_coefs.csv` and `daily_net_returns.csv` matched the committed files exactly (maximum absolute difference 0.0), `data_report.json` was byte identical, and the selection was the same: mom_20d with quintile weights, dev net Sharpe 0.325, PSR 0.754, deflated Sharpe 0.00018 against a threshold of 2.01.
- Holdout: `scripts/run_holdout.py` refuses to run, as intended, because `results/holdout/LOCK.json` exists. To check the reported numbers without touching the protocol, I recomputed the same code path with the same frozen `selection.json` into a scratch folder. This makes no new decision; it only checks that the committed report is what the code produces. Every field of `holdout_report.json` and every cell of `all_trials_posthoc.csv` matched exactly: rev_1d IC 0.0194 with Newey-West t 2.24 (H1 supported), mom_20d quintile net Sharpe 0.082 with PSR 0.548 (H2 not supported), rev_1d rank gross Sharpe -0.785 and net -4.056. `scripts/plot_posthoc.py` also reproduced `dev_vs_holdout_sharpe.csv` exactly.
- Every number in Experiments and Results appears in `results/dev/*`, `results/holdout/*` or `results/benchmark.json`.

Protocol checks:

- The SHA-256 of `PREREGISTRATION.md` today (`0f38b396...79f6dc`) equals the hash recorded in `LOCK.json` when the holdout started (03:07 UTC on 2026-09-27, which is 23:07 on 2026-09-26 Eastern). The file was last modified at 22:30:43 Eastern, 35 seconds before the download log was created and about 37 minutes before the holdout, so it was fixed before any data was seen. The project is not yet in git, so this rests on file timestamps plus the lock hash rather than on commit history.
- There is one lock, one holdout log, and `run_dev.py` cuts the panel at 2024-06-30 before building features. I found no path by which development results could have seen holdout data.
- The conclusion follows from the holdout: H1 passes its pre-registered bar (t above 2) and H2 fails its bar (PSR below 0.95). The better holdout Sharpes of ridge and gbm are correctly labeled post hoc.

What I deferred:

- Timing. The load average was between 9 and 60, so I did not rerun the benchmark for timing. The script ran cleanly with one repeat and its output redirected to a scratch folder.

Fixes made:

- `tests/test_features.py::test_features_do_not_look_ahead` compared features only at dates before T-1 (`< cut`), even though its docstring says features at T-1 must not change. I checked this with a mutation: replacing `ret_1d` with a one day lead (`close.shift(-1) / close - 1`) passed the original test. With `<= cut` the lead is caught, and the real features still pass. The test now uses `<= cut`. Suite still 25 passed.

Open issues (recorded, not fixed):

- As documented, the `*_lag1` blocks in `holdout_report.json` repeat the lag 0 IC. I confirmed this.
- In the lag 1 robustness backtest, `fwd_ret` only exists on universe member rows, so a lagged weight on a coin that left the universe the next day earns 0 instead of its return. This only affects the lag 1 robustness numbers, not H1 or H2.
- A missing next bar (delisting) earns 0 rather than a delisting loss. It affects 58 of 79,190 development member rows, so the effect is small, but it is optimistic for long positions.
