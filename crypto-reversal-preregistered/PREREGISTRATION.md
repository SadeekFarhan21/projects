# Pre-registration: cross-sectional short-term reversal in liquid crypto

Written 2026-09-26, before any data was downloaded. Nothing in this file may be
edited after the holdout run except the "Amendments" section at the bottom,
which must record the date and reason for every change. The holdout lock file
(`results/holdout/LOCK.json`) records the SHA-256 of this file at the moment the
holdout was evaluated, so any later edit is detectable.

## 1. Hypotheses

### H1 (primary, signal level)
Among the most liquid USDT spot pairs on Binance, the cross-sectional rank of a
coin's past 1 day return negatively predicts the rank of its next 1 day return
(short-term reversal).

Test statistic: the daily Spearman rank information coefficient (IC) between
the reversal signal `rev_1d = -ret_1d` and the next day return, averaged over
the holdout period, with a Newey-West t-statistic (lag 5).

- H1 is supported if the holdout mean IC is > 0 and t > 2.0.
- H1 is rejected (reversal absent) if the mean IC is <= 0.
- Anything in between is "inconclusive".

### H2 (secondary, tradability)
A dollar-neutral long-short portfolio built from the dev-selected model earns a
positive Sharpe ratio net of realistic trading costs in the holdout.

- H2 is supported if the holdout annualized net Sharpe is > 0 and the
  probabilistic Sharpe ratio PSR(SR > 0) is > 0.95.
- Otherwise H2 is not supported.

## 2. Data
- Source: Binance public archive `data.binance.vision`, spot, daily klines
  (UTC days), all `*USDT` symbols listed in the archive including delisted ones
  (to limit survivorship bias).
- Download range: 2018-01 through 2026-08 (monthly files).

## 3. Universe (point in time)
On each day t a symbol is eligible only if, using data up to and including day t:
- it has at least 60 daily bars of history,
- its base asset is not a stablecoin, fiat, wrapped/staked duplicate, gold
  token or leveraged token (fixed exclusion list in `src/marketpred/universe.py`),
- its trailing 30 day median dollar volume ranks in the top 50 of eligible names,
- that median dollar volume is at least 1,000,000 USDT.
Days with fewer than 20 eligible names are skipped.

## 4. Periods
| Period | Dates (inclusive) | Use |
|---|---|---|
| Warm-up | 2018-01-01 to 2019-12-31 | feature history and first training data only |
| Development | 2020-01-01 to 2024-06-30 | walk-forward evaluation, all model selection |
| Holdout | 2024-07-01 to 2026-08-31 | evaluated exactly once, after selection is frozen |

Walk-forward in development: expanding training window, 6 month test folds
(2020H1 ... 2024H1, 9 folds), a 5 day embargo between the last training target
and the first test day. Model training may use data from 2018 onward.

For the holdout, the selected model is refit once on all data up to 2024-06-25
(5 day embargo) and then run without refitting over the holdout.

## 5. Features (computed at the close of day t, ranked cross-sectionally)
`ret_1d, ret_5d, ret_20d, ret_60d` (momentum/reversal), `vol_20d` (realized
volatility), `dvol_20d` (log median dollar volume), `amihud_20d` (illiquidity),
`dist_high_20d` (close / 20 day high - 1), `beta_btc_60d` (rolling beta to BTC).
Each feature is converted to a centered cross-sectional rank in [-0.5, 0.5].

## 6. Target and execution
- Target: return from the close of day t to the close of day t+1.
- Signals formed at the day t close (00:00 UTC) are assumed to trade at the
  open of day t+1, which on a 24/7 market is the same instant. A 1 day extra
  execution lag is reported as a robustness check, not used for selection.
- If a held symbol has no bar on day t+1 (delisting), its return is set to 0.
  Weights never use any information from day t+1.

## 7. Models and variants (the full list of trials)
Signals/models:
1. `rev_1d` single factor (no fitting)
2. `rev_5d` single factor
3. `mom_20d` single factor
4. `mom_60d` single factor
5. Ridge on all ranked features, alpha = 1
6. Ridge, alpha = 10
7. Ridge, alpha = 100
8. Gradient boosting (sklearn HistGradientBoostingRegressor), shallow
   (max_depth 3, 200 iterations, learning rate 0.05)
9. Gradient boosting, deeper (max_depth 6, 200 iterations, learning rate 0.05)

Models are trained to predict the cross-sectional rank of the next day return.

Portfolio constructions (each crossed with every model above):
- A. rank-weighted: weights proportional to the centered rank of the forecast,
  scaled to gross exposure 1 (0.5 long, 0.5 short).
- B. quintile: equal-weight long the top 20%, short the bottom 20%, gross 1.

That is 9 x 2 = 18 trials. Every trial run in development is appended to
`results/dev/trials.jsonl`, including any not listed here, and the count used
for the deflated Sharpe ratio is the number of lines in that file.

Gradient boosting is only a candidate if it is in the list above; no tuning
beyond these two configurations.

## 8. Costs
- Base case: 15 bps per unit of one-way turnover (10 bps Binance taker fee plus
  5 bps slippage). Turnover is sum |w_t - w_{t-1}| with drift ignored.
- Sensitivity (reported, not used for selection): 0, 5, 10, 25 bps.
- Short borrow or perpetual funding costs are not modeled. This is a known
  limitation and is stated with every result.

## 9. Selection rule (frozen)
The holdout candidate is the trial with the highest development-period
annualized net Sharpe (15 bps) across all 9 folds concatenated. Ties broken by
higher mean IC. The dev-best Sharpe is also reported as a deflated Sharpe ratio
using the number of trials and the variance of trial Sharpes.

## 10. Metrics reported
Mean daily rank IC, IC t-stat (Newey-West, lag 5), IC information ratio,
hit rate, annualized gross and net return, volatility, Sharpe, max drawdown,
average daily turnover, PSR, deflated Sharpe (dev best), cost sensitivity.
Annualization uses 365 days.

## 11. Holdout protocol
`scripts/run_holdout.py` refuses to run if `results/holdout/LOCK.json` exists.
On its first run it writes the lock with a timestamp, the git-independent
SHA-256 of this file, and the selected configuration. The result is reported
whatever it is.

## Amendments
(none yet)
