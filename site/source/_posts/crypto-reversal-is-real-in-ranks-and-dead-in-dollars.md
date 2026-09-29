---
layout: post
title: "Testing Crypto Reversal Under a Pre-Registered Protocol"
tags:
  - quant
  - crypto
  - research
description: >-
  A pre-registered study of the 50 most liquid Binance pairs finds that one day
  reversal predicts next day ranks in a two year holdout, while the portfolio
  chosen in development earns a net Sharpe of 0.08 after 15 bps costs and fails
  its test.
date: 2026-09-29 02:17:08
---


I wrote down one hypothesis, a universe, a period split, a list of 18 trials, a cost model and a decision rule, and froze that file before downloading a single price. Then I did all model selection on 2020 to mid 2024 data, and opened a two year holdout (July 2024 to August 2026) exactly once.

The result splits in two. Among the 50 most liquid USDT pairs on Binance, yesterday's losers do rank above yesterday's winners the next day. The holdout mean rank information coefficient of one day reversal is 0.019 with a Newey-West<sup>[[1]](#ref-1)</sup> $t$ of 2.24, which clears the pre-registered bar of $t > 2$ by a small margin. That hypothesis is supported. The same signal traded as a dollar-neutral portfolio has a holdout gross Sharpe of -0.78 and a net Sharpe of -4.06. It lost money before paying a single basis point of cost. The second hypothesis, that the portfolio picked in development makes money after 15 bps per trade, is not supported. That portfolio earned a net Sharpe of 0.08 with a probabilistic Sharpe ratio<sup>[[2]](#ref-2)</sup> of 0.55, against a required 0.95.

So the honest answer to "can we predict markets" from this study is yes, a little, at the level of daily cross-sectional ranks, and no, not in a way that paid after costs in the pre-registered test. There is also a post-hoc observation that the ridge and gradient boosting models did much better in the holdout than the model my selection rule picked. I explain below why I cannot claim that as a result.

Every number in this post comes from a result file the pipeline wrote.

*Reading note.* If you only want the verdicts, read [Results](#results). The most useful sections for anyone building a backtest are [Problems](#problems) and the selection failure at the end of [Experiments](#experiments).

## What I Wanted to Build

I wanted a research pipeline I could defend line by line, applied to one falsifiable question. Does yesterday's loser beat yesterday's winner tomorrow among liquid crypto pairs (short-term reversal), and if so, is any of that left after paying to trade it?

The question was chosen to be boring. Reversal is one of the oldest documented cross-sectional effects in equities<sup>[[3]](#ref-3)</sup><sup>[[4]](#ref-4)</sup>, it needs nothing but prices, and it has an obvious enemy in trading costs because it turns the book over almost completely every day. A study where the answer could plausibly go either way, on free data, with a known failure mode, is a good test of the process rather than of my ability to find a clever signal.

The process was the point. Most backtests I have read fail in the same four ways. They peek at the future inside a feature or a universe definition. They quietly drop the coins or stocks that died. They re-use the test period while iterating, so it stops being a test. And they report the best of many variants without saying how many variants there were. I wanted each of those to be either impossible by construction or caught by a test.

### The Pre-Registration

The pre-registration document was written before any data was downloaded and fixes everything a result could be tuned against. Nothing in it may change after the holdout except an amendments section, which is still empty.

The two hypotheses and their decision rules, in full.

| Hypothesis | Statistic | Supported if | Fails if |
|---|---|---|---|
| H1, signal level. The cross-sectional rank of a coin's past 1 day return negatively predicts the rank of its next 1 day return | Holdout mean daily Spearman IC of `rev_1d = -ret_1d`, Newey-West $t$ with 5 lags | mean IC above 0 and $t > 2.0$ | mean IC at or below 0 (anything between is inconclusive) |
| H2, tradability. A dollar-neutral portfolio built from the dev-selected model earns a positive net Sharpe | Holdout annualized net Sharpe at 15 bps, and PSR that the true Sharpe exceeds 0 | net Sharpe above 0 and PSR above 0.95 | otherwise |

The file also fixes the data source, the point-in-time universe rule, the 9 features, the target and execution timing, all 9 models and both portfolio constructions (18 trials), the cost of 15 bps per unit of one-way turnover, the sensitivity grid of 0, 5, 10 and 25 bps, and the selection rule. The rule is to take the trial with the highest development net Sharpe at 15 bps, breaking ties by mean IC.

The periods are fixed too.

```
2018-01-01        2020-01-01                       2024-07-01              2026-08-31
    |  warm-up         |  development, 9 half-year folds  |  holdout, run once       |
    |  features and    |  walk-forward, all selection     |  refit once on data      |
    |  training only   |  5 day embargo per fold          |  up to 2024-06-25        |
```

The holdout script writes a lock file containing the SHA-256 of the pre-registration before it computes anything, and refuses to run if the lock exists. While writing this post I recomputed the hash of the current pre-registration, and it matches the one recorded in the lock, so the file has not been edited since the holdout ran.

An independent reviewer who did not build the project later checked the same thing from the other side. The hash of the pre-registration still equals the one recorded in the lock when the holdout started, at 03:07 UTC on 2026-09-27. The file was last modified at 22:30:43 US Eastern the evening before, 35 seconds before the download log was created and about 37 minutes before the holdout ran, so it was fixed before any price was seen. The project was not yet in git at that point, so this rests on file timestamps plus the lock hash rather than on commit history. The same reviewer reran the development study and recomputed the holdout from the frozen selection, and both matched the committed files exactly (see [Reproducibility](#reproducibility)).

A negative answer was an acceptable outcome, and half of one is what I got.

## Theory

### Cross-Sectional Prediction and the Information Coefficient

The study never predicts whether the market goes up. Each day it ranks the coins in the universe by a forecast and asks whether that ranking lines up with the ranking of next-day returns. The daily Spearman correlation between the two is the information coefficient.

$$
\mathrm{IC}_t = \operatorname{corr}\big(\operatorname{rank}(f_{i,t}),\ \operatorname{rank}(r_{i,t+1})\big)
$$

where $f_{i,t}$ is the forecast for coin $i$ at the close of day $t$ and $r_{i,t+1}$ is its return over the next day. A mean IC of 0.02 sounds like nothing. Grinold's fundamental law of active management<sup>[[5]](#ref-5)</sup> says the information ratio of a strategy scales like $\mathrm{IC}\sqrt{B}$, where $B$ is the number of independent bets per year, and 50 names a day for 365 days a year is a lot of bets. The catch is that the law ignores costs, and it treats every bet as equally sized. Both assumptions turn out to matter here.

### Why Reversal Might Exist

In equities, short-term reversal is usually explained as payment for liquidity<sup>[[6]](#ref-6)</sup>. Someone who has to trade pushes the price, and whoever absorbs that flow is paid by the rebound. Part of it is also bid-ask bounce<sup>[[7]](#ref-7)</sup>, where a close that prints at the bid looks like a loss and the next close at the ask looks like a gain. Bounce should be small here, because Binance daily closes on liquid pairs are last trades on books with spreads of a few basis points.

### What Costs Do to a Daily Book

A daily-rebalanced long-short book with average turnover $T$ per day (the sum of absolute weight changes) and a one-way cost $c$ loses $T \cdot c \cdot 365$ per year. At 15 bps and a turnover of 1.3, which is typical for a 1 day reversal signal, that is about 71% per year. The holdout confirms the arithmetic to the decimal. The reversal portfolio's annualized gross return was -17.2% and its net return -89.0%, a gap of 71.8 points, and $1.311 \times 0.0015 \times 365 = 0.718$. Any gross edge has to beat that number before it is worth anything.

### Statistics That Account for How the Numbers Were Made

Daily ICs are autocorrelated when a signal changes slowly. A 20 day momentum ranking barely moves day to day, so consecutive ICs are not independent draws, and the naive $t$-statistic overstates confidence. I use a Newey-West $t$-statistic<sup>[[1]](#ref-1)</sup>, which replaces the variance of the mean with a Bartlett-weighted sum of autocovariances up to lag $L = 5$.

$$
\hat{\sigma}^2_{\text{NW}} = \hat{\gamma}_0 + 2\sum_{l=1}^{L}\left(1 - \frac{l}{L+1}\right)\hat{\gamma}_l, \qquad t = \frac{\bar{x}}{\sqrt{\hat{\sigma}^2_{\text{NW}}/n}}
$$

For portfolio returns I use the probabilistic Sharpe ratio (Bailey and López de Prado, 2012)<sup>[[2]](#ref-2)</sup>. It is the probability that the true Sharpe exceeds a threshold $\mathrm{SR}^*$, given the observed per-period Sharpe $\widehat{\mathrm{SR}}$, the sample length $n$, and the skew $\gamma_3$ and kurtosis $\gamma_4$ of the returns.

$$
\mathrm{PSR}(\mathrm{SR}^*) = \Phi\left(\frac{(\widehat{\mathrm{SR}} - \mathrm{SR}^*)\sqrt{n-1}}{\sqrt{1 - \gamma_3\widehat{\mathrm{SR}} + \frac{\gamma_4 - 1}{4}\widehat{\mathrm{SR}}^2}}\right)
$$

Fat tails widen the denominator, so a crypto strategy needs a longer or cleaner record than a normal-returns calculation would suggest.

For the selection step I use the deflated Sharpe ratio (Bailey and López de Prado, 2014)<sup>[[8]](#ref-8)</sup>, which is the PSR with the threshold raised to the Sharpe you would expect from the luckiest of $N$ worthless strategies. With $\gamma$ the Euler-Mascheroni constant and $z(\cdot)$ the standard normal quantile,

$$
\mathrm{SR}^* = \sqrt{\operatorname{Var}\left[\widehat{\mathrm{SR}}_{\text{trials}}\right]}\,\Big((1-\gamma)\,z\big(1 - \tfrac{1}{N}\big) + \gamma\, z\big(1 - \tfrac{1}{Ne}\big)\Big)
$$

The more variants you try, and the more their Sharpes disagree, the higher the bar. The implementation is five lines.

```python
def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
```

A formula like this is easy to get subtly wrong and impossible to eyeball, so a unit test checks it against a Monte Carlo estimate of the mean maximum of 50 normals over 20,000 draws and requires agreement within 5%.

## Architecture

The pipeline is a straight line from a public archive to a report, with two stages, the development run and the one-time holdout, that carry the protocol.

<figure class="excal" data-diagram="crypto-pipeline"><a href="/img/diagrams/crypto-pipeline.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/crypto-pipeline.webp" alt="Crypto reversal pipeline from the Binance archive to one forecast that splits into a rank IC path, where H1 is supported, and a dollar backtest path, where H2 is not supported after costs, above a timeline of the development period and the one-time holdout." width="2400" height="4823" loading="lazy" decoding="async"></a></figure>

Three design choices carry most of the weight.

### Everything Is a Rank

Features and the training label are converted to centered cross-sectional ranks in $[-0.5, 0.5]$ each day. Crypto daily returns have tails that would dominate any raw regression. The largest move inside the development universe was DOGE at +392% on 2021-01-28, and a squared-error fit on raw returns would spend most of its capacity on a handful of such days. The cost of ranking, which I did not appreciate until the holdout, is that the models optimize rank IC, and rank IC is not the same thing as PnL.

### Wide Frames for Features, a Long Frame for Models

Every rolling feature is one vectorized call on a date by symbol frame with a complete daily index, so a row is a day and a missing bar is a NaN that propagates through each rolling window's minimum-observations requirement. Only universe members are stacked into the long `(date, symbol)` frame the models see. In development the wide frames are 2,373 days by 522 listing episodes.

### The Backtest Is Two Lines of Arithmetic

Weights formed at the close of day $t$ earn the return from close $t$ to close $t+1$. On a market that trades around the clock, the daily close at midnight UTC and the next open are the same instant, which removes a whole class of execution-timing questions that daily equity backtests have to answer.

```python
W = weights.unstack("symbol").fillna(0.0).sort_index()
R = fwd_ret.unstack("symbol").reindex_like(W).fillna(0.0)
if lag:
    W = W.shift(lag).fillna(0.0)
gross = (W * R).sum(axis=1)
turnover = W.diff().abs().sum(axis=1)
```

Net returns at any cost are `gross - turnover * bps / 1e4`, so the whole cost grid comes from one backtest. A coin that has no bar the next day (a delisting) earns 0. The weight was chosen without knowing the bar would be missing, and the backtest must not know it either. Zero is still a choice, and an optimistic one for long positions, since a real delisting usually comes with a loss. It touches 58 of 79,190 development member rows, so the effect on the results is small, but it is not modeled.

## Implementation

The library is about 600 lines of NumPy, pandas, scikit-learn<sup>[[9]](#ref-9)</sup> and SciPy, plus 25 pytest tests that run on synthetic panels with no network.

### Data

I list every USDT symbol in the Binance public data archive<sup>[[10]](#ref-10)</sup> (735) and download every monthly daily-kline zip from 2018-01 to 2026-08. That is 27,721 files, fetched with 32 threads in 400 seconds, parsed into 829,342 rows across 734 symbols and a 25 MB parquet file. Downloading every pair rather than a list of well-known coins is the main defense against survivorship bias. A hand-picked list of "top coins" encodes today's knowledge of which ones survived.

### The Universe

On each day a symbol is eligible only if, using data up to and including that day, it has at least 60 bars of history, its base asset is not a stablecoin, fiat currency, gold token, wrapped or staked duplicate, or leveraged token, its trailing 30 day median quote volume is at least 1,000,000 USDT, and that volume ranks in the top 50 of eligible names. Days with fewer than 20 names are skipped.

The rule sounds simple and is the easiest place in the whole pipeline to leak the future. "Top 50 by volume" computed over the full sample, or a coin's history counted from its first bar in a later file, silently uses information from after day $t$. The eligibility rule builds every quantity from cumulative or trailing operations on the wide frames, and a test perturbs every price from a date $T$ on and asserts that the universe and every feature up to and including $T-1$ are unchanged. That test had a hole in its first version, described in [Problems](#problems).

### Features

Nine features, computed at the close of day $t$ and ranked among that day's members.

| Feature | Definition |
|---|---|
| `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d` | trailing returns, the reversal and momentum family<sup>[[3]](#ref-3)</sup><sup>[[11]](#ref-11)</sup> |
| `vol_20d` | 20 day standard deviation of log returns |
| `dvol_20d` | log of 20 day median quote volume |
| `amihud_20d` | 20 day mean of absolute return over quote volume, Amihud's illiquidity measure<sup>[[12]](#ref-12)</sup> |
| `dist_high_20d` | close over the 20 day high, minus 1 |
| `beta_btc_60d` | 60 day rolling beta of log returns to BTC |

A member with a missing value gets rank 0, the cross-sectional median, so a model never sees a NaN and never learns from missingness itself.

### Models

Four single factors with no fitting (`rev_1d`, `rev_5d`, `mom_20d`, `mom_60d`), ridge regression<sup>[[13]](#ref-13)</sup> at $\alpha \in \{1, 10, 100\}$, and scikit-learn's<sup>[[9]](#ref-9)</sup> gradient boosting<sup>[[14]](#ref-14)</sup> regressor, `HistGradientBoostingRegressor`, at depth 3 and depth 6 (200 iterations, learning rate 0.05). The fitted models predict the cross-sectional rank of the next-day return from all nine ranked features.

### Walk-Forward with an Embargo

Development runs nine half-year test folds, 2020H1 through 2024H1, each with an expanding training window. The label on a training row dated $d$ is only known at the close of $d+1$, so the embargo<sup>[[15]](#ref-15)</sup> has to be counted from the label, not the row.

```python
@property
def train_end(self) -> pd.Timestamp:
    # A training row dated d has a label that is known at the close of d+1.
    # Requiring d + 1 <= test_start - EMBARGO_DAYS leaves a gap of
    # EMBARGO_DAYS between the last label and the first test day.
    return self.test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)
```

Five days is more than a 1 day label strictly needs, and it costs almost nothing. A test fits a spy model that records the latest date it was shown and asserts it never passes the end of the embargoed training window.

### The Holdout Lock

The holdout script is short and its first few lines are the protocol.

```python
if LOCK.exists():
    sys.exit(f"refusing: holdout already evaluated, see {LOCK}")
sel = json.loads(SELECTION.read_text())   # the trial frozen in development
prereg = PREREGISTRATION.read_bytes()
lock = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": hashlib.sha256(prereg).hexdigest(),
        "selection": sel, "status": "started"}
LOCK.write_text(json.dumps(lock, indent=2))
```

The lock is written before any holdout number exists, so a crash halfway through still counts as the one touch. This is not tamper-proof against me, since I could delete the file, but deleting it would be a deliberate act rather than an accident, and the hash makes any later edit to the pre-registration detectable. The lock records a start and a completion four minutes apart on 2026-09-27, with verdicts "supported" and "not supported".

## Problems

None of these broke the build. Most of them would have produced plausible numbers.

### The API Was Blocked, and the Replacement Was Better

The Binance REST API returns HTTP 451 from the US. I switched to the static archive at data.binance.vision, which turned out to be the better research source anyway, because it keeps the monthly files of pairs that have since been delisted.

### Timestamps Changed Units Mid-Archive

The archive switched its kline timestamps from milliseconds to microseconds starting in 2025<sup>[[10]](#ref-10)</sup>. Parsed naively as milliseconds, every bar from 2025 on lands tens of thousands of years in the future. I knew about the switch before writing the parser, so the parser checks magnitude row by row.

```python
t = pd.to_numeric(df["open_time"]).astype("int64").to_numpy()
unit_us = t > 10**14   # Binance spot archives use microseconds from 2025-01 onward
t_ms = np.where(unit_us, t // 1000, t)
```

A test feeds it one row of each unit. The end-to-end check is that the parsed panel ends on 2026-08-31.

### One Ticker, Two Assets

LUNAUSDT was the original Terra until its collapse in May 2022, and the new Terra from late May 2022 onward. A 60 day return computed across that gap compares two unrelated assets, and the "return" is meaningless. The fix is to split any gap longer than 7 days into a new listing episode, `LUNAUSDT#2`, which the rest of the pipeline treats as a separate symbol. In the development data this turned 507 symbols into 522 episodes. Both episodes show up among the five largest daily moves in the development universe, the old LUNA at -99.97% on 2022-05-12 and the new one at +168% on 2022-09-09.

### The Leveraged-Token Filter Ate a Real Coin

My first rule for leveraged tokens excluded any base ending in UP, DOWN, BULL or BEAR. That catches BTCUP and ETHBEAR, and also JUP. The rule now fires only when the stripped stem is itself a listed base, and the test includes JUP and SUPER as coins that must survive.

### pandas 3 Changed stack

pandas 3 changed `DataFrame.stack()` to keep NaN rows<sup>[[16]](#ref-16)</sup>. My first version of the long frame stacked the ranked feature frames directly, which under pandas 3 would have included every cell of the date by symbol grid rather than only universe members, so rows for coins outside the universe would have reached the models and the IC. I now stack the boolean mask, keep the true cells, and reindex every feature onto that index explicitly.

### The Run That Hung

This one cost the most. The first development run logged 14 trials, all the single factors and all the ridge models, and then sat at 0.2% CPU for more than ten minutes with no output. Nothing crashed and nothing timed out.

A standalone repro that dumped stack traces showed the process stuck inside the gradient boosting fit, while building the root of the first tree, on 80,000 random rows. With OpenMP limited to one thread the same fit finished in 3.4 seconds. With 4 or 14 threads it did not finish within 60 seconds. scikit-learn ships its own OpenMP runtime, and the machine was also running 14 other agents' jobs at the time. I did not find the root cause. The fix is a one-thread limit around every scikit-learn fit and predict.

```python
def fit(self, train: pd.DataFrame):
    t = train.dropna(subset=["y_rank"])
    with threadpool_limits(1):
        self.est.fit(t[FEATURES].to_numpy(np.float32), t["y_rank"].to_numpy(np.float32))
    return self
```

A depth 3 fit on 75,250 rows takes about 3 seconds single-threaded, so the price is small. The more interesting cost is on the record. The pre-registration says every trial ever evaluated in development is appended to a trial log and the deflated Sharpe uses its line count as $N$. The hung run had already appended 14 lines, so the log has 32 lines rather than 18, and the deflated Sharpe below uses $N = 32$. That overstates the number of distinct trials, since 14 of them are repeats, which makes the bar higher and the test more conservative. I left it that way because the rule said to count lines, and a rule you adjust when it is inconvenient is not a rule.

### A Bookkeeping Bug I Found After the Holdout

After the holdout ran I noticed that the lag 1 robustness blocks in the holdout report give the IC of the unlagged forecast. The IC fields are copied from the lag 0 evaluation, while the Sharpe and return fields in those blocks are correctly lagged. Fixing it means rerunning the holdout, which the protocol forbids, so the report stays as it is and the bug is documented. The lagged IC numbers are not used anywhere in this post. The verification found one more issue in the same lag 1 blocks. The forward return only exists on rows where a coin is a universe member, so a lagged weight on a coin that left the universe the next day earns 0 instead of its actual return. This affects only the lag 1 robustness numbers, such as the -0.43 quoted below, and neither H1 nor H2.

### A Look-Ahead Test That Could Not Fail

The test that guards against look-ahead in the features changes every price from a date $T$ on and checks that nothing before it moves. Its docstring said features at $T-1$ must not change, but the code compared only dates strictly before $T-1$, with a `<` where it needed `<=`. That left exactly the case that matters most unguarded, a feature at the close of $T-1$ that quietly uses the bar at $T$.

I did not catch this. The independent verification did, with a mutation test. Replacing `ret_1d` with a one day lead, `close.shift(-1) / close - 1`, passed the original test. With `<=` the lead is caught and the real features still pass, so the test now uses `<=` and the suite is still 25 passing tests. The real features never had a lead, so no result changes, but for the whole study the test that was supposed to prove that could not have failed on the most likely bug. A test only counts once you have seen it fail on the thing it is meant to catch.

### The One That Shapes the Result

The last problem is not a bug. High IC did not mean high PnL, and the pre-registered selection rule picked the only trial family with a negative IC. That is the story of the next two sections.

## Experiments

All runs use the data described above. Development is 2020-01-01 to 2024-06-30 in 9 walk-forward folds. The holdout is 2024-07-01 to 2026-08-31, 792 days. Costs are 15 bps per unit of turnover unless stated, and Sharpe ratios are annualized with 365 days. Annual returns are the daily mean times 365, not compounded.

### Data Checks

The development universe had a median of 50 names and a minimum of 26 on 1,643 days, drawn from 279 distinct listing episodes over the period. Of 79,190 member rows, only 58 had no next-day bar. The holdout universe also had a median of 50 names.

### Development Walk-Forward, All 18 Trials

A selection of the 18 rows from the development summary.

| Trial | Mean IC | NW $t$ | Gross Sharpe | Net Sharpe | Turnover per day |
|---|---|---|---|---|---|
| rev_1d, rank | +0.036 | 6.98 | 0.15 | -2.86 | 1.33 |
| rev_5d, rank | +0.033 | 6.03 | -1.13 | -2.34 | 0.60 |
| mom_20d, quintile | -0.029 | -5.18 | 0.91 | 0.33 | 0.39 |
| mom_60d, rank | -0.049 | -8.51 | -0.03 | -0.42 | 0.19 |
| ridge alpha 10, rank | +0.097 | 17.29 | 0.24 | -0.92 | 0.56 |
| ridge alpha 10, quintile | +0.097 | 17.29 | 0.45 | -0.75 | 0.74 |
| gbm depth 3, rank | +0.108 | 21.38 | 1.49 | -0.17 | 0.71 |
| gbm depth 3, quintile | +0.108 | 21.38 | 1.64 | 0.06 | 0.88 |
| gbm depth 6, quintile | +0.099 | 20.77 | 1.77 | -0.08 | 0.99 |

The signals behaved exactly as the literature would predict at the level of ranks<sup>[[3]](#ref-3)</sup><sup>[[4]](#ref-4)</sup>. Both reversal factors had positive IC in all 9 folds. Both momentum factors had negative IC, in 8 of 9 folds for `mom_20d` and all 9 for `mom_60d`. Momentum over 20 and 60 days is, in this universe, a reversal signal at a longer horizon.

<figure data-figure="chart:projects/can-we-predict-markets/reversal-ic"></figure>

The fitted models roughly tripled the reversal IC, to about 0.10, and they did it mostly without reversal. Ridge put its largest coefficient on low volatility, about -0.08 on `vol_20d` in every fold, and its second largest on 1 day reversal, about -0.04 on `ret_1d`. One possible reading, not tested here, is that this is partly a property of rank labels. On a typical day the median coin loses a little, high-volatility coins spread into both tails, and a forecast that says "low-volatility coins land nearer the middle and slightly above" wins a lot of small rank bets. Whether that is worth anything in dollars is exactly the question rank IC cannot answer.

Look at the last two columns of the table. The gradient boosting models had gross Sharpes of 1.49 to 1.77 and handed almost all of it to costs, because a forecast that re-ranks 50 coins daily trades most of the book daily. The depth 3 quintile portfolio earned 50.4% a year gross and 1.9% net at a turnover of 0.88. The single trial with a clearly positive net Sharpe was `mom_20d` with quintile weights, which had the wrong-signed IC for its name and a low turnover of 0.39.

### Selection and Multiple Testing

The frozen rule picked `mom_20d` with quintile weights, the highest development net Sharpe at 0.33, with a PSR of 0.75. Then the deflated Sharpe ratio asked how impressive 0.33 is after 32 logged trials with the observed spread of trial Sharpes. The expected best annualized Sharpe of 32 worthless strategies is 2.01, and the deflated Sharpe ratio of the selected trial is 0.00018.

<figure data-figure="chart:projects/can-we-predict-markets/deflated-sharpe"></figure>

In plain words, the best development result was exactly what you would expect from the luckiest of 32 useless strategies. This was known before the holdout was opened, and the protocol said to open it anyway with the selected trial. That is the right call. The deflated Sharpe is an argument about how much to believe a number, not a license to go back and pick something else.

### The Holdout

Every model was refit once on data up to 2024-06-25 and run without refitting over the 792 holdout days. Only two evaluations are confirmatory, `rev_1d` for H1 and `mom_20d` quintile for H2. The holdout run also evaluates the other 16 trials, but writes them to a separate file labeled post hoc, so the name carries the warning.

## Results

### H1, Reversal as a Signal, Supported

The holdout mean rank IC of `rev_1d` is +0.019, with a Newey-West $t$ of 2.24 and a daily hit rate of 53.1%. That clears the pre-registered bar of $t > 2$, barely. It is about half the development IC of 0.036, which is what decay out of sample usually looks like, and it is still the same sign it had in all nine development folds. Short-term reversal exists in the cross-section of liquid crypto, in ranks, in a period the model never saw.

### Reversal as a Strategy, Dead Before Costs

The same signal traded with rank weights has a holdout gross Sharpe of -0.78 and a net Sharpe of -4.06 at 15 bps, with turnover of 1.31 a day and a maximum drawdown of -86%. It lost money at zero cost. The ordering is right slightly more often than not, and the dollars go the other way.

This is the central finding, so it is worth being precise about how it can happen. Rank IC weights every coin equally and counts only whether the order is right. PnL weights every coin by its position times its return, and in crypto the returns that matter are the few coins that move 20% in a day. A signal can win 53% of its rank comparisons and lose on the coins where it is wrong by the most. My working explanation, which I have not tested, is that the coins that just fell hardest are also the ones most likely to keep falling hard. Reversal holds on average across the cross-section and fails in the tail that sets the PnL. Testing that would mean splitting the holdout PnL by the size of the prior move, and that analysis would be post hoc on a spent holdout.

### H2, the Selected Portfolio, Not Supported

`mom_20d` with quintile weights earned a holdout gross Sharpe of 0.77, a net Sharpe of 0.08, a PSR of 0.55 and a maximum drawdown of -29%. Positive, but indistinguishable from zero, and far from the 0.95 PSR the pre-registration required. Its holdout IC was -0.004 with a $t$ of -0.52, so it had no measurable rank skill at all. Its gross PnL came from the right tail of a few trending coins, which is the same tail effect that sank reversal, pointing the other way.

<figure data-figure="chart:projects/can-we-predict-markets/holdout-cost"></figure>

The cost sensitivity shows how thin the edge was. Net Sharpe falls from 0.77 at 0 bps to 0.54 at 5 bps and -0.37 at 25 bps. Delaying execution by one day takes it to -0.43 at 15 bps (this comes from the lag 1 robustness block, whose Sharpe fields are correctly lagged). None of this includes the funding cost of holding the short leg as a perpetual future, which is not modeled at all and would make things worse in a rising market.

### Post Hoc, and Not a Claim

Because the holdout script evaluated every trial for context, I can see that the selection rule picked badly. The gradient boosting depth 3 quintile portfolio had a holdout net Sharpe of 0.71, and the three ridge quintile portfolios about 0.55, with holdout ICs near 0.10. The Spearman correlation between development and holdout net Sharpe across the 18 trials is 0.46.

<figure data-figure="chart:projects/can-we-predict-markets/dev-vs-holdout"></figure>

It is tempting to write "the machine learning models work". I cannot. I have now looked at the holdout, and picking the winner after looking is exactly the multiple-testing error the protocol exists to prevent. If I had pre-registered "select on IC", I would have picked gradient boosting and this would be a positive result. I did not, and I would be writing a different rule after seeing which one would have won. Even the observation itself is weaker than it looks. None of those holdout Sharpes would pass the pre-registered PSR bar (the best, gbm depth 3 quintile, has a PSR of 0.85), none carries short-side funding costs, and all of them sit inside a set of 18 that includes four portfolios below -2. The honest status is a hypothesis for the next fresh holdout.

### So, Can We Predict Markets?

At the level of daily cross-sectional ranks in liquid crypto, yes, a little. The reversal IC was positive in nine half-years of development and in a separate two year holdout, and the fitted models found a larger and equally stable rank signal. Turning any of it into money after 15 bps per trade did not happen in the pre-registered test. The main thing the study measured is the distance between those two statements.

### Benchmark

Stage timings, median of 3 runs on an M4 Pro that was heavily loaded by other jobs at the time, so treat them as rough.

| Stage | Time |
|---|---|
| Build the development dataset (84,350 member rows) | 11.1 s |
| Ridge fit on 75,250 rows | 0.01 to 0.19 s |
| Gradient boosting fit, depth 3 | 3.0 s |
| Gradient boosting fit, depth 6 | 4.6 s |
| Daily IC over the whole development set | 2.5 s |
| One backtest | 0.12 s |

In the development run, the two gradient boosting walk-forwards took 76 s and 96 s of a roughly 4 minute total. The whole study, from raw archive to holdout verdict, runs in about 15 minutes of wall time on a laptop, most of it the download.

## What I Would Change

### Select on IC, or on Something That Knows About Turnover

The IC ranking of trials was stable from development to holdout. The net Sharpe ranking was noisy, and the rule chose a strategy whose only merit was a lucky right tail. A pre-registered score that combines IC with expected turnover and cost would have been more stable. I did not change the rule after seeing development results, and I should not have, but it is the first thing I would write differently.

### Cut Turnover Inside the Portfolio

The gradient boosting models had gross Sharpes near 1.5 to 1.8 in development and lost most of that to costs. Trading toward the target weights with a no-trade band, or smoothing the forecast over a few days, attacks turnover directly rather than hoping a low-turnover signal wins the selection.

### Keep a Second, Untouched Holdout

The post-hoc gradient boosting result deserves a real test, and there is no clean data left to give it one. Either wait for data after 2026-08, or hold out a second period or asset class from the start next time so a finding like this has somewhere to go.

### Model What the Short Leg Actually Costs

Binance spot does not allow shorting, so a real implementation would short perpetual futures and pay or receive funding. That needs a separate downloader for historical funding rates. Weight drift between rebalances should also enter the turnover calculation, which currently uses target weights and slightly understates it.

### Use a Smaller, Stricter Universe

The names ranked 30 to 50 by volume are where both the tails and the costs live. A top 20 or top 30 universe would test whether the rank signal survives where trading is cheap.

### Test the Thing That Broke

Fix the lag 1 IC bookkeeping and the missing returns for coins that leave the universe before the next holdout, and add a test that the lag option changes every lag-dependent field. The bug was only possible because no test asserted that. More generally, run a mutation against every guard test, the way the verification did for the look-ahead test, so each one is known to fail when it should.

### Price Delistings as Losses

A coin with no next bar currently earns 0. A pessimistic delisting return, or at least a sensitivity run with one, would remove a small bias in favor of long positions.

## Reproducibility

The project is a `uv` environment pinned to Python 3.12. The tests run on synthetic panels, so they need no download. The full study runs in three stages: the download takes about 7 minutes, the development walk-forward about 4, and the holdout is a single shot. Two small follow-up steps redraw the development versus holdout figure from saved results only and time each stage for the benchmark. Because the holdout has already been evaluated and the lock is in place, the holdout stage now exits with an error. That is the point. Re-running the development study appends another 18 lines to the trial log, which by the pre-registered rule raises $N$ for any future deflated Sharpe.

### Verification

An independent reviewer who did not build the project reran it on 2026-09-26, with threads capped at 2 on a shared machine. The test suite passed (25 tests). Re-running the download from the cached archive produced an identical parquet file (829,342 rows, 734 symbols). The full development study, run into a scratch folder seeded so the trial count again reached 32, reproduced the trial summary, the per-fold ICs, the ridge coefficients and the daily net returns exactly, with the same selection, the same deflated Sharpe of 0.00018 and the same threshold of 2.01. The holdout stage refused to run because the lock exists, so the reviewer recomputed the same code path from the frozen selection into a scratch folder, which makes no new decision. Every field of the holdout report and every cell of the post-hoc trial table matched exactly. The reviewer also confirmed the pre-registration hash against the lock and the file timestamps described above, and fixed the look-ahead test. Timing was deferred. The load average was between 9 and 60, so the benchmark was only checked to run, and the stage timings above have not been remeasured on a quiet machine.

Find the code at [github.com/SadeekFarhan21/projects/crypto-reversal-preregistered](https://github.com/SadeekFarhan21/projects/tree/main/crypto-reversal-preregistered).

## References

1. <span id="ref-1"></span>Whitney K. Newey and Kenneth D. West. *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*. Econometrica 55(3), 1987. [doi:10.2307/1913610](https://doi.org/10.2307/1913610)
2. <span id="ref-2"></span>David H. Bailey and Marcos López de Prado. *The Sharpe Ratio Efficient Frontier*. Journal of Risk 15(2), 2012. [doi:10.21314/JOR.2012.255](https://doi.org/10.21314/JOR.2012.255)
3. <span id="ref-3"></span>Narasimhan Jegadeesh. *Evidence of Predictable Behavior of Security Returns*. Journal of Finance 45(3), 1990. [doi:10.1111/j.1540-6261.1990.tb05110.x](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x)
4. <span id="ref-4"></span>Bruce N. Lehmann. *Fads, Martingales, and Market Efficiency*. Quarterly Journal of Economics 105(1), 1990. [doi:10.2307/2937816](https://doi.org/10.2307/2937816)
5. <span id="ref-5"></span>Richard C. Grinold. *The Fundamental Law of Active Management*. Journal of Portfolio Management 15(3), 1989. [doi:10.3905/jpm.1989.409211](https://doi.org/10.3905/jpm.1989.409211)
6. <span id="ref-6"></span>Stefan Nagel. *Evaporating Liquidity*. Review of Financial Studies 25(7), 2012. [doi:10.1093/rfs/hhs066](https://doi.org/10.1093/rfs/hhs066)
7. <span id="ref-7"></span>Richard Roll. *A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market*. Journal of Finance 39(4), 1984. [doi:10.1111/j.1540-6261.1984.tb03897.x](https://doi.org/10.1111/j.1540-6261.1984.tb03897.x)
8. <span id="ref-8"></span>David H. Bailey and Marcos López de Prado. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management 40(5), 2014. [doi:10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)
9. <span id="ref-9"></span>Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research 12, 2011. [link](https://jmlr.org/papers/v12/pedregosa11a.html)
10. <span id="ref-10"></span>Binance. *Binance Public Data*. Binance Data Collection, data.binance.vision. [link](https://github.com/binance/binance-public-data)
11. <span id="ref-11"></span>Narasimhan Jegadeesh and Sheridan Titman. *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*. Journal of Finance 48(1), 1993. [doi:10.1111/j.1540-6261.1993.tb04702.x](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
12. <span id="ref-12"></span>Yakov Amihud. *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects*. Journal of Financial Markets 5(1), 2002. [doi:10.1016/S1386-4181(01)00024-6](https://doi.org/10.1016/S1386-4181%2801%2900024-6)
13. <span id="ref-13"></span>Arthur E. Hoerl and Robert W. Kennard. *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics 12(1), 1970. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
14. <span id="ref-14"></span>Jerome H. Friedman. *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics 29(5), 2001. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
15. <span id="ref-15"></span>Marcos López de Prado. *Advances in Financial Machine Learning*, chapter 7, "Cross-Validation in Finance". Wiley, 2018. [link](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
16. <span id="ref-16"></span>The pandas development team. *What's New in 3.0.0*. pandas documentation, 2026. [link](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html)
