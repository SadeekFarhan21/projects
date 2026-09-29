---
layout: post
title: "Why Peeking Breaks A/B Tests, and What to Use Instead"
tags:
  - statistics
  - experimentation
  - causal-inference
  - python
description: >-
  An A/B testing and off-policy evaluation engine built from scratch, where
  checking a test fifty times turns a 5 percent false positive rate into 33.
date: 2026-09-29 02:17:12
---


I wrote expope, a small engine for the two questions a product data science team answers every week. The first is whether an A/B test moved a metric, and whether the p-value can be trusted. The second is what a new recommendation policy would have scored on traffic the old policy served, using only the old policy's logs. The first half is a DuckDB metrics layer over a raw event log plus from-scratch Welch, delta method, CUPED, sample ratio mismatch, Holm, Benjamini-Hochberg and a mixture sequential probability ratio test (mSPRT). The second half is from-scratch off-policy estimators (IPS, SNIPS, direct method, doubly robust and switch-DR) with bootstrap intervals, run on a synthetic bandit with known truth and on the Open Bandit Dataset from ZOZOTOWN.

The headline is a simulation. I ran 2,000 A/A experiments, looked at each one 50 times as users arrived, and stopped at the first look where Welch's t-test gave p below 0.05. **32.8 percent** of those experiments with no effect at all were declared significant. The same data tested once at the final look gave 5.25 percent, as it should. The always-valid mSPRT, looking at the same 50 points, rejected **1.95 percent**.

The second finding is on the off-policy side. A cross-fitted gradient-boosted reward model that looks accurate on average made the direct method **11.0 percent low** on a synthetic bandit, and its bootstrap intervals covered the truth in 0 of 200 replicates. Doubly robust with the same model was 0.02 percent low and covered in 192 of 200. The bias is not a bug. The model shrinks toward the mean, and the new policy concentrates on exactly the context and action pairs where shrinkage pulls predictions down the most.

On correctness, every off-policy estimator agrees with Open Bandit Pipeline (obp) to within 1e-9 on 25 randomly generated slate problems, and on the real Open Bandit sample the largest gap was 1.7e-18. On 2026-09-27 an independent rerun of the full test suite passed all **38 tests**, including the obp cross-checks at 1e-9. The simulation experiments were not rerun independently, so every simulated rate below is from my own single run.

One limit belongs up front. The Open Bandit experiment uses the 10,000-row sample that ships with obp, and the random-policy log in it holds **38 clicks**. All five estimators land inside the on-policy confidence interval, which is a sanity check, but the intervals are so wide that this data cannot rank the estimators, and I do not try to.

Code is in `projects/cuped-msprt-ope-engine`.

## table of contents

- [what I wanted to build](#what-i-wanted-to-build)
- [theory](#theory)
- [architecture](#architecture)
- [implementation](#implementation)
- [problems](#problems)
- [experiments](#experiments)
- [results](#results)
- [what I would change](#what-i-would-change)
- [reproducibility](#reproducibility)

## what I wanted to build

Most statistics bugs in experimentation do not crash. A variance formula that treats sessions as independent, a dashboard that recomputes a t-test every morning and a reward model with a small average error all still return a number. The only way I know to catch them is to run the code on data where the answer is known and count how often it is wrong.

So the goal was an engine where every method has a simulation that measures it against truth. False positive rate on A/A tests, power against the noncentral t, interval coverage with a nonzero true effect, false positives under continuous monitoring, variance reduction against the CUPED formula, and bias and coverage of every off-policy estimator on a bandit whose true policy value is computed from 2 million contexts. For the off-policy estimators I also wanted an external reference, so every one is cross-checked against obp on identical arrays.

I wanted the A/B side to start from a raw event log, not a tidy table of per-user outcomes, because who counts as exposed, which events fall in the window and what a user with no events contributes all change the answer before any statistics run.

## theory

### Welch and power

For per-user means in two arms, the difference has variance s_c²/n_c + s_t²/n_t. Welch's test uses that directly and gets its degrees of freedom from the Welch-Satterthwaite approximation, so it does not assume equal variances. With equal n and a common σ, the power at effect d comes from a noncentral t distribution with noncentrality d / (σ √(2/n)). That formula is the reference the power simulation is checked against.

### the delta method for ratio metrics

Clicks per session is not a mean of per-user values. It is the sum of clicks over the sum of sessions, R = x̄ / ȳ, where x and y are per-user totals. Users are the randomisation unit, so the sessions inside one user are correlated, and treating every session as an independent observation understates the variance. A first-order Taylor expansion of R around (μ_x, μ_y) gives

```
Var(R) ≈ (1/n) [ Var(x)/μ_y²  -  2 μ_x Cov(x,y)/μ_y³  +  μ_x² Var(y)/μ_y⁴ ]
```

which needs only five sums per arm. That matters for the architecture below.

### CUPED

CUPED replaces the outcome Y with Y minus θ(X minus X̄), where X is the same metric measured before the experiment started and θ = Cov(Y, X) / Var(X). Because X is fixed before treatment, the adjustment has mean zero in both arms and the difference in means stays unbiased. The variance of the adjusted outcome is Var(Y)(1 minus ρ²), where ρ is the correlation between X and Y. A pre-period covariate with ρ = 0.7 roughly halves the variance, which is the same as doubling the sample. A covariate with ρ near 0 does nothing, and the experiments below have one of each.

### sample ratio mismatch and multiple testing

SRM is a Pearson chi-square goodness-of-fit test of arm counts against the designed split. When it fails, the randomiser or the exposure trigger is broken and no metric result should be trusted. Holm is a step-down Bonferroni that controls the family-wise error rate. Benjamini-Hochberg is a step-up rule that controls the false discovery rate. 

### why peeking breaks the t-test, and what mSPRT does instead

A fixed-horizon test promises that if you test once, at a sample size chosen in advance, you reject a true null 5 percent of the time. It promises nothing about testing repeatedly. Under the null the z-statistic, viewed as a function of sample size, behaves like a scaled random walk, and a random walk eventually crosses any fixed boundary. Every extra look is another chance to cross ±1.96. With 50 looks the chances add up to about a third.

The mixture SPRT (Johari, Pekelis and Walsh, 2017) replaces the p-value with a likelihood ratio that is safe to monitor. Take the current estimate Z of the effect with variance s², and put a normal N(0, τ²) prior over the effect under the alternative. Integrating the normal likelihood against that prior gives a closed form

```
Λ = sqrt( s² / (s² + τ²) ) · exp( τ² Z² / (2 s² (s² + τ²)) )
```

Under the null, Λ as a function of sample size is a nonnegative martingale with expectation 1. Ville's inequality says that such a martingale exceeds 1/α at any point, ever, with probability at most α. So the rule "stop the first time Λ reaches 1/α" has type I error at most α no matter how often you look. The always-valid p-value is the running minimum of 1/Λ, and inverting the same inequality in θ gives a confidence sequence that only shrinks.

The price is power. The test has to budget for all future looks, so at any single sample size it is more conservative than a fixed-horizon test there. The parameter τ sets where along the sample size axis the test is most sensitive. I used τ = 0.1 in outcome units, with outcomes of unit standard deviation and a real effect of 0.05 in the power scenario.

### off-policy evaluation

The logs are rounds (x, a, r) where a context x arrived, a behaviour policy π_b chose action a with known probability (the propensity score), and reward r was observed. The question is the expected reward of a different policy π_e.

- **IPS** reweights each logged reward by w = π_e(a|x) / π_b(a|x). It is unbiased when the propensities are right, and its variance grows with the weights.
- **SNIPS** divides by the mean weight instead of n, trading a small bias for lower variance.
- **The direct method (DM)** fits a reward model q̂(x, a) and averages it over π_e. It has low variance and is exactly as biased as the model.
- **Doubly robust (DR)** is DM plus the weighted residual on the logged action, DM + w(r minus q̂(x, a)). If the propensities are right, the correction term has expectation equal to the model's error under π_e, so it cancels that error. If instead the model is right, the correction has mean zero. Either suffices.
- **Switch-DR** keeps the correction only on rounds where w ≤ τ, and falls back to the model where weights are large.

The Open Bandit data is a slate, three items shown per impression, so everything is computed per position following obp's conventions. Each logged round carries a position, and the evaluation policy is an (n, A, L) array of probabilities over A actions at each of L positions.

## architecture

<figure data-figure="diagram:peeking-pipeline"></figure>

The A/B side is two SQL stages and one Python stage. The contract between SQL and Python is a set of frozen dataclasses of per-arm sums (`MeanStats`, `RatioStats`, `CovariateStats`). Every test takes those, and each dataclass has a `from_array` constructor, so the same Welch code runs on SQL output and on numpy arrays in the simulations. That is what lets the SQL layer be tested against a polars reference and the statistics be tested against scipy independently.

The off-policy side reduces every estimator to a mean of per-round terms, with SNIPS as a ratio of two means. One function turns the arrays into a `RowTerms` of per-round weight, reward, DM term and factual q̂. The bootstrap then only resamples those vectors and never touches the (n, A, L) tensors again.

## implementation

Everything statistical is from scratch. scipy supplies only t, normal and chi-square distribution functions and the noncentral t for power. The package is about 1,370 lines of Python across metrics, stats, ope and sim, plus 585 lines of tests.

### the metrics layer

Metrics are SQL expressions over per-user aggregates, so revenue per user, conversion and clicks per session are one line each. The first stage builds one row per exposed user. The part that matters is who counts.

```sql
first_exposure AS (
    SELECT experiment_id, user_id, MIN(exposed_at) AS exposed_at
    FROM exposures GROUP BY experiment_id, user_id
),
units AS (
    SELECT a.experiment_id, a.user_id, a.variant, f.exposed_at, x.start_at, x.end_at,
           x.start_at - to_days(x.pre_period_days) AS pre_start
    FROM assignments a
    JOIN first_exposure f USING (experiment_id, user_id)
    JOIN experiments x USING (experiment_id)
    WHERE f.exposed_at >= x.start_at AND f.exposed_at < x.end_at
),
```

Only users whose first exposure falls inside the experiment window count. Post-period events count from that first exposure to the window end, and pre-period events from `start_at` minus the pre-period length up to `start_at`. The final select wraps every aggregate in `COALESCE(..., 0)`, so an exposed user with no events contributes a zero rather than vanishing from n. That one function call is the difference between measuring the treatment effect and measuring the effect among users who happened to do something. The tests cover exactly these cases (a user first exposed before the window, a user never exposed, an exposed user with no events) against a polars reference implementation of the same metrics.

The second stage reduces `user_metrics` to sums per arm, `SUM(y)`, `SUM(y*y)`, and for ratio and CUPED metrics `SUM(x)`, `SUM(x*x)` and `SUM(x*y)`. SRM is checked twice, once on assignments (is the randomiser healthy) and once on exposed users (is the trigger healthy).

### CUPED from sums alone

Most CUPED code adjusts per-user arrays. Mine had to run on the five sums the SQL returns, so the adjusted outcome's sum of squares is expanded algebraically.

```python
def adjusted(s: CovariateStats) -> MeanStats:
    n = s.n
    sum_adj = s.sum_y - theta * (s.sum_x - n * xbar)
    c = theta * xbar
    sum_adj2 = (s.sum_y2 + theta * theta * s.sum_x2 + n * c * c
                - 2 * theta * s.sum_xy + 2 * c * s.sum_y - 2 * theta * c * s.sum_x)
    return MeanStats(n=n, sum_y=sum_adj, sum_y2=sum_adj2)
```

θ and x̄ are pooled across both arms, so the adjustment cannot create a difference between arms on its own. A hypothesis test checks that CUPED from sums equals CUPED on explicitly adjusted arrays, and another that θ = 0 reduces to plain Welch.

### the mSPRT in a few lines

The log likelihood ratio is vectorised so the peeking simulation can compute it for 200 experiments by 50 looks in one call, and the always-valid p-value is an accumulated minimum along the look axis.

```python
def msprt_log_lr(est, var, tau2, theta0=0.0):
    d = est - theta0
    return 0.5 * np.log(var / (var + tau2)) + tau2 * d * d / (2.0 * var * (var + tau2))

def msprt_always_valid_p(est, var, tau2, theta0=0.0, axis=-1):
    p = np.minimum(1.0, np.exp(-msprt_log_lr(est, var, tau2, theta0)))
    return np.minimum.accumulate(p, axis=axis)
```

A test checks the closed form against numerical integration of the normal likelihood over the N(0, τ²) mixture. A streaming `MSPRTMonitor` keeps the running minimum p-value and intersects each new confidence interval with the previous one, so the sequence only shrinks. It uses plug-in variances, the standard practical choice, not a known-variance version.

### every estimator is a mean of per-round terms

This is the core of the off-policy side.

```python
w = action_dist[idx, action, pos] / pscore
pi_at_pos = action_dist[idx, :, pos]          # (n, A)
q_at_pos  = q_hat[idx, :, pos]                # (n, A)
dm = (q_at_pos * pi_at_pos).sum(axis=1) / pi_at_pos.sum(axis=1)
...
def dr_terms(t):
    return t.dm + t.w * (t.reward - t.q_factual), None

def switch_dr_terms(t, tau):
    keep = (t.w <= tau).astype(float)
    return t.dm + keep * t.w * (t.reward - t.q_factual), None
```

IPS is `w * reward`, SNIPS is the pair (`w * reward`, `w`), DM is `dm`. With that shape, switch-DR at τ = infinity is DR and at τ = 0 is DM, and both limits are tested.

### a bootstrap that is one matrix product

Resampling n rounds with replacement is the same as drawing a multinomial count vector over rounds. One replicate is then a dot product of counts with the per-round terms, and a chunk of replicates is one matrix product.

```python
counts = rng.multinomial(n, p, size=b).astype(float)
top = counts @ num
reps[s : s + b] = top / (counts @ den) if den is not None else top / n
```

For SNIPS this recomputes the ratio inside every replicate, so the interval includes the variability of the normaliser. I also wrote `bootstrap_ci_obp_compatible`, which reproduces obp's random draws exactly, but it resamples already-normalised SNIPS terms and so ignores that variability. It exists for the cross-check, not for the experiments.

### the reward model is cross-fitted

The reward model is one LightGBM classifier for all actions, with the action index as a categorical feature, so items share strength. That matters when Open Bandit has 80 items and a few dozen clicks. Rows are split into three folds, and each fold is predicted by a model trained on the other two, so no round is ever scored by a model that saw its reward. Without cross-fitting, the model's memorisation of the logged rewards leaks straight into DR's residual term. The default is 200 trees at learning rate 0.05, 15 leaves and at least 50 rows per leaf. Those regularisation settings matter for the DM bias story below.

### Thompson sampling slates, vectorised

The ZOZOTOWN evaluation policy is Bernoulli Thompson sampling with a production Beta prior per item. It does not depend on context, so its slate distribution is computed once. Each simulation draws θ_a from Beta(α_a, β_a) for all 80 items, sorts, and records which items land in the top three slots. obp does this in a Python loop. I do 20,000 simulations at a time with one `argsort`, and then broadcast the resulting (80, 3) distribution to every round.

## problems

### 1. the direct method was 11 percent low with a good model

This was the surprise of the project and it is now the main off-policy result. The cross-fitted LightGBM direct method came out 11.0 percent below the true policy value, averaged over 200 replicates, with bootstrap intervals that never covered the truth. My first suspicion was a feature bug, such as the action column being misaligned when scoring counterfactual actions. I checked one replicate by hand. The per-action averages of q̂ were close to the per-action averages of the true q, and yet the direct method was well below truth. (That hand check is in the devlog, and its numbers were not saved to `results/`, so I do not quote them here.)

The explanation is selection on shrinkage. The synthetic bandit's evaluation policy is a softmax of 3 times the log of the true click probability, mixed with 10 percent uniform, so π_e(a|x) is roughly proportional to q(x, a)³. It piles its mass onto the specific context and action pairs where the true reward is highest. A regularised tree model does the opposite at exactly those points. With at least 50 rows per leaf and 200 shallow trees, the model's predictions are pulled toward the mean, so where q is unusually high, q̂ is too low, and where q is unusually low, q̂ is too high. Those errors roughly cancel in a per-action average. They do not cancel in the direct method, which is

```
DM - V = E_x [ sum_a  π_e(a|x) · (q̂(x,a) - q(x,a)) ]
```

and weights the errors by π_e, which is large where the error is negative. Increasing capacity to 600 trees, 31 leaves and 20 rows per leaf barely moved it, according to the devlog, because the model is still regularised and still sees very few logged rounds in the extreme regions π_e cares about.

DR fixes this without a better model. Its correction term, w(r minus q̂(x, a)) on the logged action, has expectation E_x[Σ_a π_e(a|x)(q(x, a) minus q̂(x, a))] when the propensities are correct, which is exactly the negative of the error above. I kept the default model and reported the bias, because this is the textbook reason DR exists, and a concrete demonstration of it is worth more than a tuned model that hides it.

### 2. switch-DR looked broken

Switch-DR at τ = 2 came out 8.5 percent low, which looked like a bug in the threshold logic. It is not. In the first replicate, 13.8 percent of rounds had importance weights above 2, and on those rounds switch-DR drops the correction and uses the model alone. Those rounds are not a random 13.8 percent. A large weight means π_e likes the logged action much more than π_b did, which means a high-q action, which is where DM's error is concentrated. So switch-DR drops the correction on a seventh of the rounds and keeps roughly three quarters of DM's bias. On Open Bandit, where only 1.04 percent of weights exceed τ = 10, it behaves well. The lesson is that τ must be chosen with the weight distribution in view, and ideally tuned from data.

### 3. matching obp's bootstrap bit for bit

To cross-check intervals as well as point estimates I needed obp's exact random stream. obp calls legacy `RandomState(seed).choice(samples, size=n)`, and for uniform sampling with replacement that draws `randint(0, n, size=n)` indices internally. Reproducing that call gives bootstrap bounds equal to obp's within 1e-12 in the test. It also made me look at what obp resamples, which for SNIPS is the already-normalised terms, and that is why the experiments use my multinomial version instead.

### 4. the Thompson sampling slate cannot match to 1e-9

Both libraries compute the BTS slate distribution by Monte Carlo with different generators, so they will never agree to machine precision. I compare them in total variation instead, which came out 0.0118, 0.0093 and 0.0120 for the three slots at 100,000 simulations. For the estimator cross-check both libraries receive the identical `action_dist` array. As a check that the Monte Carlo difference does not matter, I reran every estimator with obp's distribution in place of mine, and no estimate moved by more than 1e-5.

### 5. small ones

pytest tried to collect the `TestResult` dataclass as a test class because of its name, fixed with `__test__ = False`. The first A/A figure had ten long metric names overlapping on the x axis, so it became a horizontal layout, and a `--plot-only` flag redraws it from saved CSVs without regenerating 30 million events.

### 6. a shared machine

The session was paused partway through to reduce load on a shared machine that was running other jobs. On resume I capped BLAS and OpenMP at 2 threads, cut the synthetic OPE default to 2 worker processes and LightGBM to 2 threads, and ran one experiment at a time. The synthetic OPE run in `results/` predates the pause and used 6 workers, which changes wall time but not the numbers, since each replicate is seeded by its index.

### 7. the Open Bandit sample is tiny in click terms

The sample shipped with obp has 10,000 rounds per logging policy, with 38 clicks in the random log and 42 in the BTS log. The estimators are correct, and the cross-check shows they compute exactly what obp computes, but the data cannot tell them apart. I say so in the results rather than reading a ranking into noise.

## experiments

All runs were on an Apple M4 Pro shared with other jobs, with BLAS, OpenMP and LightGBM threads capped at 2 (the synthetic OPE run used 6 worker processes, as noted above). Timings are indicative only. None of these experiments was rerun independently.

1. **A/A through the full DuckDB pipeline.** 1,000 experiments of 2,000 users, 30,568,177 events and 1,800,259 exposed users, five metrics with Welch, CUPED or the delta method, plus both SRM checks, each with a Wilson interval and a Kolmogorov-Smirnov test of p-value uniformity.
2. **Power.** Eleven effect sizes from 0 to 0.2 standard deviations, 1,000 users per arm, 2,000 simulations each, normal and log-normal outcomes.
3. **Coverage.** 1,000 simulations of 2,000 users per arm with nonzero true effects, for Welch on zero-inflated log-normal revenue, the delta method on clicks per session and CUPED on a correlated outcome.
4. **Peeking.** 2,000 experiments per scenario, up to 10,000 unit-variance users per arm, a look every 200 users for 50 looks. The naive analyst stops at the first Welch p below 0.05. The mSPRT uses τ = 0.1 and stops when the always-valid p falls below 0.05. Run under the null and under a true effect of 0.05.
5. **CUPED.** The variance ratio for ρ from 0 to 0.9, 1,000 simulations of 1,000 users per arm each, plus 50 event-log experiments of 4,000 users.
6. **Synthetic OPE.** A bandit with 10 actions, 5-dimensional normal contexts and a nonlinear true click probability. The behaviour policy is a softmax at inverse temperature 0.5 and the evaluation policy at 3.0. The true policy value, 0.50492, comes from 2 million fresh contexts with no reward noise. Each of 200 replicates logs 5,000 rounds, fits the cross-fitted LightGBM and a deliberately weak context-free model (the mean reward of each action), runs every estimator, and builds a 500-replicate bootstrap interval.
7. **Open Bandit.** The all-campaign random-policy log (propensity 1/80 per slot, 3 slots) as the logged data, BTS with the production prior as the evaluation policy, 100,000 Monte Carlo slates, LightGBM with item features, 2,000 bootstrap replicates, and the obp cross-check. The BTS policy's own log gives the on-policy truth to compare against.

## results

### A/A false positive rates are where they should be

Every one of the ten metric and method pairs landed in the 4 to 6 percent band over 1,000 A/A experiments.

| Metric | Method | False positive rate |
|---|---|---|
| Revenue per user | Welch | 5.5% |
| Revenue per user | CUPED | 5.4% |
| Conversion | Welch | 4.2% |
| Conversion | CUPED | 4.3% |
| Sessions per user | Welch | 4.0% |
| Sessions per user | CUPED | 4.0% |
| Clicks per session | delta method | 5.1% |
| Revenue per session | delta method | 5.8% |
| SRM on assignments | chi-square | 4.7% |
| SRM on exposed users | chi-square | 4.8% |

No KS test of p-value uniformity rejected, and the smallest KS p-value was 0.104. Across the family of five primary tests, the chance of at least one false positive in an experiment was 16.8 percent uncorrected, 3.8 percent with Holm and 4.2 percent with Benjamini-Hochberg. That 16.8 is the everyday version of the peeking problem. Five metrics tested at 5 percent each is not a 5 percent test.

The user-metrics SQL processed the 30.6 million events in 12.8 s, about 2.4 million events per second, and the sufficient-statistics query took 1.8 s. Generating and loading the events took 55 s and 49 s, so the SQL is not the bottleneck. These timings are from the shared, thread-capped machine.

### power matches the noncentral t

The largest gap between simulated and analytic power was 0.026 on normal data and 0.031 on log-normal data, at effect 0.1 where the simulation gave 0.640 against an analytic 0.608. 19 of the 22 analytic values fell inside the simulated Wilson interval. At zero effect the simulated rejection rates were 5.45 and 4.65 percent.

### intervals cover, and CUPED more than halves their width

Nominal 95 percent intervals covered the true effect in 95.2 percent of simulations for Welch on revenue, 95.1 for the delta method and 94.0 for CUPED (Wilson 92.4 to 95.3), over 1,000 simulations each. Plain Welch on the same data as CUPED covered 94.4 percent, so CUPED's slightly low number is shared with the unadjusted test on that dataset rather than introduced by the adjustment. On that data the mean CUPED interval width was 0.124 against 0.277 for Welch.

### peeking

<figure data-figure="chart:projects/cuped-msprt-ope-engine/cuped-msprt-ope-engine-peeking"></figure>

Under the null, stopping at the first significant naive test over 50 looks gave a **32.8 percent** false positive rate (Wilson 30.8 to 34.9). The curve rises fastest early, 15.0 percent by the fifth look at 1,000 users per arm, because early estimates are noisy and consecutive looks early on share less data. The fixed-horizon test at the final look alone rejected 5.25 percent. The mSPRT rejected **1.95 percent** (Wilson 1.4 to 2.7), well under its 5 percent guarantee, because Ville's bound covers infinitely many looks and this experiment stops at 50.

The cost shows up under a real effect of 0.05. The mSPRT detected it in 72.75 percent of experiments, stopping on average at 5,112 users per arm among those that stopped. A fixed-horizon test at 10,000 users per arm detected it in 93.75 percent. So the always-valid test buys the right to look whenever you like, and pays about 21 percentage points of power at this horizon and this τ. The naive peeker "detected" it 97.35 percent of the time, but that number means nothing when the same procedure flags a third of null experiments.

### CUPED follows 1 minus rho squared

<figure data-figure="chart:projects/cuped-msprt-ope-engine/cuped-msprt-ope-engine-cuped"></figure>

The simulated variance ratio tracks 1 minus ρ² across the grid, for example 0.831 against 0.84 at ρ = 0.4 and 0.180 against 0.19 at ρ = 0.9. The largest gap, 0.039 at ρ = 0.6, is sampling noise in a ratio of two variances each estimated from 1,000 simulations. The mean ratio of squared standard errors, which uses the analytic formulas rather than the spread of estimates, sits closer still, 0.641 against 0.64 at the same point.

On the event-log generator, where users have a persistent Gamma-distributed activity rate, sessions per user has a pooled pre-post correlation of 0.723. Theory predicts a variance ratio of 0.4778 and the observed ratio of squared standard errors was 0.4778. Revenue per user has ρ = 0.036, because purchases are rare and revenue is heavy tailed, so CUPED does essentially nothing there (0.998). That is a useful thing to tell a product team. CUPED is not a free halving of sample size, it is exactly as good as the covariate.

### synthetic off-policy evaluation

<figure data-figure="chart:projects/cuped-msprt-ope-engine/cuped-msprt-ope-engine-synthetic-ope"></figure>

Relative bias, relative RMSE and interval coverage over 200 replicates, against a true value of 0.50492.

| Estimator | Reward model | Bias | RMSE, percent of truth | CI coverage |
|---|---|---|---|---|
| IPS | none | minus 0.25% | 2.3% | 95.0% |
| SNIPS | none | minus 0.10% | 1.7% | 97.0% |
| DM | LightGBM | minus 11.0% | 11.1% | 0% |
| DR | LightGBM | minus 0.02% | 1.6% | 96.0% |
| Switch-DR, τ = 2 | LightGBM | minus 8.5% | 8.7% | 0% |
| DM | weak | minus 24.4% | 24.4% | 0% |
| DR | weak | minus 0.17% | 1.7% | 96.5% |
| Switch-DR, τ = 2 | weak | minus 18.1% | 18.1% | 0% |

DR with LightGBM is the best estimator here, with the lowest RMSE and nominal coverage, and it stays unbiased with the weak model that makes DM 24.4 percent low. That is double robustness working as advertised. The weak model ignores context entirely, so its error is large and strongly negative exactly where π_e concentrates, and DR's correction still removes it, at the cost of a slightly wider spread.

The coverage column carries the other lesson. DM's bootstrap intervals are tight and wrong. The bootstrap resamples rounds with q̂ held fixed, so it measures the variance of averaging a fixed model over contexts, about 0.008 standard deviation, and nothing about the model's bias, which is about 0.056. An interval like that covered the truth in 0 of 200 replicates. A tight interval around a biased estimate is the failure mode to warn people about, because it looks like confidence.

In this environment the importance weights are well behaved (in the first replicate the mean weight was 0.976, the 99th percentile 3.53 and the maximum 4.18), which is why plain IPS and SNIPS also do well. DR's advantage over SNIPS here is small. With heavier weights it would matter more, and with a worse model DM's disadvantage would too.

### Open Bandit, a sanity check and not a ranking

<figure data-figure="chart:projects/cuped-msprt-ope-engine/cuped-msprt-ope-engine-obd"></figure>

The BTS policy's own log gives an on-policy click rate of 0.0042, with a bootstrap 95 percent interval of 0.0030 to 0.0055. The random policy's own click rate is 0.0038. From the random logs alone, the estimates of the BTS value were

| Estimator | Estimate | 95% bootstrap CI | Relative error |
|---|---|---|---|
| IPS | 0.00455 | 0.00149 to 0.00950 | 8.4% |
| SNIPS | 0.00478 | 0.00157 to 0.00987 | 13.7% |
| DM | 0.00475 | 0.00470 to 0.00479 | 13.0% |
| DR | 0.00486 | 0.00168 to 0.00984 | 15.7% |
| Switch-DR, τ = 10 | 0.00461 | 0.00330 to 0.00621 | 9.8% |

All five fall inside the on-policy interval. That is the claim, and it is a weak one. IPS, SNIPS and DR have intervals roughly 0.0015 to 0.0099 wide, more than three times the width of the on-policy interval they are being compared with, because the random log has 38 clicks. With that little signal, IPS having the smallest relative error here is luck, not evidence that IPS is the best estimator for this problem. DM's narrow interval is the same illusion as in the synthetic study, a fixed model resampled, and it says nothing about the model's error.

The switch-DR threshold sweep moves the estimate between 0.00454 and 0.00486 as τ goes from 1 to 20, and at τ = 20 it equals DR exactly, because the largest weight in the log is 19.66. Only 1.04 percent of weights exceed τ = 10.

The cross-check is the strong claim on this data. With identical inputs, IPS, DM, DR and switch-DR agreed with obp exactly, and SNIPS differed by 1.7e-18, one unit in the last place. My vectorised BTS simulation took 0.42 s against obp's 1.92 s at 100,000 simulations, on the shared machine.

## what I would change

### run the full Open Bandit Dataset

The full release has about 26 million rounds. With thousands of clicks, the Open Bandit comparison would separate the estimators and let me report relative error across campaigns and bootstrap seeds rather than one point per estimator. That is the single most informative next run.

### tune the switch-DR threshold from data

A fixed τ is a guess. obp's tuning variants choose τ by minimising an estimated MSE, and the synthetic results show why that matters, since τ = 2 kept most of DM's bias.

### close the DM gap with a better reward model

A per-action linear head on context, or calibrated predictions, may reduce the shrinkage bias. The experiment is to add each and report which one closes the gap, keeping DR as the reference.

### put the mSPRT on the SQL layer

Continuous monitoring should read cumulative per-day sums straight from DuckDB rather than from numpy simulations. A group sequential design with alpha spending would be the natural comparison, since it buys back power when the number of looks is known in advance.

### speed up generation and loading

Generating and loading the A/A events took 55 s and 49 s against 13 s for the SQL itself. Writing Parquet once and letting DuckDB read it directly would remove most of that.

## reproducibility

The project uses uv and pins Python 3.12.

```sh
cd projects/cuped-msprt-ope-engine
uv sync --group dev --group crosscheck      # crosscheck installs obp, used only for cross-checks
uv run python scripts/fetch_obd.py          # copies the OBD sample shipped with obp into data/obd
uv run --group crosscheck pytest -q         # 38 tests, obp tests are skipped without the group
```

The experiments, with threads capped as they were for the reported runs.

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
cd experiments
uv run python 01_aa_fpr.py --n-experiments 1000 --n-users 2000
uv run python 02_power.py --sims 2000 --n 1000
uv run python 03_coverage.py --sims 1000 --n 2000
uv run python 04_peeking.py --sims 2000 --max-n 10000 --batch 200
uv run python 05_cuped.py --sims 1000 --n 1000
uv run python 06_synthetic_ope.py --reps 200 --n 5000 --workers 2
uv run --group crosscheck python 07_obd_ope.py
```

`01_aa_fpr.py --plot-only` redraws its figures from saved results. The full Open Bandit run, when there is disk for it, is

```sh
curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
unzip open_bandit_dataset.zip -d data/obd_full
uv run --group crosscheck python experiments/07_obd_ope.py --data-dir data/obd_full/open_bandit_dataset
```

Code is in `projects/cuped-msprt-ope-engine`, with the metrics layer in `src/expope/metrics`, the A/B statistics in `src/expope/stats`, the estimators, reward model and Open Bandit loader in `src/expope/ope`, the simulators in `src/expope/sim`, the numbered experiment scripts in `experiments/`, every CSV and JSON behind this post in `results/`, the invariants and trade-offs in `DESIGN.md`, and the build log in `DEVLOG.md`.
