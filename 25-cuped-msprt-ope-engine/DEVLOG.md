# DEVLOG: an experimentation and off-policy evaluation engine

### What I wanted to build

Product data science and causal ML teams spend most of their time on two questions. First, did the A/B test work, and can I trust the p-value? Second, before I ship a new recommendation policy, what would it have scored on traffic the old policy served? I wanted one small engine that answers both, written from scratch so I understand every variance formula, and checked against known truth instead of against my own confidence.

The v0 goal was: a DuckDB metrics layer over a raw event log, the standard A/B toolkit (Welch, delta method, CUPED, SRM, Holm, Benjamini-Hochberg, an always-valid mixture SPRT), the standard off-policy estimators (IPS, SNIPS, direct method, doubly robust, switch-DR) with bootstrap intervals, a simulation harness that measures false positive rate, power, coverage and variance reduction against theory, a real-data test on the Open Bandit Dataset, and an exact cross-check against Open Bandit Pipeline (obp).

### Theory

Welch. For per-user means the difference has variance s_c^2/n_c + s_t^2/n_t, and the Welch-Satterthwaite degrees of freedom handle unequal variances. Power at effect d with equal n and sigma comes from the noncentral t with noncentrality d / (sigma sqrt(2/n)).

Delta method. A ratio metric such as clicks per session is sum(x) / sum(y) over users, not a mean of per-user ratios, and users are the randomisation unit, so the sessions inside a user are correlated. A first-order Taylor expansion of R = xbar / ybar gives Var(R) about (1/n) [Var(x)/mu_y^2 minus 2 mu_x Cov(x, y)/mu_y^3 plus mu_x^2 Var(y)/mu_y^4]. Treating sessions as independent would understate this.

CUPED. Subtracting theta (X minus mean X) from the outcome, with X a pre-period covariate and theta = Cov(Y, X) / Var(X), leaves the mean difference unbiased (X is fixed before treatment) and scales the variance by 1 minus rho squared.

SRM. A Pearson chi-square goodness-of-fit test of arm counts against the design split. If it fails, the randomiser or the trigger is broken and no metric result should be trusted.

Multiple testing. Holm is a step-down Bonferroni that controls family-wise error. Benjamini-Hochberg is a step-up rule that controls the false discovery rate.

Mixture SPRT. For an estimate Z with variance s^2 and a N(0, tau^2) mixing distribution over the effect, the mixture likelihood ratio has a closed form, Lambda = sqrt(s^2/(s^2 + tau^2)) exp(tau^2 Z^2 / (2 s^2 (s^2 + tau^2))). It is a nonnegative martingale under the null, so by Ville's inequality P(sup Lambda at least 1/alpha) is at most alpha. The always-valid p-value is the running minimum of 1/Lambda, and inverting the same inequality gives a confidence sequence.

Off-policy evaluation. With logged (x, a, r) from a behaviour policy pi_b and an evaluation policy pi_e, IPS weights each reward by w = pi_e(a|x) / pi_b(a|x) and is unbiased when pscore is correct. SNIPS divides by the mean weight, trading a small bias for lower variance. The direct method averages a reward model over pi_e and is only as good as the model. Doubly robust adds the weighted residual w (r minus q_hat(x, a)) to the direct method, so it is unbiased if either the propensities or the model are correct. Switch-DR drops the correction for rounds with w above tau, falling back to the model where weights are large. For slates (OBD shows 3 items) everything is per position, following obp's conventions.

### Architecture

The A/B side is a three-stage pipeline. Raw tables (experiments, assignments, exposures, events) live in DuckDB. One SQL query builds `user_metrics`: first exposure per user, post-period aggregates from first exposure to window end, and the same aggregates over the pre-period window for CUPED. A second SQL query reduces that to per-arm sums (n, sum y, sum y squared, and for ratio or CUPED metrics sum x, sum x squared, sum xy). The Python statistics only ever see these few numbers. Metrics are SQL expressions, so revenue per user, conversion and clicks per session are one line each.

The OPE side follows obp's array conventions: reward, action, pscore, position and an (n, A, L) action_dist. One function turns these into per-round terms (weight, reward, the DM term, the factual q_hat), and each estimator is a mean of those terms. The bootstrap resamples rounds on the term vectors. The reward model is a cross-fitted LightGBM with the action index as a categorical feature, so no round is scored by a model that saw it.

DESIGN.md has the diagram, the invariants and the trade-offs.

### Implementation

Everything statistical is from scratch; scipy is used only for t, normal and chi-square distribution functions and for the noncentral t in the power formula.

- `stats/sufficient.py` holds the three sufficient-statistic dataclasses. Every test takes these, which is what lets the SQL layer and the numpy simulations share one implementation.
- `stats/abtest.py` has Welch, the delta method, CUPED computed entirely from sums (the adjusted outcome's sum of squares is expanded algebraically, so CUPED also runs on SQL output), SRM, Holm, BH, the vectorised mSPRT log likelihood ratio, always-valid p-values and confidence-sequence half widths, a streaming `MSPRTMonitor`, and analytic power.
- `metrics/engine.py` generates the SQL from metric definitions, loads polars frames through Arrow, and `analyze()` produces one row per (experiment, metric, method) with both SRM p-values attached.
- `ope/estimators.py` has the estimators, an obp-compatible bootstrap that reproduces obp's `RandomState.choice` draws, and a faster multinomial-count bootstrap that recomputes SNIPS as a ratio in each replicate.
- `ope/obd.py` loads the OBD CSVs with polars (one-hot user features, affinity features, positions mapped to 0..2), parses the ZOZOTOWN BTS prior YAML by hand, and computes the BTS slate distribution with a vectorised Monte Carlo.
- `sim/eventlog.py` generates event logs with a Gamma-distributed per-user activity rate, so pre-period sessions really predict post-period sessions, plus Beta click propensities and log-normal revenue for heavy tails. `sim/bandit.py` is a contextual bandit with a nonlinear true reward and softmax behaviour and evaluation policies at different temperatures.

Tests (38, all passing): Welch and SRM and BH match scipy under hypothesis-generated inputs; delta-method variance equals the variance of the linearised statistic and matches simulated sampling variance; CUPED from sums equals CUPED on adjusted arrays; the mSPRT closed form matches numerical integration of the mixture; the SQL layer matches a polars reference implementation of the same metrics, including edge cases (a user first exposed before the window, a user never exposed, an exposed user with no events); OPE estimators match a hand-computed example, satisfy switch-DR limits (tau = infinity is DR, tau = 0 is DM) and on-policy identities; and every estimator matches obp to 1e-9 on 25 random slate problems.

### Problems

- The DM bias with a good reward model surprised me. In the synthetic study the cross-fitted LightGBM DM was 11.0 percent low (results/synthetic_ope.csv). I first suspected a feature bug, so I checked one replicate by hand: the per-action means of q_hat were close to the truth (for example 0.378 vs 0.330 and 0.169 vs 0.189), yet DM gave 0.450 against a truth of 0.504. The bias is selection on the model's shrinkage: pi_e concentrates on the context-action pairs where the true q is highest, and exactly there a regularised tree model pulls predictions toward the mean. Tripling capacity (600 trees, 31 leaves, min_child_samples 20) only moved DM to 0.456. I kept the default model and reported the bias, since this is the textbook reason DR exists; DR with the same model was unbiased.
- Switch-DR looked broken at first, with 8.5 percent bias. It is not: with tau = 2, 13.8 percent of rounds have weights above tau (results/synthetic_ope_summary.json, weights_rep0), so on those rounds it inherits DM's bias. On OBD, where only 1.0 percent of weights exceed tau = 10, it behaves well. The lesson for the write-up is that tau must be chosen with the weight distribution in view.
- Matching obp's bootstrap bit for bit needed obp's exact random stream. obp calls legacy `RandomState(seed).choice(samples, size=n)`, which draws `randint(0, n, size=n)` indices internally, so `bootstrap_ci_obp_compatible` does the same. That function resamples already-normalised SNIPS terms, which ignores the normaliser's variability, so the experiments use my multinomial bootstrap instead.
- The BTS slate distribution cannot match obp to 1e-9 because both are Monte Carlo with different generators. I compare it in total variation (0.012, 0.009 and 0.012 per slot at 100,000 simulations) and feed both libraries the identical action_dist array for the estimator cross-check. The policy value moves by less than 1e-5 when obp's distribution is used instead of mine (results/obd_ope_summary.json).
- pytest tried to collect my `TestResult` dataclass as a test class because of its name. Setting `__test__ = False` on it fixed the warning.
- The first A/A figure had ten long metric names on the x axis and they overlapped. I switched to a horizontal layout and added `--plot-only` so the figure can be redrawn from saved CSVs without rerunning 30 million events. The OBD legend sat on top of the SNIPS error bar, so I moved it under the axes.
- The session was paused partway through to reduce load on a shared machine. On resume I capped BLAS and OpenMP threads at 2, cut the synthetic OPE default to 2 worker processes and LightGBM to 2 threads, and ran one experiment at a time.
- The OBD sample is tiny in click terms: 38 clicks in 10,000 random-policy rounds and 42 in 10,000 BTS rounds. The estimators are correct but the data cannot distinguish them; I say so in Results rather than over-reading it.

### Experiments

All runs on an Apple M4 Pro with threads capped at 2.

1. A/A through the full DuckDB pipeline (`experiments/01_aa_fpr.py --n-experiments 1000 --n-users 2000`). 1,000 independent experiments, 30,568,177 events, 1,800,259 exposed users. Five metrics (revenue per user, conversion, sessions per user with Welch and CUPED; clicks per session and revenue per session with the delta method), plus both SRM checks. Outputs: results/aa_fpr.csv, results/aa_summary.json, results/raw/aa_results.csv, results/figures/aa_fpr.png and aa_pvalue_hist.png.
2. Power (`02_power.py --sims 2000 --n 1000`). Eleven effect sizes from 0 to 0.2 standard deviations, normal and skewed log-normal outcomes, compared with the noncentral t. Outputs: results/power.csv, results/power_summary.json, results/figures/power_curve.png.
3. Coverage (`03_coverage.py --sims 1000 --n 2000`) with nonzero true effects: Welch on zero-inflated log-normal revenue, the delta method on clicks per session, CUPED on a correlated outcome. Outputs: results/coverage.csv, results/coverage_summary.json.
4. Peeking (`04_peeking.py --sims 2000 --max-n 10000 --batch 200`). 50 looks per experiment, naive Welch stopping at the first p below 0.05 against the mSPRT with tau = 0.1, under the null and under an effect of 0.05. Outputs: results/peeking.csv, results/peeking_summary.json, results/figures/peeking_fpr.png.
5. CUPED (`05_cuped.py --sims 1000 --n 1000`). Variance ratio over rho from 0 to 0.9, plus CUPED on 50 event-log experiments of 4,000 users. Outputs: results/cuped_variance.csv, results/cuped_summary.json, results/figures/cuped_variance.png.
6. Synthetic OPE (`06_synthetic_ope.py --reps 200 --n 5000`). 10 actions, 5-dimensional contexts, true policy value 0.50492 from 2 million contexts. Each replicate fits a cross-fitted LightGBM and a deliberately weak per-action-mean model, runs every estimator and a 500-replicate bootstrap. Outputs: results/synthetic_ope.csv, results/synthetic_ope_summary.json, results/raw/synthetic_ope_reps.csv, results/figures/synthetic_ope_error.png. This run used 6 workers before the pause; the default is now 2.
7. Open Bandit Dataset (`07_obd_ope.py`). Random-policy logs (pscore 1/80, 3 slots), BTS evaluation policy from the production prior, 100,000 Monte Carlo slates, LightGBM with item features, 2,000 bootstrap replicates, plus the obp cross-check. Outputs: results/obd_ope.csv, results/obd_ope_summary.json, results/figures/obd_ope.png, results/raw/07_obd_ope.log.

### Results

A/A (results/aa_fpr.csv, results/aa_summary.json). Every one of the ten metric and method pairs is in the 4 to 6 percent band: revenue per user 5.5 (Welch) and 5.4 (CUPED), conversion 4.2 and 4.3, sessions per user 4.0 and 4.0, clicks per session 5.1, revenue per session 5.8, SRM on assignments 4.7 and on exposures 4.8. No Kolmogorov-Smirnov test of p-value uniformity rejects (smallest p 0.10). Across the family of 5 primary tests the chance of at least one false positive is 16.8 percent uncorrected, 3.8 percent with Holm and 4.2 percent with BH. The SQL stage that builds per-user metrics processed the 30.6 million events in 12.8 s (about 2.4 million events per second), and the sufficient-statistics query took 1.8 s.

Power (results/power.csv, results/power_summary.json). Largest gap between simulated and analytic power is 0.026 for normal data and 0.031 for log-normal data (at effect 0.1, 0.640 simulated vs 0.608 analytic); 19 of 22 analytic values fall inside the simulated Wilson interval. At zero effect the simulated rejection rates are 5.45 and 4.65 percent.

Coverage (results/coverage.csv). Nominal 95 percent intervals covered the truth 95.2 percent of the time for Welch, 95.1 for the delta method and 94.0 for CUPED, over 1,000 simulations each. On the same data CUPED's mean interval width was 0.124 against 0.277 for Welch.

Peeking (results/peeking_summary.json). Under the null, stopping at the first significant naive test over 50 looks produced a 32.8 percent false positive rate (Wilson 30.8 to 34.9); the fixed-horizon test at the final look alone was 5.25 percent. The mSPRT rejected 1.95 percent of A/A tests. With a real effect of 0.05 the mSPRT found it 72.8 percent of the time, stopping on average at 5,112 users per arm, against 93.8 percent for a fixed-horizon test at 10,000. That is the price of being allowed to look whenever you like.

CUPED (results/cuped_variance.csv, results/cuped_summary.json). The variance ratio tracks 1 minus rho squared across the grid, for example 0.831 vs 0.84 at rho 0.4 and 0.180 vs 0.19 at rho 0.9. On the event-log data, sessions per user has pooled rho 0.723, theory predicts a ratio of 0.4778, and the observed ratio of squared standard errors is 0.4778. Revenue per user has rho 0.036, so CUPED does essentially nothing there (0.998), which is itself a useful thing to report to a product team.

Synthetic OPE (results/synthetic_ope.csv). Relative bias and relative RMSE over 200 replicates, truth 0.50492:

| Estimator | Reward model | Bias | RMSE / truth | CI coverage |
|---|---|---|---|---|
| IPS | none | minus 0.25 percent | 2.3 percent | 95.0 percent |
| SNIPS | none | minus 0.10 percent | 1.7 percent | 97.0 percent |
| DM | LightGBM | minus 11.0 percent | 11.1 percent | 0 percent |
| DR | LightGBM | minus 0.02 percent | 1.6 percent | 96.0 percent |
| Switch-DR, tau 2 | LightGBM | minus 8.5 percent | 8.7 percent | 0 percent |
| DM | weak | minus 24.4 percent | 24.4 percent | 0 percent |
| DR | weak | minus 0.17 percent | 1.7 percent | 96.5 percent |

DR is the best estimator here and it stays unbiased even with the weak model, which is double robustness working as advertised. The bootstrap intervals only cover when the estimator is unbiased; a tight interval around a biased DM is the failure mode to warn people about.

Open Bandit Dataset (results/obd_ope.csv, results/obd_ope_summary.json). The on-policy BTS click rate is 0.0042 (bootstrap 95 percent CI 0.0030 to 0.0055); the random policy's own click rate is 0.0038. From the random logs: IPS 0.00455 (8.4 percent relative error), SNIPS 0.00478 (13.7), DM 0.00475 (13.0), DR 0.00486 (15.7), switch-DR with tau 10 0.00461 (9.8). All five fall inside the on-policy interval. IPS, SNIPS and DR have bootstrap intervals roughly 0.0015 to 0.0099 wide, which with 38 clicks is honest: this sample cannot rank the estimators. Every estimator matched obp with a maximum absolute difference of 1.7e-18, and my vectorised BTS simulation took 0.43 s against obp's 1.91 s.

### What I would change

- Run the full Open Bandit Dataset (about 26 million rounds). With thousands of clicks the OBD comparison would actually separate the estimators, and it would let me report relative error across campaigns and bootstrap seeds instead of one point.
- Tune the switch-DR threshold from data, for example by minimising an estimated MSE the way obp's tuning variants do, rather than fixing tau.
- Reduce DM's shrinkage bias with a better reward model: a per-action linear head on context, or calibrated predictions, and report which one closes the gap.
- Move the mSPRT onto the SQL layer so continuous monitoring reads cumulative sums per day straight from DuckDB, and add a group sequential design for comparison.
- Speed up event generation and loading, which took 55 s and 49 s of the A/A run against 13 s for the SQL itself; writing Parquet once and letting DuckDB read it directly would remove most of that.
- Add the roadmap items: offline evaluation of item-kNN, two-tower and SASRec recommenders on MovieLens-1M, off-policy learning, heterogeneous treatment effects, interleaving and switchback designs, and a results dashboard.

### Verification

On 2026-09-27 an independent rerun of the full test suite passed all 38 tests, including the obp cross-checks at 1e-9. The simulation experiments were not rerun independently.
