# Design

### Data flow

```
                         A/B side                                            OPE side

 sim/eventlog.py or production logs                          sim/bandit.py or Open Bandit Dataset CSVs
            |                                                              |
            v                                                              v
 +-------------------------------+                         +-------------------------------+
 | DuckDB raw tables             |                         | arrays, obp conventions       |
 | experiments  assignments      |                         | reward (n,) action (n,)       |
 | exposures    events           |                         | pscore (n,) position (n,)     |
 +-------------------------------+                         | action_dist (n, A, L)         |
            | SQL: first exposure, post and pre windows    +-------------------------------+
            v                                                    |                 |
 +-------------------------------+                               |   ope/reward_model.py
 | user_metrics (temp table)     |                               |   K-fold cross-fit LightGBM
 | one row per exposed user:     |                               |   q_hat (n, A, L)
 | sessions clicks purchases     |                               v                 v
 | revenue, pre_* copies         |                         +-------------------------------+
 +-------------------------------+                         | row_terms: RowTerms           |
            | SQL: SUM(y), SUM(y*y), SUM(x*y) per arm      | w, reward, dm, q_factual      |
            v                                              +-------------------------------+
 +-------------------------------+                               |                 |
 | suff_stats: O(1) per arm      |                               v                 v
 | MeanStats RatioStats          |                     estimator_terms()     bootstrap_ci()
 | CovariateStats                |                     IPS SNIPS DM DR       multinomial
 +-------------------------------+                     switch-DR             resampling of rounds
            | Python (stats/abtest.py)                          |                 |
            v                                                   v                 v
 Welch, delta, CUPED, SRM, Holm, BH, mSPRT             policy value + percentile CI
            |                                                   |
            +---------------------> results/ <------------------+
                                       ^
                          obp cross-check on identical arrays (tests, experiment 07)
```

### Key data structures

- Event-log tables (`src/expope/metrics/schema.sql`). `experiments` holds the design (variants, treatment share, window, pre-period length). `assignments` has one row per (experiment, user). `exposures` can have many rows per user; analysis uses the first. `events` is a generic stream of (user, type, ts, value); nothing is pre-aggregated.
- `UserAggregate`, `MeanMetric`, `RatioMetric` (`metrics/engine.py`). Metrics are SQL expressions over `user_metrics` columns. A mean metric may name a pre-period covariate, which turns on CUPED.
- `MeanStats`, `RatioStats`, `CovariateStats` (`stats/sufficient.py`). Frozen dataclasses of per-arm sums. They are the contract between SQL and Python: every test in `abtest.py` takes these, and each has a `from_array` constructor so the same test runs on raw numpy arrays.
- `TestResult`. Estimate, standard error, statistic, p-value, CI and degrees of freedom (infinity means z-test).
- `MSPRTMonitor`. Streaming state for the always-valid test: running minimum p-value and the intersected confidence sequence.
- `RowTerms` (`ope/estimators.py`). Per-round weight, reward, DM term and factual q. Every estimator is a mean of per-round terms (SNIPS is a ratio of two means), so the bootstrap only resamples these vectors and never recomputes the (n, A, L) tensors.

### Invariants

- Effects are always treatment minus control.
- Only users whose first exposure falls in the experiment window enter `user_metrics`; post-period events count from first exposure to window end, pre-period events from `start_at - pre_period_days` to `start_at`. Users with no events get zeros, not NULLs (COALESCE), so they still count in n.
- SRM is checked twice: on assignments (randomiser health) and on exposed users (trigger health).
- CUPED theta and the covariate mean are pooled across arms, so the adjustment cannot create a difference between arms by itself.
- The mSPRT p-value is non-increasing over looks and the confidence sequence only shrinks.
- `pscore > 0`, `action_dist` has shape (n, A, L) and sums to 1 over actions at each position; `_validate` rejects violations.
- The reward model never scores a round it was trained on (cross-fitting).
- Array conventions match obp exactly, so the cross-check passes the same objects to both libraries.

### Trade-offs chosen and rejected

- Sufficient statistics in SQL, statistics in Python. Chosen because grouping 30M events is what DuckDB is good at, and the tests need only five sums per arm. Rejected: pulling per-user rows into Python (slower, more memory) and writing the tests in SQL (hard to test, hard to read). Cost: tests that need order statistics (medians, quantile treatment effects) do not fit this contract yet.
- Delta method as a z-test, Welch as a t-test. Per-user samples here are in the thousands so the difference is cosmetic; keeping Welch exact lets the tests compare it to scipy to machine precision.
- Mixture SPRT with a normal mixing distribution (Johari et al. 2017). Chosen for its closed form and its confidence sequence. Rejected for v0: group sequential alpha spending, which needs the number of looks planned in advance.
- Plain Python loop plus numpy for OPE, not a class hierarchy like obp. A function per estimator over `RowTerms` is shorter and made the bootstrap trivial.
- Bootstrap by multinomial counts. One replicate is a dot product with the per-round terms, and SNIPS is recomputed as a ratio in each replicate. A second function, `bootstrap_ci_obp_compatible`, reproduces obp's legacy `RandomState.choice` draws exactly so the intervals can also be cross-checked.
- Cross-fitted LightGBM with the action as a categorical feature. One model for all actions shares strength across items, which matters when OBD has 80 items and a few dozen clicks. Rejected: one model per action (too little data per item).
- BTS slate distribution computed once by vectorised Monte Carlo and broadcast to every round, because the ZOZOTOWN BTS policy is context-free. This is 4 times faster than obp's loop at 100,000 simulations.
- DuckDB in-process, no server. Enough for tens of millions of events on one machine and trivially testable.
