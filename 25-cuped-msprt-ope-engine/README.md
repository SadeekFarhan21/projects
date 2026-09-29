# expope: experimentation and off-policy evaluation engine

A small, verified engine for the two questions a product data science or causal ML team answers every week:

1. Did this A/B test move the metric? A DuckDB metrics layer turns a raw event log (assignments, exposures, events) into per-arm sufficient statistics in SQL, and from-scratch statistics code (Welch, delta method, CUPED, SRM, Holm, Benjamini-Hochberg, mixture SPRT) turns those sums into effects, intervals and p-values.
2. What would a new policy have scored, using only logs from the old one? From-scratch off-policy estimators (IPS, SNIPS, direct method with a LightGBM reward model, doubly robust, switch-DR) with bootstrap intervals, evaluated on a synthetic bandit with known truth and on the Open Bandit Dataset from ZOZOTOWN.

Everything is checked against known truth by simulation, and every OPE estimator agrees with Open Bandit Pipeline (obp) to 1e-9 on identical inputs (observed max gap 1.7e-18).

### Layout

```
src/expope/
  metrics/   schema.sql, engine.py      DuckDB event-log schema, SQL metric definitions, analyze()
  stats/     sufficient.py, abtest.py   sufficient statistics and every A/B test, from scratch
  ope/       estimators.py              IPS, SNIPS, DM, DR, switch-DR, bootstrap
             reward_model.py            cross-fitted LightGBM q(x, a, position)
             obd.py                     Open Bandit Dataset loader and BTS slate distribution
  sim/       eventlog.py, bandit.py     synthetic generators with known ground truth
experiments/ 01 to 07                   the scripts that produced results/
tests/                                  pytest and hypothesis, including the obp cross-check
scripts/fetch_obd.py                    populates data/obd (gitignored)
results/                                CSV and JSON outputs, results/figures PNGs, results/raw logs
```

### Setup

Requires uv. The project pins Python 3.12 (`.python-version`).

```
cd projects/25-cuped-msprt-ope-engine
uv sync --group dev --group crosscheck      # crosscheck installs obp, used only for cross-checks
uv run python scripts/fetch_obd.py          # copies the OBD sample shipped with obp into data/obd
```

There is no compiled component, so build is just `uv sync`.

### Test

```
uv run --group crosscheck pytest -q         # 38 tests, about 6 seconds
```

Without the crosscheck group the obp tests are skipped, not failed.

### Run the experiments

The machine was shared, so all runs below used capped threads:

```
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
cd experiments
uv run python 01_aa_fpr.py --n-experiments 1000 --n-users 2000   # about 2.5 min, 30.6M events through DuckDB
uv run python 02_power.py --sims 2000 --n 1000
uv run python 03_coverage.py --sims 1000 --n 2000
uv run python 04_peeking.py --sims 2000 --max-n 10000 --batch 200
uv run python 05_cuped.py --sims 1000 --n 1000
uv run python 06_synthetic_ope.py --reps 200 --n 5000 --workers 2
uv run --group crosscheck python 07_obd_ope.py
```

`01_aa_fpr.py --plot-only` redraws its figures from saved results.

### Headline results

| Check | Target | Result | File |
|---|---|---|---|
| A/A false positive rate, 10 metric and method pairs, 1,000 experiments | 4 to 6 percent | 4.0 to 5.8 percent | results/aa_fpr.csv |
| Family of 5 metrics, any false positive | Holm and BH bring it near 5 percent | 16.8 raw, 3.8 Holm, 4.2 BH | results/aa_summary.json |
| Welch power vs noncentral t, 22 points | close to theory | max gap 0.031, 19 of 22 analytic values inside the Wilson interval | results/power.csv |
| 95 percent interval coverage | about 95 percent | Welch 95.2, delta 95.1, CUPED 94.0 | results/coverage.csv |
| Peeking at 50 looks, A/A | mSPRT at or below 5 percent | naive t 32.8, mSPRT 1.95 | results/peeking_summary.json |
| CUPED variance ratio vs 1 minus rho squared | on the curve | rho 0.72 on event logs, theory 0.4778, observed 0.4778 | results/cuped_summary.json |
| Synthetic OPE, 200 replicates | DR unbiased, CIs cover | DR bias minus 0.02 percent, coverage 96 percent | results/synthetic_ope.csv |
| OBD, BTS value from random logs | inside on-policy CI | all 5 estimators inside [0.0030, 0.0055] | results/obd_ope.csv |
| Agreement with obp | 1e-9 | 1.7e-18 on OBD, 1e-9 asserted in 25 hypothesis cases | results/obd_ope_summary.json, tests/test_obp_crosscheck.py |

### Milestone status

| Milestone | Status |
|---|---|
| DuckDB metrics layer: schema, mean, ratio and revenue-per-user metrics, SQL sufficient statistics | done |
| A/B statistics: Welch, delta method, CUPED, SRM, Holm, BH, mixture SPRT | done |
| OPE: IPS, SNIPS, DM (LightGBM), DR, switch-DR, bootstrap intervals | done |
| Simulation harness: A/A FPR, power, coverage, peeking, CUPED theory | done |
| Open Bandit Dataset evaluation (obp's 10,000-row sample per policy) | done |
| Cross-check against obp | done |
| OBD full dataset (about 26M rounds) | not started, command below |
| Recommender policies (item-kNN, two-tower, SASRec on MovieLens-1M) evaluated offline | later |
| Contextual bandits and off-policy learning | later |
| Heterogeneous treatment effects | later |
| Interleaving and switchback designs | later |
| Results dashboard | later |

### What was cut from v0

- The Open Bandit experiment uses the 10,000-row-per-policy sample that ships with obp, not the full multi-gigabyte release. With only 38 clicks in the random log the estimates are real but their intervals are wide. Full run, when there is time and disk:
  ```
  curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
  unzip open_bandit_dataset.zip -d data/obd_full
  uv run --group crosscheck python experiments/07_obd_ope.py --data-dir data/obd_full/open_bandit_dataset
  ```
- The BTS slate distribution is Monte Carlo in both libraries, so it is compared with obp in total variation (about 0.01 per slot), not to 1e-9. The estimator cross-check feeds both libraries the same action_dist array.
- The mSPRT uses plug-in variances, the standard practical choice; an exact known-variance version is not included.
