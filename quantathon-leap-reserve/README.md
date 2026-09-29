# Columbus LEAP reserve: cohort Monte Carlo for a lead service-line loan fund

Port of a two-person Quantathon 2026 (SIAM-MTI challenge) entry. Upstream repo:
https://github.com/SadeekFarhan21/Columbis-LEAP (renamed from quantathon-2026; branch `main`, HEAD `197702c`, 2026-03-29).
No LICENSE file exists upstream, so none is added here.

**Question.** How much should Columbus reserve in 2026 so the LEAP lead/galvanized service-line
loan fund stays solvent through the 2037 federal replacement deadline? The loan is 0%, up to
$10,000, 99-year term, repaid in one lump sum on sale or equity withdrawal (not if inherited and
kept). Compare that with a grant.

**Headline (reproduced, see below).** P95 of the maximum cumulative deficit over 3,000 paths:
**$2.73M for loans vs $4.01M for grants**, conditional on the model's assumptions.

## Authorship and AI assistance

- **Aaditya** wrote the core cohort Monte Carlo (`solution/model.py`, 228/228 lines by git blame)
  and the first versions of `assumptions.py`, `charts.py`, `data_utils.py`, `run_analysis.py`
  (two commits on 2026-03-28). His files are included here as a port of the team repo; only the
  authors' own work should be quoted from them.
- Everything else was committed under **Farhan Sadeek**'s identity: `markov.py`, `deterministic.py`,
  `bootstrap_ci.py`, `sensitivity.py`, `voi.py`, `stress_tests.py`, `npv_analysis.py`,
  `breakeven.py`, `adaptive_funding.py`, `geographic.py`, `ml_models.py`, `agent_simulation.py`,
  `test_v4.py`, `analysis/`, the Astro slide deck (`presentation/`), and the LaTeX paper. He also
  extended the report text builders in `run_analysis.py` and added fields to the other files.
- AI assistance was used and own-writing cannot be separated from AI output. Three upstream commits
  carry `Co-Authored-By: Claude Opus 4.6` trailers. `58d9c0c` first introduced `markov.py`,
  `deterministic.py`, `bootstrap_ci.py`, `sensitivity.py`, `npv_analysis.py`, `paper.tex` and the
  first Astro deck; `a2bdda2` redesigned the deck and `f6a9ec2` added its build output. Git author
  is only the committer identity.
- `scripts/measure.py` and `results/` were added in this port (not in the upstream repo).
- No competition placing is recorded in the repo, and none is claimed.

## Layout

| Path | What |
|---|---|
| `solution/` | Python package. Core MC (`model.py`), Markov chain, deterministic benchmark, bootstrap, OAT tornado, "VOI", stress tests, NPV, break-even, adaptive funding, geographic scoring, PyTorch parcel nets + agent simulation, `run_analysis.py` driver. |
| `analysis/` | Two earlier exploratory scripts (not run here; hard-coded author paths). |
| `output/` | Checked-in `summary.json`, `annual_cashflows.csv`, 12 SVG charts, `paper.tex`, `executive_summary.tex`, `executive_report.md`. |
| `presentation/` | Astro/Tailwind slide deck source (no build output). |
| `scripts/measure.py` | Reproduction and diagnostics added for this port. |
| `results/` | `measurements.json`, `test_v4_output.txt`. |

**Not copied:** the challenge datasets (`dataset/*.csv`), the problem PDFs, trained `.pt`
checkpoints, build output, LaTeX artefacts, `output/final_report.md` (stale, see below). To run
the code, put the five challenge CSVs into `dataset/` as in the upstream repo.

## Running

Core model and all cross-checks need only the Python 3 standard library:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/measure.py results/   # ~2 min
PYTHONDONTWRITEBYTECODE=1 python3 -m solution.test_v4           # ~8 s
```

`python3 -m solution.run_analysis` regenerates everything but also needs torch, numpy and
scikit-learn, and overwrites `output/`. It was not run for this port.

## What the measurements show (`results/measurements.json`)

All numbers below come from running the commands above (Python 3.14.7) or from the checked-in
`output/summary.json`.

1. **Headline reproduced exactly.** 3,000 paths, seed 20260328, 21 s: loan mean $2,489,536.59,
   P95 $2,731,561.30 (P2.5-P97.5 $2.20M-$2.77M); grant P95 $4,005,262.08.
2. **The "deterministic check failed" flag is Jensen's inequality, not a bug.** Deterministic
   benchmark $2,471,551. The maximum of the MC *average* cumulative-deficit curve is $2,471,915
   (peak year 2034), within 0.015%. The MC mean of per-path maxima is $2,489,537 because the
   peak year varies by path (2031:1, 2032:47, 2033:618, 2034:1616, 2035:699, 2036:19).
   `det_within_bootstrap_ci = false` compares the deterministic value to a CI for the wrong statistic.
3. **The bootstrap CI is Monte Carlo sampling error only.** Mean CI $2,484,206-$2,494,821, P95 CI
   $2,716,604-$2,741,433 (width about $25k). One-at-a-time uptake-rate sweep (500 paths each): P95 goes
   from $2.21M at 0.16% to $3.20M at 0.25% (start rate; base 0.20% gives $2.73M). Parameter
   uncertainty dwarfs sampling error. Illustration only: pooling paths over an assumed uniform prior on the
   start rate in [0.16%, 0.25%] (the tornado's own range; the prior is my assumption, not from
   the repo) gives a mixture P95 of $3.03M vs $2.73M. The sweep at 0.30% and 0.40% is identical
   because the base uptake cap (0.30%) binds.
4. **Stress-test percentage bug.** `mean_increase_pct` divides by the base P95 instead of the
   base mean. Housing crash: reported +5.4%, corrected +15.7% (1,000 paths, seed 20260401).
   P95 increases are fine: +14.9% / +82.1% / +110.7% (crash / surge / perfect storm). The core
   model does not read the regime-matrix or contractor-capacity fields, so those edits in the
   scenarios have no effect, and the scenarios apply to all years, not only from 2029.
5. **"Held-out" v4 test does not show the model generalizes.** `solution/test_v4.py` gives
   6,057 eligible parcels in the 10,000-parcel synthetic v4 file, mean loan $9,993 vs $7,386,
   loan P95 $742,138; scaled by 34,269/6,057 that is $4.20M, 1.54x the $2.73M headline.
   The v4 file is described in the paper as organizer-supplied synthetic data.

Read from `output/summary.json`, not re-run: NPV net cost $2.05M at 3% (PV recovery 38.2%),
the VOI "dollar values" (they split the $24.8k bootstrap width by squared tornado swings, so
they are not a value-of-information calculation), the adaptive-funding result (0.0% saving,
top-up trigger always fires), and the ML/agent simulation (P95 $3.60M, mean $3.17M). The
ML/agent run was not reproduced (no scikit-learn, torch OpenMP conflict in the run environment).
The PyTorch nets fit hand-written heuristic formulas on synthetic parcels, so their
correlations (0.997 uptake, 0.973 sale) say nothing about predictive power; CostNet scored
R^2 = -8.55 on v4. `output/final_report.md` upstream is stale against `summary.json`.

## Caveats

- The 0.20%/yr uptake rate rests on one pilot window (62 active loans, April 2025 - March 2026).
- The paper's abstract says all methods "converge" on $2.73M; only the core MC produces that figure.
- Paper validation thresholds (1%) differ from the code's (10% gap, 5% CV).
