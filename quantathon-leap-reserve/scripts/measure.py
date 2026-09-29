"""Reproduction + diagnostics on top of the original solution/ package.

Run from the project root (needs dataset/ from the challenge, not included):
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/measure.py results/
Pure standard library. Does not modify solution/. Written for the blog port
(not part of the original hackathon repo).
"""
import json, sys, time, random
from dataclasses import replace
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from solution.assumptions import BASE_SCENARIO, YEARS
from solution.model import run_simulation, _percentile
from solution.deterministic import run_deterministic_model
from solution.stress_tests import STRESS_SCENARIOS

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results"); out.mkdir(exist_ok=True)
R = {}

# A. base reproduction
t = time.time()
base = run_simulation(BASE_SCENARIO, 3000, 20260328)
R["A_base"] = {"seconds": round(time.time() - t, 1), "loan": base["loan_summary"], "grant": base["grant_summary"]}

# B. Jensen gap: E[max_t X_t] vs max_t E[X_t]
paths = base["paths"]
cum = []
for p in paths:
    c, row = 0.0, []
    for y in YEARS:
        c += p.annual_outflows[y] - p.annual_inflows[y]; row.append(c)
    cum.append(row)
mean_curve = [fmean(r[i] for r in cum) for i in range(len(YEARS))]
peak_hist = {}
for r in cum:
    y = YEARS[max(range(len(YEARS)), key=lambda i: r[i])]
    peak_hist[y] = peak_hist.get(y, 0) + 1
det = run_deterministic_model(BASE_SCENARIO)
R["B_jensen"] = {
    "mc_mean_of_max": base["loan_summary"]["mean"],
    "mc_max_of_mean_curve": max(mean_curve),
    "mc_max_of_mean_curve_year": YEARS[mean_curve.index(max(mean_curve))],
    "deterministic": det["deterministic_funding_requirement"],
    "peak_year_histogram": dict(sorted(peak_hist.items())),
    "mean_cum_curve": {y: round(v, 2) for y, v in zip(YEARS, mean_curve)},
}

# C. bootstrap CI (MC sampling error only), same scheme as solution/bootstrap_ci.py (2000 resamples, seed 42)
reqs = base["loan_requirements"]
def boot(stat, n=2000, seed=42):
    rng = random.Random(seed); vals = []
    for _ in range(n):
        s = [reqs[rng.randrange(len(reqs))] for _ in reqs]; vals.append(stat(s))
    return [_percentile(vals, 0.025), _percentile(vals, 0.975)]
R["C_bootstrap"] = {"mean_ci95": boot(fmean), "p95_ci95": boot(lambda v: _percentile(v, 0.95))}

# D. uptake-start sweep (OAT), 500 paths each, seed 20260328; cap and growth left at base values
sweep = {}
for s in [0.0010, 0.0015, 0.0016, 0.0020, 0.0025, 0.0030, 0.0040]:
    sim = run_simulation(replace(BASE_SCENARIO, annual_trigger_start=s), 500, 20260328)
    sweep[f"{s*100:.2f}%"] = {"mean": sim["loan_summary"]["mean"], "p95": sim["loan_summary"]["p95"]}
R["D_uptake_sweep_500paths"] = sweep

# E. mixture over an ASSUMED uniform prior on uptake start in [0.16%, 0.25%] (the tornado's own range):
#    pool 300 paths at each of 10 grid points. Illustrative; the prior is not from the repo.
pool = []; grid = [0.0016 + i * (0.0009 / 9) for i in range(10)]
for s in grid:
    pool += run_simulation(replace(BASE_SCENARIO, annual_trigger_start=s), 300, 7)["loan_requirements"]
R["E_mixture_uniform_uptake_0.16_0.25pct"] = {
    "n": len(pool), "mean": fmean(pool), "p50": _percentile(pool, 0.5), "p95": _percentile(pool, 0.95),
    "p975": _percentile(pool, 0.975),
}
# EVPI for uptake under that prior, decision = reserve amount, loss = shortfall covered at cost 1 per $ and
# surplus held at cost 0; here we report the simple reserve-difference view: P95 by grid point vs mixture P95.
per = {}
for s in grid:
    per[f"{s*100:.3f}%"] = _percentile(run_simulation(replace(BASE_SCENARIO, annual_trigger_start=s), 300, 7)["loan_requirements"], 0.95)
R["E_p95_by_grid_point_300paths"] = per

# F. stress tests, with mean compared to base MEAN (original compares to base P95)
base1000 = run_simulation(BASE_SCENARIO, 1000, 20260401)["loan_summary"]
st = {}
for name, fn in STRESS_SCENARIOS.items():
    sim = run_simulation(fn(), 1000, 20260401)["loan_summary"]
    st[name] = {
        "mean": sim["mean"], "p95": sim["p95"],
        "p95_vs_base_p95_pct": round((sim["p95"] / R["A_base"]["loan"]["p95"] - 1) * 100, 1),
        "mean_vs_base_p95_pct__original_formula": round((sim["mean"] / R["A_base"]["loan"]["p95"] - 1) * 100, 1),
        "mean_vs_base_mean_pct__corrected": round((sim["mean"] / R["A_base"]["loan"]["mean"] - 1) * 100, 1),
    }
R["F_stress_1000paths"] = {"base_1000paths_seed20260401": base1000, "scenarios": st}

(out / "measurements.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps({k: v for k, v in R.items() if k != "B_jensen"}, indent=1, default=str)[:6000])
print(json.dumps({k: v for k, v in R["B_jensen"].items() if k != "mean_cum_curve"}, indent=1))
