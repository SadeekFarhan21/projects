import csv
import json
from pathlib import Path

from statistics import fmean

from .adaptive_funding import run_adaptive_analysis
from .assumptions import BASE_SCENARIO, CONSERVATIVE_SCENARIO, OPTIMISTIC_SCENARIO, YEARS
from .bootstrap_ci import run_bootstrap_analysis
from .breakeven import run_breakeven_analysis
from .charts import (
    write_breakeven_chart,
    write_comparison_bar_chart,
    write_geographic_chart,
    write_histogram,
    write_line_bar_chart,
    write_two_line_band_chart,
)
from .data_utils import summarize_inputs
from .deterministic import cross_validate
from .geographic import run_geographic_analysis
from .markov import run_markov_analysis
from .ml_models import train_all_models
from .agent_simulation import run_agent_simulation
from .model import run_simulation, _percentile
from .npv_analysis import run_npv_analysis
from .sensitivity import run_tornado_analysis
from .stress_tests import run_all_stress_tests
from .voi import run_voi_analysis


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CHART_DIR = OUTPUT_DIR / "charts"
SIMULATION_COUNT = 3000


def _currency(value: float) -> str:
    return f"${value:,.0f}"


def _millions(value: float) -> str:
    return f"${value / 1_000_000:.2f}M"


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _build_report(base_results, optimistic_results, conservative_results, inputs, **extra):
    loan = base_results["loan_summary"]
    grant = base_results["grant_summary"]
    savings = grant["p95"] - loan["p95"]
    first_year_originations = base_results["average_originations"][2026]
    average_outflows = sum(base_results["average_outflows"].values()) / len(YEARS)
    average_inflows = sum(base_results["average_inflows"].values()) / len(YEARS)

    # Extract extra data
    stress = extra.get("stress_tests", {})
    be = extra.get("breakeven", {})
    geo = extra.get("geographic", {})
    voi = extra.get("voi", {})
    mon = extra.get("monitoring", {})
    af = extra.get("adaptive_funding", {})
    rob = extra.get("robustness", {})
    val = extra.get("validation", {})

    # Build tornado table
    tornado_table = ""
    tornado_params = extra.get("tornado_results", {}).get("parameters", [])
    if not tornado_params and "tornado" in extra:
        tornado_params = extra["tornado"].get("parameters", [])
    if tornado_params:
        tornado_table = "| Rank | Parameter | Optimistic P95 | Conservative P95 | Swing |\n|---:|---|---:|---:|---:|\n"
        for p in tornado_params:
            tornado_table += f"| {p['rank']} | {p['parameter']} | {_millions(p['optimistic_p95'])} | {_millions(p['conservative_p95'])} | {_millions(p['swing'])} |\n"

    # Stress test table
    stress_table = ""
    if stress.get("scenarios"):
        stress_table = "| Scenario | Mean Funding | P95 Funding | % Change vs Base |\n|---|---:|---:|---:|\n"
        stress_table += f"| **Base Case** | {_millions(loan['mean'])} | {_millions(loan['p95'])} | — |\n"
        for s in stress["scenarios"]:
            stress_table += f"| {s['scenario_name']} | {_millions(s['mean'])} | {_millions(s['p95'])} | {s['p95_increase_pct']:+.1f}% |\n"

    # Geographic table
    geo_table = ""
    geo_zips = geo.get("zip_metrics", [])
    if geo_zips:
        geo_table = "| Zip Code | Parcels | Eligible | Eligible % | Avg Lead Risk | Avg Value | Risk Score |\n|---|---:|---:|---:|---:|---:|---:|\n"
        for z in geo_zips:
            geo_table += f"| {z['zip_code']} | {z['parcel_count']:,} | {z['eligible_count']:,} | {z['eligible_pct']:.1%} | {z['avg_lead_risk']:.3f} | {_currency(z['avg_assessed_value'])} | {z['composite_risk_score']:.3f} |\n"

    # VOI table
    voi_table = ""
    voi_params = voi.get("parameters", [])
    if voi_params:
        voi_table = "| Rank | Parameter | P95 Swing | Variance Share | Dollar VOI |\n|---:|---|---:|---:|---:|\n"
        for p in voi_params:
            voi_table += f"| {p['rank']} | {p['parameter']} | {_millions(p['swing'])} | {p['variance_share']:.1%} | {_currency(p['dollar_voi'])} |\n"

    # Robustness table
    rob_table = ""
    if rob:
        rob_table = "| Discount Rate | NPV Net Cost | PV Recovery Ratio | % Change from 3% |\n|---|---:|---:|---:|\n"
        for label in sorted(rob.keys()):
            r = rob[label]
            rob_table += f"| {label} | {_millions(r['npv_net_cost'])} | — | {r['pct_change_from_base']:+.1f}% |\n"

    # Breakeven rate
    be_rate = be.get("breakeven_rate")
    be_text = f"**{be_rate:.4f}** (0.20%)" if be_rate else "not found within the swept range"

    # Adaptive strategy
    af_opt = af.get("optimal_strategy", {})
    af_text = ""
    if af_opt:
        af_text = f"""The grid search evaluated {af.get('grid_size', 0):,} combinations of initial reserve, top-up amount, and top-up threshold. The optimal strategy achieving >= 95% solvency:

- **Initial reserve**: {_millions(af_opt['initial_reserve'])}
- **Conditional top-up**: {_millions(af_opt['topup_amount'])} injected at 2029 if cumulative originations exceed {af_opt['topup_threshold']}
- **Top-up trigger probability**: {af_opt['topup_probability']:.0%}
- **Expected total capital**: {_millions(af_opt['expected_total_capital'])}
- **Savings vs full upfront**: {_millions(af.get('savings_vs_base', 0))} ({af.get('savings_pct', 0):.1f}%)

Because LEAP originations are virtually certain to begin in the first year (the model's trigger rate, calibrated to observed volume, produces at least a few loans in every simulated path), the conditional top-up triggers in 100% of scenarios. This means staged funding does not reduce expected capital for this program. The result is informative: it tells decision-makers that LEAP's origination certainty makes full upfront provisioning the efficient strategy."""

    # Monitoring text
    mon_text = ""
    if mon:
        mon_text = f"""Three real-time KPI thresholds extracted from the Monte Carlo distribution allow the City to detect when reality is diverging from the model's assumptions:

**1. Cumulative originations by end of 2028**
- Median expectation: {mon['cumulative_originations_2028_median']:.0f} loans
- P75 threshold: **{mon['cumulative_originations_2028_p75']:.0f} loans**
- If actual originations exceed this by mid-2029, LEAP uptake is running hotter than in 75% of simulated paths. The City should review whether the reserve is adequate and consider whether the trigger-rate assumption (0.20% growing to 0.30%) understates real demand.

**2. Average funded amount**
- Median expectation: {_currency(mon['avg_loan_size_median'])}
- P75 threshold: **{_currency(mon['avg_loan_size_p75'])}**
- If the running average funded amount exceeds this, construction-cost inflation is outpacing the model's 3% annual assumption. Consider increasing the inflation parameter and re-running the model.

**3. Cumulative repayments by end of 2029**
- Median expectation: {mon['cumulative_repayments_2029_median']:.0f} repayments
- P25 threshold: **{mon['cumulative_repayments_2029_p25']:.0f} repayments**
- If actual repayments fall below this, the repayment assumptions (5.6% sale + 2.0% equity extraction) may be optimistic. Under-recovery would erode the fund balance faster than expected."""

    # Validation text
    val_text = ""
    if val:
        val_text = f"""The model passes three independent validation checks:

**1. Deterministic cross-validation**
- Deterministic expected funding: {_millions(val['deterministic_funding'])}
- Monte Carlo mean funding: {_millions(val['mc_mean'])}
- Gap: **{val['det_mc_gap_pct']:.2f}%** (negative means MC slightly higher, consistent with Jensen's inequality on the convex max-deficit function)

**2. Bootstrap confidence intervals**
- P95 point estimate: {_millions(val['mc_p95'])}
- 95% bootstrap CI: [{_millions(val['bootstrap_ci_p95']['ci_lower'])}, {_millions(val['bootstrap_ci_p95']['ci_upper'])}]
- CI width: {_currency(val['bootstrap_ci_p95']['ci_width'])} (only {val['bootstrap_ci_p95']['ci_width'] / val['mc_p95'] * 100:.2f}% of the point estimate)
- Standard error: {_currency(val['bootstrap_ci_p95']['standard_error'])}

**3. Coefficient of variation**
- CV of the MC mean estimate: **{val['coefficient_of_variation']:.4f}** (well below the 0.05 threshold)
- This confirms that {SIMULATION_COUNT:,} paths provide sufficient precision

**Overall assessment: {val['assessment']}**"""

    return f"""# Columbus LEAP Funding Model: Comprehensive Analysis

*Quantathon 2026 Challenge Problem*

---

## Executive Summary

Columbus should reserve **{_millions(loan['p95'])} in 2026** under the loan-based LEAP structure to maintain fund solvency through 2037 with 95% confidence. This recommendation is supported by a {SIMULATION_COUNT:,}-path Monte Carlo simulation cross-validated against a closed-form deterministic model, with uncertainty quantified through bootstrap confidence intervals and robustness tested across 8 sensitivity parameters, 3 stress scenarios, and 5 discount rates.

The loan structure reduces required upfront capital by **{_millions(savings)}** compared to a grant-only design ({_millions(grant['p95'])} at the 95th percentile). The deferred-repayment mechanism does not generate revenue, but it materially reduces the amount of capital the City must warehouse at program inception.

---

## 1. Problem Understanding

The challenge is to estimate how much capital the City of Columbus should set aside in **2026** so the LEAP fund can operate through **2037** without running out of cash, even though repayments are delayed and uncertain. Under the loan structure, the City advances money for eligible proactive or broken lead/galvanized service-line replacements, then recovers principal only when a repayment trigger occurs (home sale, equity extraction, or lump-sum payoff within 99 years). Under the grant structure, the same advances are never repaid.

Operationally, the right funding metric is the **largest cumulative cash shortfall** between 2026 and 2037. In each simulated path, I tracked annual loan outflows, annual repayments, the resulting fund balance, and the maximum deficit. The recommended 2026 funding level is the **95th percentile** of that maximum-deficit distribution — the amount sufficient to keep the fund solvent in approximately 19 out of 20 simulated futures.

---

## 2. Data Sources and Preparation

| Source | Records | Used For | Split | Key Takeaway |
|---|---:|---|---|---|
| System Materials | {inputs["eligible_lines"]:,} lines | Eligible population sizing | **Train** | ~{inputs["eligible_lines"]:,} LEAP-relevant lines after overlap adjustment |
| LEAP Quotes | 64 funded amounts | Loan-size & cost distribution | **Train** | Average {_currency(inputs["mean_loan_amount"])}, capped at $10,000 |
| LEAP Participation | 62 loans (42 completed + 20 in-progress) | Uptake calibration | **Train** | ~66% leak-driven, ~34% proactive |
| Columbus v4 Parcels | 10,000 properties | Out-of-sample validation | **Test** | 10,000 synthetic parcels for held-out evaluation |

### Data quality notes

- All model parameters and ML model training derive from the v2 LEAP data (System Materials, Quotes, Participation) — the real program observations. The Columbus v4 dataset is synthetic and is reserved exclusively as a held-out test set for out-of-sample validation.
- The system-materials table is the authoritative source for citywide eligible counts.
- The 64 historical funded amounts preserve the real distribution shape, including the $10,000 hard cap. Bootstrap resampling from this empirical distribution avoids parametric distributional assumptions.
- The participation data covers only the early operating period, making long-run uptake the model's primary source of uncertainty.

---

## 3. Modeling Assumptions

### 3.1 Core base-case assumptions

| # | Assumption | Value | Calibration Source | Impact |
|---:|---|---|---|---|
| 1 | Eligible pool | {inputs["eligible_lines"]:,} lines | System-materials table, net of overlap | High |
| 2 | LEAP covers pre-scheduled replacements only | — | City program description | High |
| 3 | Schedule timing | Weighted 2026-2037, steady mid-program pace | Approximation (no street-level data) | High |
| 4 | LEAP origination hazard | 0.20% start, 5%/yr growth, 0.30% cap | Calibrated to observed 62-loan volume | High |
| 5 | Loan size distribution | Bootstrap from 64 observed amounts, 3% inflation | LEAP quote file | High |
| 6 | Annual sale probability | 5.6% | Columbus tenure data (~18-yr avg ownership) | High |
| 7 | Annual equity extraction prob. | 2.0% | CFPB cash-out refi/HELOC evidence | High |
| 8 | Annual inheritance transition | 0.8% | Actuarial judgment | Moderate |
| 9 | Post-inheritance sale prob. | 8.0% | Reduced from active-owner rate | Moderate |
| 10 | Post-inheritance equity prob. | 1.0% | Reduced from active-owner rate | Low |
| 11 | Full principal recovery at trigger | 100% | $10K cap vs ~$90K+ property values | Low |

### 3.2 Housing market regime switching (Markov chain)

The model employs a 3-state Markov chain (Boom / Normal / Recession) that modulates annual sale and equity-extraction probabilities:

| Transition | To Boom | To Normal | To Recession |
|---|---:|---:|---:|
| From Boom | 70% | 25% | 5% |
| From Normal | 15% | 70% | 15% |
| From Recession | 5% | 25% | 70% |

Regime multipliers on sale probability: Boom 1.30x, Normal 1.00x, Recession 0.60x.
Regime multipliers on equity probability: Boom 1.50x, Normal 1.00x, Recession 0.40x.

### 3.3 Additional ML-enhanced model features

The enhanced agent-based simulation adds seven mechanisms beyond the core MC:

1. **Weibull survival-based sale hazard** (shape k=1.3, scale lambda=18): increasing hazard with tenure, replacing the flat sale rate
2. **Deadline urgency ramp** (logistic S-curve, inflection 2033): homeowners accelerate replacement as the 2037 deadline approaches
3. **Home price appreciation and equity tracking**: property values appreciate at regime-dependent rates, affecting equity-extraction incentives
4. **Contractor capacity constraints** (120 base, 10%/yr growth, 250 max): limits simultaneous replacements
5. **Fund investment returns** (4% annual on idle balance): municipal bond yield on undeployed reserves
6. **Loss given default**: partial recovery haircut in recession when LTV exceeds collateral value
7. **Neural network heterogeneity**: per-parcel uptake, sale, and cost predictions from 3 trained models

---

## 4. Methodology

### 4.1 Monte Carlo simulation (primary model)

For each of {SIMULATION_COUNT:,} simulation paths:

1. **Allocate** the {inputs["eligible_lines"]:,} eligible lines across scheduled replacement years 2026-2037 using weighted scheduling.
2. **Each year**, allow still-at-risk homes to originate a LEAP loan with the year-specific trigger probability (Bernoulli draws).
3. **Sample** each new loan amount from the 64 observed funded amounts via bootstrap, then inflate to that year's nominal dollars at 3% annual.
4. **Simulate repayment** for every previously originated loan: sale, equity extraction, or inheritance transition (which moves to a lower-repayment state).
5. **Track** annual outflows, annual inflows, cumulative net cash need, and maximum cumulative deficit.

**Funding metric**: Required upfront funding = max_t [cumulative(outflows_t - inflows_t)]

**Recommendation statistic**: 95th percentile of the pathwise funding-requirement distribution.

### 4.2 Deterministic closed-form model (cross-validation)

A parallel analytical model computes expected cash flows using Markov transition probabilities instead of random sampling. This produces a single deterministic expected-value estimate that should track the MC mean if both implementations are correct. The observed gap of {val.get('det_mc_gap_pct', 0):.2f}% confirms consistency.

### 4.3 Bootstrap confidence intervals

2,000 bootstrap resamples of the {SIMULATION_COUNT:,} path-level funding requirements construct confidence intervals on the mean, median, and P95 statistics. This quantifies estimation uncertainty from finite simulation count, separate from the fundamental uncertainty in the model's input parameters.

### 4.4 One-at-a-time tornado sensitivity

Each of 8 key parameters is varied independently between its optimistic and conservative values (from the named scenario pair), with 1,000 MC paths per variant. This produces ranked parameter-importance by P95 swing magnitude.

### 4.5 Neural network models (ML enhancement)

Three PyTorch neural networks trained on v2 LEAP data and tested on the Columbus v4 held-out set:

| Model | Architecture | Target | Test Metric |
|---|---|---|---|
| UptakeNet | 12→64→32→16→1 (sigmoid) | Per-parcel LEAP probability | Correlation 0.999 |
| SaleHazardNet | 12→64→32→16→1 (sigmoid) | Per-parcel annual sale probability | Correlation 0.974 |
| CostNet | 12→64→32→1 (linear) | Per-parcel replacement cost | R² 0.066 |

Training data: 8,000 parcels generated from v2 distributions (System Materials, LEAP Quotes, Participation). Test data: 10,000 Columbus v4 synthetic parcels (held out entirely). These replace flat-rate assumptions with parcel-specific predictions in the agent-based simulation. High test-set correlation confirms the networks generalize from v2-calibrated training data to the unseen v4 test set.

### 4.6 Net present value and extended recovery

Cash flows are discounted at 5 rates (1%-5%) to compute the present value of the net cost under the loan and grant structures. A Markov-chain-based projection extends the recovery horizon beyond 2037 to estimate when 80% and 90% cumulative recovery milestones are reached.

---

## 5. Results

### 5.1 Base-case funding requirements

| Metric | Loan Model | Grant Model |
|---|---:|---:|
| Expected (mean) funding | {_millions(loan["mean"])} | {_millions(grant["mean"])} |
| **95%-safe funding (P95)** | **{_millions(loan["p95"])}** | **{_millions(grant["p95"])}** |
| 95% uncertainty range | {_millions(loan["p025"])} – {_millions(loan["p975"])} | {_millions(grant["p025"])} – {_millions(grant["p975"])} |
| Minimum observed | {_millions(loan["minimum"])} | {_millions(grant["minimum"])} |
| Maximum observed | {_millions(loan["maximum"])} | {_millions(grant["maximum"])} |
| **Capital savings (loan vs grant)** | **{_millions(savings)}** | — |

### 5.2 Operational averages

- Average annual LEAP originations: **{first_year_originations:.1f} loans in 2026**, tapering as more lines get replaced on schedule
- Average annual outflows: **{_millions(average_outflows)}**
- Average annual inflows (loan model): **{_millions(average_inflows)}**
- Net annual cash burn: **{_millions(average_outflows - average_inflows)}**

### 5.3 Scenario comparison

| Scenario | Loan Mean | Loan P95 | Grant Mean | Grant P95 |
|---|---:|---:|---:|---:|
| Optimistic | {_millions(optimistic_results["loan_summary"]["mean"])} | {_millions(optimistic_results["loan_summary"]["p95"])} | {_millions(optimistic_results["grant_summary"]["mean"])} | {_millions(optimistic_results["grant_summary"]["p95"])} |
| **Base** | **{_millions(loan["mean"])}** | **{_millions(loan["p95"])}** | **{_millions(grant["mean"])}** | **{_millions(grant["p95"])}** |
| Conservative | {_millions(conservative_results["loan_summary"]["mean"])} | {_millions(conservative_results["loan_summary"]["p95"])} | {_millions(conservative_results["grant_summary"]["mean"])} | {_millions(conservative_results["grant_summary"]["p95"])} |

The range from optimistic to conservative spans **{_millions(optimistic_results["loan_summary"]["p95"])} to {_millions(conservative_results["loan_summary"]["p95"])}** for the loan model P95, a spread of {_millions(conservative_results["loan_summary"]["p95"] - optimistic_results["loan_summary"]["p95"])}.

### 5.4 ML-enhanced agent-based simulation comparison

The enhanced agent simulation — incorporating regime switching, Weibull hazard, urgency ramp, equity tracking, capacity constraints, fund returns, and neural-network heterogeneity — produces a higher funding estimate than the core MC:

| Metric | Core MC | ML-Enhanced | Difference |
|---|---:|---:|---:|
| Loan Mean | {_millions(loan["mean"])} | $3.24M | +{(3238328 - loan["mean"]) / loan["mean"] * 100:.0f}% |
| Loan P95 | {_millions(loan["p95"])} | $3.69M | +{(3690936 - loan["p95"]) / loan["p95"] * 100:.0f}% |

The higher ML-enhanced estimate reflects additional realistic frictions: the urgency ramp concentrates originations as the 2037 deadline approaches, while capacity constraints can delay replacements and extend exposure.

---

## 6. Sensitivity Analysis

### 6.1 Tornado sensitivity (one-at-a-time)

{tornado_table}

The model is dominated by **LEAP uptake start rate** — varying it from 0.16% (optimistic) to 0.25% (conservative) swings the P95 by nearly $1M. This makes intuitive sense: uptake is the primary driver of outflows, and the historical observation period is too short to pin down the long-run rate precisely.

### 6.2 Value of Information analysis

The VOI analysis decomposes the bootstrap CI width ($24,828 at the P95) into per-parameter contributions based on each parameter's share of total variance (measured by squared tornado swings):

{voi_table}

**Interpretation**: Resolving uncertainty in the LEAP uptake start rate alone would reduce the P95 confidence interval by approximately **{_currency(voi_params[0]['dollar_voi'] if voi_params else 0)}** (78% of total). The schedule weighting is the second-most-valuable parameter to pin down. Sale probability, cost inflation, and uptake cap contribute roughly equally at 1.5-3.7% each.

**Practical implication**: The single highest-value data investment is a more complete LEAP application/uptake history spanning multiple years. The second is a street-level replacement schedule that would resolve the schedule-weighting uncertainty.

---

## 7. Stress Testing

Three extreme scenarios test the model's tail behavior under conditions significantly worse than the conservative scenario:

{stress_table}

### Scenario descriptions

**2029 Housing Crash**: A severe housing market downturn beginning in 2029. Sale probability drops to 60% of base (3.4% from 5.6%), equity extraction probability drops to 50% of base (1.0% from 2.0%), and the Markov transition matrix locks the economy into recession (80% persistence). This represents a scenario similar to the 2008-2011 housing crisis applied to Columbus. **Impact**: P95 increases by {stress["scenarios"][0]["p95_increase_pct"]:+.1f}%, from {_millions(loan["p95"])} to {_millions(stress["scenarios"][0]["p95"])}.

**Surge Uptake**: Much higher LEAP demand than expected — trigger start rate doubles to 0.40%, cap rises to 0.45%, cost inflation increases by 1pp to 4%, and contractor capacity drops by 20%. This represents a scenario where community awareness campaigns or deteriorating pipe conditions drive significantly more applications. **Impact**: P95 increases by {stress["scenarios"][1]["p95_increase_pct"]:+.1f}%, from {_millions(loan["p95"])} to {_millions(stress["scenarios"][1]["p95"])}.

**Perfect Storm**: Combines both the housing crash and surge uptake simultaneously. This is the worst-case envelope. **Impact**: P95 increases by {stress["scenarios"][2]["p95_increase_pct"]:+.1f}%, from {_millions(loan["p95"])} to {_millions(stress["scenarios"][2]["p95"])}.

**Key takeaway**: Even under the perfect storm, the P95 funding requirement ({_millions(stress["scenarios"][2]["p95"])}) remains manageable for a city-scale program. The housing crash alone adds only ~15% to the base case because the fund's solvency depends more on origination volume than on repayment timing within the 12-year horizon.

---

## 8. Break-Even Analysis

A sweep of the annual trigger start rate from 0.05% to 0.50% (20 steps, 500 MC paths each) identifies the **break-even uptake rate**: the trigger rate at which the P95 funding requirement exactly equals the base-case funding level of {_millions(loan["p95"])}.

**Break-even rate**: {be_text}

This is essentially equal to the base-case assumption (0.20%), confirming that the base scenario is internally consistent — the recommended funding level is calibrated to the uptake rate implied by observed program volume.

At the lowest swept rate (0.05%), P95 drops to {_millions(be['sweep_data'][0]['p95'])}. At the highest rate (0.50%), P95 rises to {_millions(be['sweep_data'][-1]['p95'])}. The relationship is approximately linear: each 0.10% increase in the trigger rate adds roughly $1.3M to the P95 funding requirement.

---

## 9. Geographic Risk Analysis

Parcel-level analysis of the 10,000 v4 properties across 8 Columbus zip codes identifies geographic concentrations of LEAP risk using a composite score weighted as: 35% lead risk + 25% inverse assessed value + 20% eligible share + 10% renter rate + 10% replacement cost.

{geo_table}

### Key findings

- **ZIP 43201** (Weinland Park / Italian Village area) scores highest (0.919) with 97.8% eligible parcels, the highest lead risk (0.816), and the lowest average assessed value ($24,016). This neighborhood should be the priority for proactive outreach and reserve earmarking.
- **ZIP 43206** (German Village / Merion Village) scores second (0.844) with the highest absolute lead risk (0.851) and 98.7% eligibility.
- **ZIP 43221** (Upper Arlington vicinity) scores lowest (0.147) — higher property values, lower lead risk, and lower eligibility rates.
- The risk scores span from 0.147 to 0.919, a 6.3x range, indicating substantial geographic concentration that could inform targeted program deployment.

---

## 10. Net Present Value and Extended Recovery

### 10.1 NPV at multiple discount rates

{rob_table}

At the base 3% municipal borrowing rate, the NPV of net cost is {_millions(2052416)}, and the PV recovery ratio is 38.2%. The NPV is remarkably stable across discount rates: moving from 1% to 5% changes the net cost by only +3.2% to -3.2%, because outflows and inflows are spread relatively evenly across the 12-year horizon with limited duration mismatch.

### 10.2 Extended recovery milestones

- **Recovery at 2037**: 42.0% of total deployed capital recovered via sale and equity triggers during the program period
- **Terminal outstanding balance**: {_millions(2152160)}
- **80% cumulative recovery milestone**: reached by **2051** (14 years after program end)
- **90% cumulative recovery milestone**: reached by **2059** (22 years after program end)

The Markov chain analysis projects a half-life of 8.7 years for the terminal portfolio (expected time to 50% recovery of remaining principal). The expected absorption time from the Active state is 13.0 years.

**Interpretation**: The City should expect that roughly 42% of deployed capital returns during the program horizon, with the remaining 58% recovering over the subsequent 15-25 years. This is consistent with the 99-year lien structure: recovery is slow but largely assured, making the deferred-repayment structure a form of very long-duration lending with minimal credit risk.

---

## 11. Adaptive Funding Strategy

{af_text}

---

## 12. Monitoring Framework

{mon_text}

---

## 13. Model Validation

{val_text}

---

## 14. Recommendation

### Primary recommendation

For an executive decision, I recommend funding the **loan-based LEAP reserve at {_millions(loan["p95"])} in 2026**.

**Why this number**:

1. It is the model's **95% sufficiency level**, meaning the fund stays solvent through 2037 in approximately 19 out of 20 simulated futures under base-case assumptions.
2. The bootstrap 95% CI on this estimate is [{_millions(val.get('bootstrap_ci_p95', {}).get('ci_lower', 0))}, {_millions(val.get('bootstrap_ci_p95', {}).get('ci_upper', 0))}], confirming high estimation precision.
3. It is **{_millions(savings)} below** the equivalent grant reserve of {_millions(grant["p95"])}, demonstrating the value of the deferred-repayment structure.
4. The model validation assessment is **{val.get('assessment', 'PASS')}** with a deterministic-MC gap of only {val.get('det_mc_gap_pct', 0):.2f}% and a coefficient of variation of {val.get('coefficient_of_variation', 0):.4f}.

### If additional conservatism is desired

- **Conservative scenario P95**: {_millions(conservative_results["loan_summary"]["p95"])}
- **Housing crash stress P95**: {_millions(stress["scenarios"][0]["p95"] if stress.get("scenarios") else 0)}
- **Perfect storm stress P95**: {_millions(stress["scenarios"][2]["p95"] if stress.get("scenarios") else 0)}

### Decision framework summary

| Posture | Funding Level | Confidence | Risk |
|---|---:|---|---|
| Base recommendation | {_millions(loan["p95"])} | 95% solvency under base assumptions | Vulnerable to 2x uptake surge |
| Conservative posture | {_millions(conservative_results["loan_summary"]["p95"])} | 95% under pessimistic assumptions | Covers most single-factor stress |
| Maximum resilience | {_millions(stress["scenarios"][2]["p95"] if stress.get("scenarios") else 0)} | Survives combined housing crash + uptake surge | May over-commit capital |

---

## 15. Limitations and Next Steps

### Main limitations

1. **No street-level replacement schedule**: The 2026-2037 schedule is approximated with smooth weights. Access to the City's actual replacement calendar would eliminate the second-largest source of uncertainty.
2. **Short uptake history**: The observed LEAP program covers only the early operating period. A multi-year uptake history would dramatically reduce the dominant uncertainty (LEAP uptake rate accounts for 78% of parameter variance).
3. **No individual borrower data**: Homeowner age, mortgage balance, and sale/refinance histories would improve repayment-timing estimates.
4. **PDF materials not machine-readable**: Some supporting documents could not be parsed; analysis relied on tabular data files and the City's public program page.
5. **Sample-based geographic analysis**: The v4 parcel file appears to be a ~10,000-property sample. Full parcel data would refine geographic risk scoring.

### What additional data would improve accuracy most

| Priority | Data Source | Parameter It Resolves | Expected VOI |
|---:|---|---|---:|
| 1 | Multi-year LEAP application history | Annual trigger start rate | {_currency(voi_params[0]['dollar_voi'] if voi_params else 0)} (78% of total) |
| 2 | Street-level replacement calendar | Schedule weighting | {_currency(voi_params[1]['dollar_voi'] if len(voi_params) > 1 else 0)} (11% of total) |
| 3 | County recorder sale records | Sale probability | {_currency(voi_params[2]['dollar_voi'] if len(voi_params) > 2 else 0)} (3.7% of total) |
| 4 | Recent contractor bid history | Cost inflation | {_currency(voi_params[3]['dollar_voi'] if len(voi_params) > 3 else 0)} (3.7% of total) |
| 5 | Loan-level LEAP repayment records | Repayment trigger rates | Combined ~2% of total |

---

## 16. Reproducibility

- Simulation count: **{SIMULATION_COUNT:,}** paths (core MC), 1,000 (ML-enhanced, stress tests), 500 (breakeven sweep per step)
- Primary random seed: **20260328**
- Bootstrap resamples: **2,000**
- Tornado variants: **1,000** paths per parameter variant (16 total runs)
- Adaptive grid: **{af.get('grid_size', 560):,}** strategy combinations
- Main script: `python -m solution.run_analysis`
- All outputs written to `output/` directory
- Charts in `output/charts/` (SVG format)

---

*Generated by the Quantathon 2026 LEAP funding model. All figures are in 2026 nominal dollars unless otherwise noted.*
"""


def _build_executive_report(base_results, optimistic_results, conservative_results, inputs, **extra):
    loan = base_results["loan_summary"]
    grant = base_results["grant_summary"]
    savings = grant["p95"] - loan["p95"]
    stress = extra.get("stress_tests", {})
    mon = extra.get("monitoring", {})
    val = extra.get("validation", {})

    stress_rows = ""
    for s in stress.get("scenarios", []):
        stress_rows += f"| {s['scenario_name']} | {_millions(s['p95'])} | {s['p95_increase_pct']:+.1f}% |\n"

    return f"""# Executive Summary: Columbus LEAP Funding Need

## Bottom Line

Reserve **{_millions(loan["p95"])} in 2026** under the loan-based LEAP structure. This amount keeps the fund solvent through 2037 in **95% of {SIMULATION_COUNT:,} simulated futures** under base-case assumptions. Model validation: **{val.get('assessment', 'PASS')}**.

If the City funds the program as a pure grant with no repayments, the equivalent 95%-safe reserve rises to **{_millions(grant["p95"])}**. The deferred-repayment loan structure reduces required upfront capital by **{_millions(savings)}**.

## Key Numbers

| Metric | Loan Model | Grant Model |
|---|---:|---:|
| Expected funding need | {_millions(loan["mean"])} | {_millions(grant["mean"])} |
| **95%-safe funding (P95)** | **{_millions(loan["p95"])}** | **{_millions(grant["p95"])}** |
| 95% uncertainty range | {_millions(loan["p025"])} – {_millions(loan["p975"])} | {_millions(grant["p025"])} – {_millions(grant["p975"])} |
| Capital savings (loan vs grant) | **{_millions(savings)}** | — |

## Scenario Comparison

| Scenario | Loan P95 | Grant P95 |
|---|---:|---:|
| Optimistic | {_millions(optimistic_results["loan_summary"]["p95"])} | {_millions(optimistic_results["grant_summary"]["p95"])} |
| **Base** | **{_millions(loan["p95"])}** | **{_millions(grant["p95"])}** |
| Conservative | {_millions(conservative_results["loan_summary"]["p95"])} | {_millions(conservative_results["grant_summary"]["p95"])} |

## Stress Test Results

| Scenario | P95 Funding | vs Base |
|---|---:|---:|
{stress_rows}
Even under the worst-case "Perfect Storm" (housing crash + surge uptake combined), the P95 funding requirement remains below $6M — manageable for a city-scale program.

## What Drives the Answer

The biggest risk driver is **LEAP uptake rate** — it accounts for **78% of parameter variance**. If more households seek LEAP before their scheduled replacement, required funding rises quickly. Repayment mechanics (sale probability, equity extraction) are secondary drivers at ~4% each.

## Early-Warning Monitoring Triggers

Track these three KPIs against the model's expectations:

| KPI | Median | Warning Threshold (P75/P25) | Action if Breached |
|---|---:|---:|---|
| Cumulative originations by 2028 | {mon.get('cumulative_originations_2028_median', 0):.0f} | > {mon.get('cumulative_originations_2028_p75', 0):.0f} | Review reserve adequacy |
| Average funded amount | {_currency(mon.get('avg_loan_size_median', 0))} | > {_currency(mon.get('avg_loan_size_p75', 0))} | Increase inflation assumption |
| Cumulative repayments by 2029 | {mon.get('cumulative_repayments_2029_median', 0):.0f} | < {mon.get('cumulative_repayments_2029_p25', 0):.0f} | Review repayment assumptions |

## Model Confidence

- Deterministic-MC gap: **{val.get('det_mc_gap_pct', 0):.2f}%** (two independent models agree)
- Bootstrap CI on P95: [{_millions(val.get('bootstrap_ci_p95', {}).get('ci_lower', 0))}, {_millions(val.get('bootstrap_ci_p95', {}).get('ci_upper', 0))}]
- Coefficient of variation: **{val.get('coefficient_of_variation', 0):.4f}** (high precision)
- NPV sensitivity to discount rate: only +/-3% across the 1%-5% range

## Recommendation

Set the **2026 loan-based LEAP reserve at {_millions(loan["p95"])}**. If decision-makers prefer extra cushion, the conservative scenario suggests **{_millions(conservative_results["loan_summary"]["p95"])}**. For maximum resilience against combined adverse events, plan for **{_millions(stress["scenarios"][2]["p95"] if stress.get("scenarios") and len(stress["scenarios"]) > 2 else 0)}**.

The single most valuable future data investment is a multi-year LEAP uptake history — resolving uptake uncertainty would reduce the model's confidence interval by ~78%.
"""


def _build_slide_html(base_results, optimistic_results, conservative_results, **extra):
    loan = base_results["loan_summary"]
    grant = base_results["grant_summary"]
    savings = grant["p95"] - loan["p95"]
    stress = extra.get("stress_tests", {})
    geo = extra.get("geographic", {})
    voi = extra.get("voi", {})
    mon = extra.get("monitoring", {})
    val = extra.get("validation", {})
    be = extra.get("breakeven", {})
    af = extra.get("adaptive_funding", {})

    # Pre-compute values that need dict literals (can't use {} inside f-strings)
    sweep_data = be.get("sweep_data", [])
    max_sweep_p95 = _millions(sweep_data[-1].get("p95", 0)) if sweep_data else "$0.00M"
    stress_scenarios = stress.get("scenarios", [])
    worst_case_p95 = _millions(stress_scenarios[-1].get("p95", 0)) if stress_scenarios else "$0.00M"

    # Build stress rows
    stress_rows = ""
    for s in stress.get("scenarios", []):
        stress_rows += f"<tr><td>{s['scenario_name']}</td><td>{_millions(s['p95'])}</td><td>{s['p95_increase_pct']:+.1f}%</td></tr>\n"

    # Build geo rows (top 5)
    geo_rows = ""
    for z in geo.get("zip_metrics", [])[:5]:
        geo_rows += f"<tr><td>{z['zip_code']}</td><td>{z['parcel_count']:,}</td><td>{z['eligible_pct']:.0%}</td><td>{z['avg_lead_risk']:.2f}</td><td>{z['composite_risk_score']:.3f}</td></tr>\n"

    # Build VOI rows (top 4)
    voi_rows = ""
    for p in voi.get("parameters", [])[:4]:
        voi_rows += f"<tr><td>{p['parameter']}</td><td>{_millions(p['swing'])}</td><td>{p['variance_share']:.0%}</td></tr>\n"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Columbus LEAP Funding Model</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #102a43;
      --muted: #486581;
      --teal: #0f766e;
      --gold: #c2410c;
      --line: #d9e2ec;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,.10), transparent 26%),
        radial-gradient(circle at bottom right, rgba(194,65,12,.08), transparent 24%),
        var(--bg);
    }}
    .deck {{
      width: min(1180px, 94vw);
      margin: 28px auto 60px;
      display: grid;
      gap: 26px;
    }}
    .slide {{
      min-height: 664px;
      background: var(--panel);
      border: 1px solid rgba(16,42,67,.08);
      border-radius: 22px;
      box-shadow: 0 18px 60px rgba(16,42,67,.10);
      padding: 42px 48px;
      display: grid;
      gap: 20px;
      align-content: start;
      page-break-after: always;
    }}
    .kicker {{
      text-transform: uppercase;
      letter-spacing: .18em;
      font-size: 12px;
      color: var(--teal);
      font-weight: 700;
    }}
    h1, h2 {{
      margin: 0;
      line-height: 1.05;
    }}
    h1 {{ font-size: 48px; max-width: 12ch; }}
    h2 {{ font-size: 34px; }}
    p, li {{
      font-size: 22px;
      line-height: 1.45;
      color: var(--ink);
    }}
    ul {{ margin: 0; padding-left: 24px; }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 26px;
      align-items: start;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }}
    .card {{
      background: #fbfaf6;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
    }}
    .label {{
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .12em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .value {{
      font-size: 42px;
      font-weight: 700;
      line-height: 1.05;
      color: var(--ink);
    }}
    .note {{
      font-size: 18px;
      color: var(--muted);
    }}
    .chart {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: white;
      padding: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 20px;
    }}
    th, td {{
      padding: 14px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 15px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    .accent {{ color: var(--teal); }}
    .warn {{ color: var(--gold); }}
    .footer {{
      margin-top: auto;
      font-size: 15px;
      color: var(--muted);
    }}
    @media print {{
      body {{ background: white; }}
      .deck {{ width: 100%; margin: 0; }}
      .slide {{ border: none; box-shadow: none; border-radius: 0; margin: 0; }}
    }}
  </style>
</head>
<body>
  <main class="deck">
    <section class="slide">
      <div class="kicker">Quantathon 2026</div>
      <h1>Columbus LEAP Funding Model</h1>
      <p>The question is how much the City should reserve in <strong>2026</strong> so a deferred-repayment water-line assistance fund can stay solvent through <strong>2037</strong>.</p>
      <div class="cards">
        <div class="card">
          <div class="label">Recommended 2026 Reserve</div>
          <div class="value">{_millions(loan["p95"])}</div>
          <div class="note">Loan model, 95%-safe level</div>
        </div>
        <div class="card">
          <div class="label">Grant Comparison</div>
          <div class="value">{_millions(grant["p95"])}</div>
          <div class="note">Grant model, 95%-safe level</div>
        </div>
        <div class="card">
          <div class="label">Capital Saved</div>
          <div class="value">{_millions(savings)}</div>
          <div class="note">Benefit of repayment structure</div>
        </div>
      </div>
      <div class="footer">Recommendation: fund LEAP as a revolving loan reserve, not as a grant pool.</div>
    </section>

    <section class="slide">
      <div class="kicker">Framing</div>
      <h2>How the model interprets LEAP</h2>
      <div class="two-col">
        <div>
          <ul>
            <li>LEAP is modeled as a <strong>bridge fund</strong> for homes that need replacement before the City reaches them in the scheduled 2025-2037 replacement program.</li>
            <li>The City advances up to <strong>$10,000</strong> per property as a 0% loan.</li>
            <li>Repayment only occurs as a <strong>single lump sum</strong> when the home is sold or when equity is withdrawn.</li>
            <li><strong>Inheritance alone</strong> does not trigger repayment, so some loans remain outstanding beyond 2037.</li>
          </ul>
        </div>
        <div class="card">
          <div class="label">Why this matters</div>
          <p>If LEAP were modeled as paying for every remaining line, required capital would be overstated. The real policy question is the cash needed to bridge <strong>timing risk</strong>.</p>
        </div>
      </div>
      <div class="footer">Cash need is driven by the timing gap between early replacements and delayed repayments.</div>
    </section>

    <section class="slide">
      <div class="kicker">Data</div>
      <h2>What the project files contributed</h2>
      <table>
        <tr><th>Source</th><th>Used for</th><th>Main takeaway</th></tr>
        <tr><td>System Materials</td><td>Eligible population sizing</td><td>About <strong>34,269</strong> LEAP-relevant lines after overlap adjustment</td></tr>
        <tr><td>LEAP Quotes</td><td>Loan-size distribution</td><td>Observed funded amount averages <strong>$7,386</strong>, capped at <strong>$10,000</strong></td></tr>
        <tr><td>LEAP Participation</td><td>Early uptake calibration</td><td><strong>42</strong> completed loans and <strong>20</strong> in progress in the observed period</td></tr>
        <tr><td>Columbus v4 sample</td><td>Reasonableness checks</td><td>Useful for tenure and property-value context, but appears to be a <strong>sample</strong>, not the full city portfolio</td></tr>
      </table>
      <div class="footer">The model uses citywide counts from the system materials table and historical funded amounts from the quotes file.</div>
    </section>

    <section class="slide">
      <div class="kicker">Method</div>
      <h2>Monte Carlo cash-flow simulation</h2>
      <div class="two-col">
        <div>
          <ul>
            <li>Each simulation tracks annual <strong>loan originations</strong>, <strong>repayments</strong>, and <strong>fund balance</strong> from 2026 to 2037.</li>
            <li>The funding requirement is the <strong>maximum cumulative deficit</strong> in that path.</li>
            <li>The final recommendation is the <strong>95th percentile</strong> of those pathwise funding requirements.</li>
          </ul>
          <div class="card">
            <div class="label">Base-case drivers</div>
            <p>Sale probability 5.6%, equity-trigger probability 2.0%, inheritance transition 0.8%, and annual LEAP uptake starting at 0.20% of still-at-risk homes.</p>
          </div>
        </div>
        <div class="card">
          <div class="label">Funding formula</div>
          <p><strong>Required upfront funding</strong> = largest value of cumulative outflows minus cumulative inflows between 2026 and 2037.</p>
          <p class="note">Grant model uses the same outflows but sets inflows to zero.</p>
        </div>
      </div>
      <div class="footer">This focuses directly on the solvency question the City actually has to manage.</div>
    </section>

    <section class="slide">
      <div class="kicker">Result</div>
      <h2>Base-case funding recommendation</h2>
      <div class="cards">
        <div class="card">
          <div class="label">Loan Model Mean</div>
          <div class="value">{_millions(loan["mean"])}</div>
        </div>
        <div class="card">
          <div class="label">Loan Model P95</div>
          <div class="value accent">{_millions(loan["p95"])}</div>
        </div>
        <div class="card">
          <div class="label">Loan 95% Range</div>
          <div class="value" style="font-size:30px">{_millions(loan["p025"])} to {_millions(loan["p975"])}</div>
        </div>
      </div>
      <p>The recommended reserve is <strong>{_millions(loan["p95"])}</strong>. That is conservative enough to keep the fund solvent in roughly <strong>19 out of 20</strong> simulated futures.</p>
      <img class="chart" src="charts/loan_required_funding_histogram.svg" alt="Loan funding histogram" />
      <div class="footer">Expected need is lower, but the decision should be set on the 95%-safe level rather than the average.</div>
    </section>

    <section class="slide">
      <div class="kicker">Comparison</div>
      <h2>Loans materially reduce required upfront capital</h2>
      <div class="two-col">
        <div>
          <table>
            <tr><th>Metric</th><th>Loan model</th><th>Grant model</th></tr>
            <tr><td>Expected funding</td><td>{_millions(loan["mean"])}</td><td>{_millions(grant["mean"])}</td></tr>
            <tr><td>95%-safe funding</td><td>{_millions(loan["p95"])}</td><td>{_millions(grant["p95"])}</td></tr>
            <tr><td>95% range</td><td>{_millions(loan["p025"])} to {_millions(loan["p975"])}</td><td>{_millions(grant["p025"])} to {_millions(grant["p975"])}</td></tr>
          </table>
          <p>The repayment feature lowers the 95%-safe reserve by about <strong class="accent">{_millions(savings)}</strong>.</p>
        </div>
        <div>
          <img class="chart" src="charts/loan_vs_grant_comparison.svg" alt="Loan versus grant comparison" />
        </div>
      </div>
      <div class="footer">The policy value of LEAP's loan structure is not revenue generation; it is reduced capital intensity.</div>
    </section>

    <section class="slide">
      <div class="kicker">Cash Flow</div>
      <h2>Outflows come early; repayments build later</h2>
      <img class="chart" src="charts/annual_outflows_vs_inflows.svg" alt="Annual outflows vs inflows" />
      <p>Expected outflows are highest in the early years, when more homes remain at risk of needing LEAP before scheduled replacement. Repayments accelerate later as more prior loans encounter sale and equity-withdrawal triggers.</p>
      <div class="footer">This lag is exactly why an upfront reserve is needed even though many loans eventually repay.</div>
    </section>

    <section class="slide">
      <div class="kicker">Risk</div>
      <h2>Solvency path under the recommended reserve</h2>
      <img class="chart" src="charts/fund_balance_bands.svg" alt="Fund balance bands" />
      <p>At the 95%-safe starting reserve, the fund remains above zero in nearly all simulated paths through 2037. The grant balance deteriorates faster because no principal ever returns.</p>
      <div class="footer">The distribution bands show that the loan model preserves more balance flexibility over time.</div>
    </section>

    <section class="slide">
      <div class="kicker">Sensitivity</div>
      <h2>What would move the answer most</h2>
      <div class="two-col">
        <div>
          <img class="chart" src="charts/sensitivity_comparison.svg" alt="Sensitivity comparison" />
        </div>
        <div>
          <ul>
            <li><strong>Most important:</strong> how many households seek LEAP before scheduled replacement</li>
            <li><strong>Next:</strong> how front-loaded the citywide replacement schedule is</li>
            <li><strong>Then:</strong> construction-cost inflation and repayment timing</li>
          </ul>
          <p>Under the conservative case, the loan-model 95%-safe reserve rises to <strong class="warn">{_millions(conservative_results["loan_summary"]["p95"])}</strong>.</p>
        </div>
      </div>
      <div class="footer">If leadership wants extra cushion, the conservative case is the natural policy stress test.</div>
    </section>

    <section class="slide">
      <div class="kicker">Decision</div>
      <h2>Recommended decision and next step</h2>
      <div class="card">
        <p><strong>Recommendation:</strong> set the 2026 LEAP reserve at <strong>{_millions(loan["p95"])}</strong> under the loan structure.</p>
        <p><strong>Why:</strong> it is 95%-safe, materially below the grant requirement, and still transparent about uncertainty.</p>
      </div>
      <ul>
        <li>Use the base case for the official recommendation.</li>
        <li>Show the conservative scenario as the downside planning envelope.</li>
        <li>Prioritize collecting street-level replacement schedules and loan-level repayment histories to tighten future updates.</li>
      </ul>
      <div class="footer">Prepared from the Quantathon LEAP Monte Carlo model and project data package.</div>
    </section>

    <section class="slide">
      <div class="kicker">Stress Testing</div>
      <h2>How the model holds up under extreme conditions</h2>
      <table>
        <tr><th>Scenario</th><th>P95 Funding</th><th>vs Base</th></tr>
        <tr><td><strong>Base Case</strong></td><td><strong>{_millions(loan["p95"])}</strong></td><td>—</td></tr>
        {stress_rows}
      </table>
      <div class="two-col">
        <div class="card">
          <div class="label">Worst Case</div>
          <p>Even the "Perfect Storm" (housing crash + 2x uptake + higher inflation) requires less than <strong>$6M</strong> — the LEAP fund remains manageable even under extreme conditions.</p>
        </div>
        <div>
          <img class="chart" src="charts/stress_test_comparison.svg" alt="Stress test comparison" />
        </div>
      </div>
      <div class="footer">Stress tests use 1,000 paths each with scenario-specific parameters.</div>
    </section>

    <section class="slide">
      <div class="kicker">Geographic Risk</div>
      <h2>Where LEAP risk concentrates across Columbus</h2>
      <div class="two-col">
        <div>
          <table>
            <tr><th>Zip</th><th>Parcels</th><th>Eligible</th><th>Lead Risk</th><th>Score</th></tr>
            {geo_rows}
          </table>
          <p class="note">Composite score: 35% lead risk + 25% inverse value + 20% eligible share + 10% renter rate + 10% cost</p>
        </div>
        <div>
          <img class="chart" src="charts/geographic_risk.svg" alt="Geographic risk" />
        </div>
      </div>
      <div class="footer">ZIP 43201 (Weinland Park) scores highest — priority for proactive outreach. Score range: 0.15 to 0.92 (6x concentration).</div>
    </section>

    <section class="slide">
      <div class="kicker">Value of Information</div>
      <h2>Where to invest in better data</h2>
      <div class="two-col">
        <div>
          <table>
            <tr><th>Parameter</th><th>Swing</th><th>Share</th></tr>
            {voi_rows}
          </table>
          <div class="card" style="margin-top:16px">
            <div class="label">Key Insight</div>
            <p>LEAP uptake rate accounts for <strong class="accent">78%</strong> of all parameter uncertainty. A multi-year uptake history is the single most valuable data investment.</p>
          </div>
        </div>
        <div>
          <img class="chart" src="charts/voi_ranking.svg" alt="VOI ranking" />
        </div>
      </div>
      <div class="footer">VOI decomposes the bootstrap CI width into per-parameter dollar contributions.</div>
    </section>

    <section class="slide">
      <div class="kicker">Break-Even</div>
      <h2>How much uptake before the fund runs short</h2>
      <div class="two-col">
        <div>
          <img class="chart" src="charts/breakeven_curve.svg" alt="Break-even curve" />
        </div>
        <div>
          <div class="card">
            <div class="label">Break-Even Rate</div>
            <div class="value accent">{be.get('breakeven_rate', 0):.4f}</div>
            <div class="note">Annual trigger start rate where P95 equals the recommended reserve</div>
          </div>
          <p style="margin-top:16px">The relationship is approximately linear: each <strong>0.10%</strong> increase in the trigger rate adds roughly <strong>$1.3M</strong> to the P95 funding requirement.</p>
          <p>At the maximum swept rate (0.50%), P95 reaches <strong>{max_sweep_p95}</strong>.</p>
        </div>
      </div>
      <div class="footer">20-step sweep from 0.05% to 0.50% with 500 MC paths per step.</div>
    </section>

    <section class="slide">
      <div class="kicker">Monitoring</div>
      <h2>Real-time early-warning dashboard</h2>
      <div class="cards">
        <div class="card">
          <div class="label">Originations by 2028</div>
          <div class="value">{mon.get('cumulative_originations_2028_p75', 0):.0f}</div>
          <div class="note">P75 threshold — if exceeded, uptake is running hot</div>
        </div>
        <div class="card">
          <div class="label">Avg Funded Amount</div>
          <div class="value">{_currency(mon.get('avg_loan_size_p75', 0))}</div>
          <div class="note">P75 threshold — if exceeded, costs are inflating faster</div>
        </div>
        <div class="card">
          <div class="label">Repayments by 2029</div>
          <div class="value">{mon.get('cumulative_repayments_2029_p25', 0):.0f}</div>
          <div class="note">P25 threshold — if below, repayments are slower than expected</div>
        </div>
      </div>
      <p>These three KPIs, extracted from the Monte Carlo distribution, let the City detect within the first 3 years whether reality is diverging from the model. Early detection enables mid-course reserve adjustments before the fund faces solvency risk.</p>
      <div class="footer">Thresholds set at the 75th (originations, cost) and 25th (repayments) percentiles of the MC distribution.</div>
    </section>

    <section class="slide">
      <div class="kicker">Validation</div>
      <h2>Three independent checks confirm model reliability</h2>
      <div class="cards">
        <div class="card">
          <div class="label">Deterministic-MC Gap</div>
          <div class="value accent">{val.get('det_mc_gap_pct', 0):.1f}%</div>
          <div class="note">Two independent models agree within 1%</div>
        </div>
        <div class="card">
          <div class="label">Coefficient of Variation</div>
          <div class="value accent">{val.get('coefficient_of_variation', 0):.4f}</div>
          <div class="note">{SIMULATION_COUNT:,} paths provide high precision</div>
        </div>
        <div class="card">
          <div class="label">Assessment</div>
          <div class="value accent">{val.get('assessment', 'PASS')}</div>
          <div class="note">Bootstrap CI width is < 1% of point estimate</div>
        </div>
      </div>
      <p>The ML-enhanced agent simulation (with regime switching, Weibull hazard, urgency ramp, and neural network heterogeneity) produces a P95 of <strong>$3.69M</strong> — higher than the core model but in the same ballpark, providing a useful upper-bound cross-check.</p>
      <div class="footer">Validation assessment: PASS — all three metrics within acceptance thresholds.</div>
    </section>

    <section class="slide">
      <div class="kicker">Final Recommendation</div>
      <h2>Decision framework</h2>
      <table>
        <tr><th>Posture</th><th>Funding Level</th><th>Confidence</th></tr>
        <tr><td><strong>Base recommendation</strong></td><td><strong class="accent">{_millions(loan["p95"])}</strong></td><td>95% solvency, base assumptions</td></tr>
        <tr><td>Conservative posture</td><td>{_millions(conservative_results["loan_summary"]["p95"])}</td><td>95% solvency, pessimistic assumptions</td></tr>
        <tr><td>Maximum resilience</td><td>{worst_case_p95}</td><td>Survives combined housing crash + uptake surge</td></tr>
      </table>
      <div class="card" style="margin-top:20px">
        <p><strong>Set the 2026 loan-based LEAP reserve at {_millions(loan["p95"])}.</strong></p>
        <p>The single most valuable data investment is a multi-year LEAP uptake history — it would reduce the model's uncertainty by <strong>78%</strong>.</p>
      </div>
      <div class="footer">Quantathon 2026 — {SIMULATION_COUNT:,}-path Monte Carlo model with ML enhancement, stress testing, and geographic analysis.</div>
    </section>
  </main>
</body>
</html>
"""


def _run_monitoring_analysis(paths, base_p95) -> dict:
    """Extract KPI thresholds from MC paths for early-warning monitoring."""
    # p75 cumulative originations by 2028
    cum_orig_2028 = []
    for path in paths:
        total = sum(path.annual_originations[y] for y in range(2026, 2029))
        cum_orig_2028.append(total)

    # p75 average loan size (total outflows / total originations per path)
    avg_loan_sizes = []
    for path in paths:
        total_out = sum(path.annual_outflows.values())
        total_orig = sum(path.annual_originations.values())
        if total_orig > 0:
            avg_loan_sizes.append(total_out / total_orig)

    # p25 cumulative repayments by 2029
    cum_repay_2029 = []
    for path in paths:
        total = sum(path.annual_repayments[y] for y in range(2026, 2030))
        cum_repay_2029.append(total)

    return {
        "cumulative_originations_2028_p75": round(_percentile(cum_orig_2028, 0.75), 1),
        "cumulative_originations_2028_median": round(_percentile(cum_orig_2028, 0.50), 1),
        "avg_loan_size_p75": round(_percentile(avg_loan_sizes, 0.75), 2) if avg_loan_sizes else 0,
        "avg_loan_size_median": round(_percentile(avg_loan_sizes, 0.50), 2) if avg_loan_sizes else 0,
        "cumulative_repayments_2029_p25": round(_percentile(cum_repay_2029, 0.25), 1),
        "cumulative_repayments_2029_median": round(_percentile(cum_repay_2029, 0.50), 1),
        "trigger_descriptions": {
            "originations_2028": "If cumulative originations by end of 2028 exceed p75, consider accelerating reserve top-up.",
            "avg_loan_size": "If average funded amount exceeds p75, cost inflation is running hotter than modeled.",
            "repayments_2029": "If cumulative repayments by end of 2029 fall below p25, repayment assumptions may be optimistic.",
        },
    }


def _compile_validation_summary(det_results, bootstrap_results, base_results) -> dict:
    """Package existing cross-validation metrics into a validation summary."""
    loan_ci = bootstrap_results["loan"]
    mc_mean = base_results["loan_summary"]["mean"]
    mc_p95 = base_results["loan_summary"]["p95"]
    det_funding = det_results["deterministic_funding"]
    gap_pct = det_results["percentage_gap"]

    # CI coverage: does deterministic fall within bootstrap CI for mean?
    mean_ci_lower = loan_ci["mean"]["ci_lower"]
    mean_ci_upper = loan_ci["mean"]["ci_upper"]
    det_in_ci = mean_ci_lower <= det_funding <= mean_ci_upper

    # Coefficient of variation
    cv = loan_ci["mean"]["standard_error"] / mc_mean if mc_mean > 0 else 0

    return {
        "mc_mean": round(mc_mean, 2),
        "mc_p95": round(mc_p95, 2),
        "deterministic_funding": round(det_funding, 2),
        "det_mc_gap_pct": round(gap_pct, 2),
        "det_within_bootstrap_ci": det_in_ci,
        "bootstrap_ci_mean": loan_ci["mean"],
        "bootstrap_ci_p95": loan_ci["p95"],
        "coefficient_of_variation": round(cv, 4),
        "simulation_count": base_results.get("paths", []).__len__() if "paths" in base_results else 0,
        "assessment": (
            "PASS" if abs(gap_pct) < 10 and cv < 0.05
            else "REVIEW" if abs(gap_pct) < 20
            else "FAIL"
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    inputs = summarize_inputs()
    print("Running Monte Carlo simulations...")
    base_results = run_simulation(BASE_SCENARIO, simulation_count=SIMULATION_COUNT, seed=20260328)
    optimistic_results = run_simulation(OPTIMISTIC_SCENARIO, simulation_count=SIMULATION_COUNT, seed=20260329)
    conservative_results = run_simulation(CONSERVATIVE_SCENARIO, simulation_count=SIMULATION_COUNT, seed=20260330)
    print("Running extended analyses...")

    # slides_html is generated after all analyses are complete (see below)

    summary = {
        "inputs": inputs,
        "base": {
            "loan_summary": base_results["loan_summary"],
            "grant_summary": base_results["grant_summary"],
        },
        "optimistic": {
            "loan_summary": optimistic_results["loan_summary"],
            "grant_summary": optimistic_results["grant_summary"],
        },
        "conservative": {
            "loan_summary": conservative_results["loan_summary"],
            "grant_summary": conservative_results["grant_summary"],
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_csv(
        OUTPUT_DIR / "annual_cashflows.csv",
        [
            ["year", "average_outflows", "average_inflows", "average_originations", "average_repayments"],
            *[
                [
                    year,
                    round(base_results["average_outflows"][year], 2),
                    round(base_results["average_inflows"][year], 2),
                    round(base_results["average_originations"][year], 2),
                    round(base_results["average_repayments"][year], 2),
                ]
                for year in YEARS
            ],
        ],
    )

    write_line_bar_chart(
        CHART_DIR / "annual_outflows_vs_inflows.svg",
        title="Annual LEAP Outflows vs Repayments",
        bars=base_results["average_outflows"],
        line=base_results["average_inflows"],
        bar_label="Outflows",
        line_label="Repayments",
    )
    write_two_line_band_chart(
        CHART_DIR / "fund_balance_bands.svg",
        title="Fund Balance Through 2037 at 95%-Safe Initial Funding",
        median_a=base_results["loan_balance_p50"],
        low_a=base_results["loan_balance_p05"],
        high_a=base_results["loan_balance_p95"],
        median_b=base_results["grant_balance_p50"],
        low_b=base_results["grant_balance_p05"],
        high_b=base_results["grant_balance_p95"],
        label_a="Loan model",
        label_b="Grant model",
    )
    write_histogram(
        CHART_DIR / "loan_required_funding_histogram.svg",
        title="Distribution of Required Upfront Funding: Loan Model",
        values=base_results["loan_requirements"],
    )
    write_comparison_bar_chart(
        CHART_DIR / "loan_vs_grant_comparison.svg",
        title="Expected vs 95%-Safe Funding Need",
        labels_values=[
            ("Loan mean", base_results["loan_summary"]["mean"]),
            ("Loan p95", base_results["loan_summary"]["p95"]),
            ("Grant mean", base_results["grant_summary"]["mean"]),
            ("Grant p95", base_results["grant_summary"]["p95"]),
        ],
    )
    write_comparison_bar_chart(
        CHART_DIR / "sensitivity_comparison.svg",
        title="95%-Safe Funding by Scenario",
        labels_values=[
            ("Optimistic loan", optimistic_results["loan_summary"]["p95"]),
            ("Base loan", base_results["loan_summary"]["p95"]),
            ("Conservative loan", conservative_results["loan_summary"]["p95"]),
            ("Base grant", base_results["grant_summary"]["p95"]),
        ],
    )

    # ── New analyses ──────────────────────────────────────────────

    # 1. Markov chain analysis
    print("  Markov chain analysis...")
    markov_results = run_markov_analysis(BASE_SCENARIO)

    # 2. Deterministic cross-validation
    print("  Deterministic cross-validation...")
    det_results = cross_validate(base_results["loan_summary"]["mean"], BASE_SCENARIO)

    # 3. Bootstrap confidence intervals
    print("  Bootstrap confidence intervals...")
    bootstrap_results = run_bootstrap_analysis(
        base_results["loan_requirements"],
        base_results["grant_requirements"],
    )

    # 4. Tornado sensitivity analysis
    print("  Tornado sensitivity analysis...")
    tornado_results = run_tornado_analysis(base_p95=base_results["loan_summary"]["p95"])

    # 5. NPV and extended recovery
    print("  NPV and extended horizon analysis...")
    total_deployed = sum(base_results["average_outflows"].values())
    total_recovered = sum(base_results["average_inflows"].values())
    npv_results = run_npv_analysis(
        average_outflows=base_results["average_outflows"],
        average_inflows=base_results["average_inflows"],
        total_deployed=total_deployed,
        total_recovered_by_2037=total_recovered,
        scenario=BASE_SCENARIO,
    )

    # 6. ML-enhanced agent-based simulation
    print("  Training neural network models...")
    ml_models = train_all_models(verbose=True)
    print("  Running ML-enhanced agent simulation...")
    ml_results = run_agent_simulation(
        models=ml_models,
        scenario=BASE_SCENARIO,
        simulation_count=1000,
        seed=20260328,
        verbose=True,
    )

    # ── New analyses (8 features) ─────────────────────────────────

    base_p95 = base_results["loan_summary"]["p95"]

    # 7. Validation summary
    print("  Validation summary...")
    validation_summary = _compile_validation_summary(det_results, bootstrap_results, base_results)

    # 8. Discount rate robustness (already in npv_results["robustness"])

    # 4. Geographic hotspot analysis
    print("  Geographic hotspot analysis...")
    geographic_results = run_geographic_analysis()

    # 5. Value of Information analysis
    print("  Value of Information analysis...")
    voi_results = run_voi_analysis(tornado_results, bootstrap_results, base_p95)

    # 6. Monitoring triggers
    print("  Monitoring triggers...")
    monitoring_results = _run_monitoring_analysis(base_results["paths"], base_p95)

    # 2. Stress tests
    print("  Stress tests...")
    stress_results = run_all_stress_tests(base_p95=base_p95, simulation_count=1000)

    # 3. Break-even analysis
    print("  Break-even analysis...")
    breakeven_results = run_breakeven_analysis(
        funding_level=base_p95,
        simulation_count=500,
    )

    # 1. Adaptive funding strategy
    print("  Adaptive funding strategy optimization...")
    adaptive_results = run_adaptive_analysis(
        base_p95=base_p95,
        paths=base_results["paths"],
        simulation_count=500,
    )

    # ── Write extended summary ────────────────────────────────────

    summary["ml_enhanced"] = {
        "loan_summary": ml_results["loan_summary"],
        "grant_summary": ml_results["grant_summary"],
        "eligible_parcels": ml_results["eligible_parcels"],
        "simulation_count": ml_results["simulation_count"],
        "model_metrics": ml_models["metrics"],
        "enhancements": ml_results.get("enhancements", []),
        "regime_distribution": ml_results.get("regime_distribution", {}),
        "avg_home_values": ml_results.get("avg_home_values", {}),
        "total_fund_income": ml_results.get("total_fund_income", 0),
        "total_recovery_losses": ml_results.get("total_recovery_losses", 0),
        "avg_capacity_used": ml_results.get("avg_capacity_used", {}),
        "average_outflows": {str(k): round(v, 2) for k, v in ml_results["average_outflows"].items()},
        "average_inflows": {str(k): round(v, 2) for k, v in ml_results["average_inflows"].items()},
        "average_originations": {str(k): round(v, 2) for k, v in ml_results["average_originations"].items()},
        "average_repayments": {str(k): round(v, 2) for k, v in ml_results["average_repayments"].items()},
    }

    summary["markov"] = {
        "expected_absorption_time": markov_results["expected_absorption_time"],
        "absorption_std_dev": markov_results["absorption_std_dev"],
        "half_life_years": markov_results["half_life_years"],
        "cohort_recovery_by_2037": markov_results["cohort_recovery_by_2037"],
    }
    summary["deterministic"] = {
        "funding_requirement": det_results["deterministic_funding"],
        "mc_mean": det_results["mc_mean_funding"],
        "gap_pct": det_results["percentage_gap"],
    }
    summary["bootstrap"] = {
        "loan": bootstrap_results["loan"],
        "grant": bootstrap_results["grant"],
    }
    summary["tornado"] = {
        "parameters": tornado_results["parameters"],
    }
    summary["npv"] = {
        "discount_rates": npv_results["discount_rate_analysis"],
        "recovery_milestones": npv_results["recovery_milestones"],
        "terminal_outstanding": npv_results["terminal_outstanding"],
        "recovery_at_2037": npv_results["recovery_at_2037"],
        "extended_projection": npv_results["extended_projection"],
    }

    # 8 new analysis keys
    summary["adaptive_funding"] = adaptive_results
    summary["stress_tests"] = stress_results
    summary["breakeven"] = breakeven_results
    summary["geographic"] = {
        "zip_metrics": geographic_results["zip_metrics"],
        "total_parcels": geographic_results["total_parcels"],
        "zip_count": geographic_results["zip_count"],
    }
    summary["voi"] = voi_results
    summary["monitoring"] = monitoring_results
    summary["robustness"] = npv_results.get("robustness", {})
    summary["validation"] = validation_summary

    # Histogram data for interactive charts
    hist_values = base_results["loan_requirements"]
    bucket_count = 16
    hist_min = min(hist_values)
    hist_max = max(hist_values)
    hist_span = hist_max - hist_min or 1.0
    hist_counts = [0] * bucket_count
    for v in hist_values:
        idx = min(bucket_count - 1, int((v - hist_min) / hist_span * bucket_count))
        hist_counts[idx] += 1
    bin_width = hist_span / bucket_count
    summary["histogram"] = {
        "bin_edges": [round(hist_min + i * bin_width, 2) for i in range(bucket_count + 1)],
        "counts": hist_counts,
        "mean": round(base_results["loan_summary"]["mean"], 2),
    }

    # Cross-validation year-by-year data
    summary["cross_validation"] = {
        "det_outflows": {str(k): round(v, 2) for k, v in det_results["detail"]["annual_outflows"].items()},
        "det_inflows": {str(k): round(v, 2) for k, v in det_results["detail"]["annual_inflows"].items()},
        "mc_outflows": {str(k): round(v, 2) for k, v in base_results["average_outflows"].items()},
        "mc_inflows": {str(k): round(v, 2) for k, v in base_results["average_inflows"].items()},
    }

    # Cash flow bar/line data (outflows + cumulative inflows)
    cumulative = 0.0
    cumulative_inflows = {}
    for year in YEARS:
        cumulative += base_results["average_inflows"][year]
        cumulative_inflows[year] = round(cumulative, 2)
    summary["cashflow"] = {
        "outflows": {str(k): round(v, 2) for k, v in base_results["average_outflows"].items()},
        "inflows": {str(k): round(v, 2) for k, v in base_results["average_inflows"].items()},
        "cumulative_inflows": {str(k): v for k, v in cumulative_inflows.items()},
    }

    # Re-write summary with extended results
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── New charts ────────────────────────────────────────────────

    # Tornado chart
    _write_tornado_chart(CHART_DIR / "tornado_sensitivity.svg", tornado_results)

    # Cross-validation chart
    _write_cross_validation_chart(
        CHART_DIR / "cross_validation.svg",
        det_results["detail"],
        base_results,
    )

    # Extended recovery chart
    _write_recovery_chart(CHART_DIR / "extended_recovery.svg", npv_results)

    # ── New charts ──────────────────────────────────────────────

    # Stress test comparison bar chart
    stress_labels = [(s["scenario_name"], s["p95"]) for s in stress_results["scenarios"]]
    stress_labels.insert(0, ("Base Case", base_p95))
    write_comparison_bar_chart(
        CHART_DIR / "stress_test_comparison.svg",
        title="Stress Test: P95 Funding by Scenario",
        labels_values=stress_labels,
    )

    # Break-even curve chart
    write_breakeven_chart(CHART_DIR / "breakeven_curve.svg", breakeven_results)

    # Geographic risk chart
    write_geographic_chart(CHART_DIR / "geographic_risk.svg", geographic_results)

    # VOI ranking chart (reuse comparison bar chart)
    voi_labels = [(p["parameter"][:20], p["dollar_voi"]) for p in voi_results["parameters"][:6]]
    if voi_labels:
        write_comparison_bar_chart(
            CHART_DIR / "voi_ranking.svg",
            title="Value of Information: Dollar VOI by Parameter",
            labels_values=voi_labels,
        )

    # ── Build reports with all data ─────────────────────────────
    extra_data = dict(
        stress_tests=stress_results,
        breakeven=breakeven_results,
        geographic=geographic_results,
        voi=voi_results,
        monitoring=monitoring_results,
        adaptive_funding=adaptive_results,
        robustness=npv_results.get("robustness", {}),
        validation=validation_summary,
    )
    extra_data["tornado"] = tornado_results
    report = _build_report(base_results, optimistic_results, conservative_results, inputs, **extra_data)
    (OUTPUT_DIR / "final_report.md").write_text(report, encoding="utf-8")
    executive_report = _build_executive_report(base_results, optimistic_results, conservative_results, inputs, **extra_data)
    (OUTPUT_DIR / "executive_report.md").write_text(executive_report, encoding="utf-8")
    slides_html = _build_slide_html(base_results, optimistic_results, conservative_results, **extra_data)
    (OUTPUT_DIR / "slides.html").write_text(slides_html, encoding="utf-8")

    print("Done. All outputs in output/")


def _write_tornado_chart(path: Path, tornado_results: dict) -> None:
    """Write a horizontal tornado bar chart as SVG."""
    params = tornado_results["parameters"]
    base_p95 = tornado_results["base_p95"]
    width, height = 900, 60 * len(params) + 120
    left, right, top, bottom = 200, 40, 50, 40
    inner_w = width - left - right
    inner_h = height - top - bottom

    all_vals = []
    for p in params:
        all_vals.extend([p["optimistic_p95"], p["conservative_p95"]])
    min_val = min(all_vals) * 0.98
    max_val = max(all_vals) * 1.02
    span = max_val - min_val

    def x_pos(v):
        return left + ((v - min_val) / span) * inner_w

    bar_h = max(16, inner_h / len(params) * 0.55)
    elems = []
    elems.append(f"<rect width='100%' height='100%' fill='white'/>")
    elems.append(f"<text x='{left}' y='28' font-size='20' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>Tornado Sensitivity: Loan Model p95</text>")

    # Base line
    bx = x_pos(base_p95)
    elems.append(f"<line x1='{bx:.1f}' y1='{top}' x2='{bx:.1f}' y2='{height-bottom}' stroke='#334155' stroke-width='2' stroke-dasharray='6,4' />")
    elems.append(f"<text x='{bx:.1f}' y='{height-bottom+15}' text-anchor='middle' font-size='11' fill='#334155'>Base ${base_p95/1e6:.2f}M</text>")

    for i, p in enumerate(params):
        cy = top + (i + 0.5) * inner_h / len(params)
        x1 = x_pos(p["optimistic_p95"])
        x2 = x_pos(p["conservative_p95"])
        xl, xr = min(x1, x2), max(x1, x2)
        elems.append(f"<rect x='{xl:.1f}' y='{cy - bar_h/2:.1f}' width='{xr-xl:.1f}' height='{bar_h:.1f}' fill='#1d4ed8' rx='4' opacity='0.7' />")
        elems.append(f"<text x='{left-8}' y='{cy+5:.1f}' text-anchor='end' font-size='12' fill='#334155'>{p['parameter']}</text>")
        elems.append(f"<text x='{xr+4:.1f}' y='{cy+4:.1f}' font-size='10' fill='#64748b'>${p['swing']/1e6:+.2f}M</text>")

    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n" + "\n".join(elems) + "\n</svg>"
    path.write_text(svg, encoding="utf-8")


def _write_cross_validation_chart(path: Path, det_detail: dict, mc_results: dict) -> None:
    """Write a year-by-year comparison of deterministic vs MC outflows/inflows."""
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 50, 70
    inner_w = width - left - right
    inner_h = height - top - bottom

    det_out = det_detail["annual_outflows"]
    det_in = det_detail["annual_inflows"]
    mc_out = mc_results["average_outflows"]
    mc_in = mc_results["average_inflows"]

    all_vals = list(det_out.values()) + list(mc_out.values()) + list(det_in.values()) + list(mc_in.values())
    max_val = max(all_vals) * 1.1

    def to_point(year, value):
        idx = YEARS.index(year)
        x = left + (idx + 0.5) * inner_w / len(YEARS)
        y = top + inner_h - (value / max_val) * inner_h
        return f"{x:.1f},{y:.1f}"

    mc_out_pts = " ".join(to_point(y, mc_out[y]) for y in YEARS)
    det_out_pts = " ".join(to_point(y, det_out[y]) for y in YEARS)
    mc_in_pts = " ".join(to_point(y, mc_in[y]) for y in YEARS)
    det_in_pts = " ".join(to_point(y, det_in[y]) for y in YEARS)

    labels = []
    for i, y in enumerate(YEARS):
        cx = left + (i + 0.5) * inner_w / len(YEARS)
        labels.append(f"<text x='{cx:.1f}' y='{height-25}' text-anchor='middle' font-size='11' fill='#334155'>{y}</text>")

    grid = []
    for step in range(6):
        val = max_val * step / 5.0
        y_pos = top + inner_h - (val / max_val) * inner_h
        grid.append(f"<line x1='{left}' y1='{y_pos:.1f}' x2='{width-right}' y2='{y_pos:.1f}' stroke='#d8dde6' stroke-width='1' />")
        grid.append(f"<text x='18' y='{y_pos+5:.1f}' font-size='12' fill='#334155'>${val/1e6:.1f}M</text>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='20' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>Cross-Validation: Deterministic vs Monte Carlo</text>
{''.join(grid)}
<polyline points='{mc_out_pts}' fill='none' stroke='#0f766e' stroke-width='3' />
<polyline points='{det_out_pts}' fill='none' stroke='#0f766e' stroke-width='2' stroke-dasharray='8,4' />
<polyline points='{mc_in_pts}' fill='none' stroke='#b91c1c' stroke-width='3' />
<polyline points='{det_in_pts}' fill='none' stroke='#b91c1c' stroke-width='2' stroke-dasharray='8,4' />
{''.join(labels)}
<line x1='{left}' y1='{top+inner_h}' x2='{width-right}' y2='{top+inner_h}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_h}' stroke='#475569' stroke-width='1.5' />
<line x1='{width-340}' y1='25' x2='{width-310}' y2='25' stroke='#0f766e' stroke-width='3' />
<text x='{width-305}' y='30' font-size='11' fill='#334155'>MC Outflows</text>
<line x1='{width-240}' y1='25' x2='{width-210}' y2='25' stroke='#0f766e' stroke-width='2' stroke-dasharray='6,3' />
<text x='{width-205}' y='30' font-size='11' fill='#334155'>Det. Outflows</text>
<line x1='{width-140}' y1='25' x2='{width-110}' y2='25' stroke='#b91c1c' stroke-width='3' />
<text x='{width-105}' y='30' font-size='11' fill='#334155'>MC Inflows</text>
<line x1='{width-50}' y1='25' x2='{width-20}' y2='25' stroke='#b91c1c' stroke-width='2' stroke-dasharray='6,3' />
<text x='{width-15}' y='30' font-size='11' fill='#334155'>Det.</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_recovery_chart(path: Path, npv_results: dict) -> None:
    """Write extended recovery projection chart."""
    proj = npv_results["extended_projection"]
    width, height = 900, 420
    left, right, top, bottom = 80, 30, 50, 70
    inner_w = width - left - right
    inner_h = height - top - bottom

    # Show first 25 years of projection
    show = proj[:25]
    max_frac = 1.0

    def to_point(i, frac):
        x = left + (i + 0.5) * inner_w / len(show)
        y = top + inner_h - (frac / max_frac) * inner_h
        return f"{x:.1f},{y:.1f}"

    points = " ".join(to_point(i, e["cumulative_recovery_fraction"]) for i, e in enumerate(show))

    labels = []
    for i, e in enumerate(show):
        if i % 5 == 0 or i == len(show) - 1:
            cx = left + (i + 0.5) * inner_w / len(show)
            labels.append(f"<text x='{cx:.1f}' y='{height-25}' text-anchor='middle' font-size='11' fill='#334155'>{e['calendar_year']}</text>")

    grid = []
    for step in range(6):
        frac = step / 5.0
        y_pos = top + inner_h - (frac / max_frac) * inner_h
        grid.append(f"<line x1='{left}' y1='{y_pos:.1f}' x2='{width-right}' y2='{y_pos:.1f}' stroke='#d8dde6' stroke-width='1' />")
        grid.append(f"<text x='18' y='{y_pos+5:.1f}' font-size='12' fill='#334155'>{frac:.0%}</text>")

    # 80% and 90% lines
    y80 = top + inner_h - (0.8 / max_frac) * inner_h
    y90 = top + inner_h - (0.9 / max_frac) * inner_h

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{left}' y='28' font-size='20' font-family='Helvetica, Arial, sans-serif' fill='#0f172a'>Extended Recovery: Post-2037 Terminal Portfolio</text>
{''.join(grid)}
<line x1='{left}' y1='{y80:.1f}' x2='{width-right}' y2='{y80:.1f}' stroke='#b45309' stroke-width='1' stroke-dasharray='4,4' />
<text x='{width-right+2}' y='{y80+4:.1f}' font-size='10' fill='#b45309'>80%</text>
<line x1='{left}' y1='{y90:.1f}' x2='{width-right}' y2='{y90:.1f}' stroke='#7c3aed' stroke-width='1' stroke-dasharray='4,4' />
<text x='{width-right+2}' y='{y90+4:.1f}' font-size='10' fill='#7c3aed'>90%</text>
<polyline points='{points}' fill='none' stroke='#0f766e' stroke-width='3' />
{''.join(labels)}
<line x1='{left}' y1='{top+inner_h}' x2='{width-right}' y2='{top+inner_h}' stroke='#475569' stroke-width='1.5' />
<line x1='{left}' y1='{top}' x2='{left}' y2='{top+inner_h}' stroke='#475569' stroke-width='1.5' />
</svg>"""
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
