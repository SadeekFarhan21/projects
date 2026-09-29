"""Enhanced agent-based Monte Carlo simulation with 4 advanced features.

Each parcel is an individual agent with NN-predicted baseline rates, modulated by:
1. Housing market regime switching (Boom/Normal/Recession Markov chain)
2. Weibull survival-based sale hazard (time-dependent, not flat)
3. Deadline urgency ramp (logistic S-curve as 2037 approaches)
4. Home price appreciation & equity-dependent extraction probability

Neural network models are trained on v2 LEAP data and applied to the
Columbus v4 test set parcels to generate heterogeneous per-parcel
predictions.  The v4 parcels are then scaled to the citywide eligible
count for simulation.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean
from typing import Dict, List, Tuple

import numpy as np

from .assumptions import (
    END_YEAR, START_YEAR, YEARS, Scenario, BASE_SCENARIO,
    REGIME_BOOM, REGIME_NORMAL, REGIME_RECESSION,
)
from .data_utils import load_system_materials
from .ml_models import (
    load_v4_parcels,
    predict_parcel_features,
    train_all_models,
    load_trained_models,
)


REGIME_NAMES = {REGIME_BOOM: "Boom", REGIME_NORMAL: "Normal", REGIME_RECESSION: "Recession"}


@dataclass
class AgentPathResult:
    annual_outflows: Dict[int, float]
    annual_inflows: Dict[int, float]
    annual_originations: Dict[int, int]
    annual_repayments: Dict[int, int]
    required_funding_loan: float
    required_funding_grant: float
    regime_history: List[int]
    annual_avg_home_value: Dict[int, float]
    annual_fund_income: Dict[int, float]       # investment returns on fund balance
    annual_capacity_used: Dict[int, int]        # contractor slots used per year
    annual_recovery_losses: Dict[int, float]    # $ lost to partial recovery


# ---------------------------------------------------------------------------
# Regime switching
# ---------------------------------------------------------------------------

def _transition_regime(current: int, matrix: List[List[float]], rng: random.Random) -> int:
    """Transition to next housing market regime via Markov chain."""
    row = matrix[current]
    draw = rng.random()
    cumulative = 0.0
    for next_regime, prob in enumerate(row):
        cumulative += prob
        if draw < cumulative:
            return next_regime
    return len(row) - 1


# ---------------------------------------------------------------------------
# Weibull hazard
# ---------------------------------------------------------------------------

def _weibull_hazard(t: float, k: float, lam: float) -> float:
    """Compute Weibull hazard rate h(t) = (k/λ)(t/λ)^(k-1).

    Returns the instantaneous hazard at time t (years since origination).
    """
    if t < 1.0:
        t = 1.0  # guard against t=0
    return (k / lam) * (t / lam) ** (k - 1)


def _weibull_normalization(k: float, lam: float) -> float:
    """Compute the average Weibull hazard over 1-12 years for normalization.

    This ensures the NN-predicted baseline sale prob remains calibrated on average.
    """
    total = sum(_weibull_hazard(t, k, lam) for t in range(1, 13))
    return total / 12.0


# ---------------------------------------------------------------------------
# Deadline urgency
# ---------------------------------------------------------------------------

def _urgency_multiplier(year: int, A: float, B: float, inflection: int) -> float:
    """Logistic S-curve urgency multiplier.

    Returns 1.0 + A / (1 + exp(-B * (year - inflection)))
    Ramps from ~1.0 early to ~1+A by 2037.
    """
    return 1.0 + A / (1.0 + math.exp(-B * (year - inflection)))


# ---------------------------------------------------------------------------
# Loss given default
# ---------------------------------------------------------------------------

def _compute_recovery(
    loan_principal: float,
    current_home_value: float,
    is_recession: bool,
    scenario: Scenario,
) -> float:
    """Compute actual recovery amount on a sale-triggered repayment.

    In normal/boom markets: full principal recovery (home value >> loan).
    In recession: if home value has dropped, city may only recover a fraction.
    The LEAP lien is subordinate to the primary mortgage, so recovery depends
    on whether there's enough equity after the mortgage is satisfied.
    """
    if not is_recession:
        return loan_principal

    # In recession, apply a haircut to reflect distressed sales
    # Recovery = min(principal, home_value * (1 - haircut))
    distressed_value = current_home_value * (1.0 - scenario.recession_recovery_haircut)
    recovery = min(loan_principal, max(0.0, distressed_value))
    # Floor: never recover less than ltv_recovery_floor * principal
    recovery = max(recovery, loan_principal * scenario.ltv_recovery_floor)
    return round(recovery, 2)


# ---------------------------------------------------------------------------
# Parcel scaling
# ---------------------------------------------------------------------------

def _scale_parcels_to_citywide(
    parcels: List[Dict],
    predictions: Dict[str, np.ndarray],
    target_eligible: int,
    rng: random.Random,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """Scale the v4 test set up to citywide eligible count."""
    eligible_idx = [
        i for i, p in enumerate(parcels)
        if p["line_material"] in ("lead", "galvanized")
    ]

    eligible_parcels = [parcels[i] for i in eligible_idx]
    eligible_uptake = predictions["uptake_prob"][eligible_idx]
    eligible_sale = predictions["sale_prob"][eligible_idx]
    eligible_cost = predictions["replacement_cost"][eligible_idx]

    n_eligible = len(eligible_parcels)
    if n_eligible >= target_eligible:
        chosen = rng.sample(range(n_eligible), target_eligible)
        return (
            [eligible_parcels[i] for i in chosen],
            {
                "uptake_prob": eligible_uptake[chosen],
                "sale_prob": eligible_sale[chosen],
                "replacement_cost": eligible_cost[chosen],
            },
        )

    # Replicate with noise
    replicated_parcels = list(eligible_parcels)
    replicated_uptake = list(eligible_uptake)
    replicated_sale = list(eligible_sale)
    replicated_cost = list(eligible_cost)

    remaining = target_eligible - n_eligible
    while remaining > 0:
        batch = min(remaining, n_eligible)
        indices = rng.sample(range(n_eligible), batch)
        for i in indices:
            replicated_parcels.append(eligible_parcels[i])
            noise = 1.0 + rng.gauss(0, 0.05)
            replicated_uptake.append(eligible_uptake[i] * max(0.5, min(1.5, noise)))
            noise = 1.0 + rng.gauss(0, 0.05)
            replicated_sale.append(eligible_sale[i] * max(0.5, min(1.5, noise)))
            noise = 1.0 + rng.gauss(0, 0.03)
            replicated_cost.append(eligible_cost[i] * max(0.9, min(1.1, noise)))
        remaining -= batch

    return (
        replicated_parcels[:target_eligible],
        {
            "uptake_prob": np.array(replicated_uptake[:target_eligible], dtype=np.float32),
            "sale_prob": np.array(replicated_sale[:target_eligible], dtype=np.float32),
            "replacement_cost": np.array(replicated_cost[:target_eligible], dtype=np.float32),
        },
    )


def _assign_schedule_years(
    n_parcels: int,
    schedule_weights: List[float],
    rng: random.Random,
) -> List[int]:
    """Assign each parcel a scheduled city replacement year."""
    total_weight = sum(schedule_weights)
    cumulative = []
    running = 0.0
    for w in schedule_weights:
        running += w / total_weight
        cumulative.append(running)

    assignments = []
    for _ in range(n_parcels):
        r = rng.random()
        for i, thresh in enumerate(cumulative):
            if r <= thresh:
                assignments.append(YEARS[i])
                break
        else:
            assignments.append(YEARS[-1])
    return assignments


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_agent_path(
    parcels: List[Dict],
    predictions: Dict[str, np.ndarray],
    scenario: Scenario,
    rng: random.Random,
) -> AgentPathResult:
    """Run one simulation path with all 4 enhancements.

    Enhancements:
    1. Housing market regime switching (Boom/Normal/Recession)
    2. Weibull survival-based sale hazard
    3. Deadline urgency ramp (logistic S-curve)
    4. Home price appreciation & equity-dependent extraction
    """
    n_parcels = len(parcels)
    schedule_years = _assign_schedule_years(n_parcels, scenario.schedule_weights, rng)

    uptake_probs = predictions["uptake_prob"]
    sale_probs = predictions["sale_prob"]
    costs = predictions["replacement_cost"]

    # Per-parcel state
    parcel_status = [-1] * n_parcels  # -1=untriggered, 0=active, 1=inherited, 2=repaid
    parcel_loan_amount = [0.0] * n_parcels
    parcel_origination_year = [0] * n_parcels

    # Enhancement 4: Home value tracking
    parcel_home_value = [float(p["assessed_value"]) for p in parcels]
    parcel_last_sale_year = [int(p["last_sale_year"]) for p in parcels]

    # Enhancement 2: Weibull normalization constant
    weibull_norm = _weibull_normalization(scenario.weibull_k, scenario.weibull_lambda)

    # Regime state (Enhancement 1)
    current_regime = scenario.initial_regime
    regime_history = []

    annual_outflows = {year: 0.0 for year in YEARS}
    annual_inflows = {year: 0.0 for year in YEARS}
    annual_originations = {year: 0 for year in YEARS}
    annual_repayments = {year: 0 for year in YEARS}
    annual_avg_home_value = {year: 0.0 for year in YEARS}
    annual_fund_income = {year: 0.0 for year in YEARS}
    annual_capacity_used = {year: 0 for year in YEARS}
    annual_recovery_losses = {year: 0.0 for year in YEARS}

    for year in YEARS:
        years_from_start = year - START_YEAR

        # ── Enhancement 1: Regime transition ──
        current_regime = _transition_regime(
            current_regime, scenario.regime_transition_matrix, rng
        )
        regime_history.append(current_regime)
        sale_multiplier = scenario.regime_sale_multipliers[current_regime]
        equity_multiplier = scenario.regime_equity_multipliers[current_regime]
        appreciation_rate = scenario.regime_appreciation_rates[current_regime]
        is_recession = (current_regime == REGIME_RECESSION)

        # ── Enhancement 4: Update home values ──
        for i in range(n_parcels):
            parcel_home_value[i] *= (1.0 + appreciation_rate)

        annual_avg_home_value[year] = sum(parcel_home_value) / n_parcels

        # ── Enhancement 3: Deadline urgency ──
        urgency = _urgency_multiplier(
            year, scenario.urgency_A, scenario.urgency_B, scenario.urgency_inflection
        )

        # Cost inflation
        inflation_factor = (1.0 + scenario.annual_cost_inflation) ** years_from_start

        # ── NEW: Contractor capacity for this year ──
        capacity = min(
            scenario.contractor_capacity_max,
            int(scenario.contractor_capacity_base * (1.0 + scenario.contractor_capacity_growth) ** years_from_start),
        )
        originations_this_year = 0

        # ── Phase 1: New LEAP loans (with capacity constraint) ──
        adjusted_cap = scenario.annual_trigger_cap * min(urgency, 2.0)

        # Shuffle parcel order to avoid systematic bias in who gets served first
        parcel_order = list(range(n_parcels))
        rng.shuffle(parcel_order)

        for i in parcel_order:
            if originations_this_year >= capacity:
                break  # contractor capacity exhausted
            if parcel_status[i] != -1:
                continue
            if schedule_years[i] <= year:
                continue

            p_uptake = min(uptake_probs[i] * urgency, adjusted_cap)

            if rng.random() < p_uptake:
                raw_cost = costs[i] * inflation_factor
                loan_amount = min(10000.0, round(raw_cost, 2))
                parcel_status[i] = 0
                parcel_loan_amount[i] = loan_amount
                parcel_origination_year[i] = year
                annual_outflows[year] += loan_amount
                annual_originations[year] += 1
                originations_this_year += 1

        annual_capacity_used[year] = originations_this_year

        # ── Phase 2: Loan repayments (with loss given default) ──
        for i in range(n_parcels):
            if parcel_status[i] == -1 or parcel_status[i] == 2:
                continue
            if parcel_origination_year[i] >= year:
                continue

            if parcel_status[i] == 0:  # Active loan
                # Enhancement 2: Weibull hazard
                t = year - parcel_origination_year[i]
                weibull_h = _weibull_hazard(t, scenario.weibull_k, scenario.weibull_lambda)
                p_sale = min(0.25, sale_probs[i] * (weibull_h / weibull_norm) * sale_multiplier)

                # Enhancement 4: Equity-dependent extraction
                mortgage_age = year - parcel_last_sale_year[i]
                mortgage_fraction = max(0.0, 1.0 - mortgage_age * scenario.mortgage_amortization_rate)
                equity = parcel_home_value[i] - (parcels[i]["assessed_value"] * mortgage_fraction)
                equity_ratio = max(0.0, equity / parcel_home_value[i]) if parcel_home_value[i] > 0 else 0.0
                equity_factor = max(0.2, min(2.5, equity_ratio / scenario.equity_baseline_ratio))
                p_equity = min(0.15, scenario.annual_equity_probability * equity_factor * equity_multiplier)

                p_inherit = scenario.annual_inheritance_probability

                draw = rng.random()
                if draw < p_sale:
                    # NEW: Loss given default — partial recovery in recession
                    recovery = _compute_recovery(
                        parcel_loan_amount[i], parcel_home_value[i],
                        is_recession, scenario,
                    )
                    annual_inflows[year] += recovery
                    annual_recovery_losses[year] += parcel_loan_amount[i] - recovery
                    annual_repayments[year] += 1
                    parcel_status[i] = 2
                    continue
                draw -= p_sale
                if draw < p_equity:
                    # Equity extraction: full recovery (owner is refinancing, not selling)
                    annual_inflows[year] += parcel_loan_amount[i]
                    annual_repayments[year] += 1
                    parcel_status[i] = 2
                    continue
                draw -= p_equity
                if draw < p_inherit:
                    parcel_status[i] = 1
                    continue

            elif parcel_status[i] == 1:  # Inherited
                p_inherited_sale = min(0.25, scenario.inherited_sale_probability * sale_multiplier)
                p_inherited_equity = min(0.15, scenario.inherited_equity_probability * equity_multiplier)

                draw = rng.random()
                if draw < p_inherited_sale:
                    recovery = _compute_recovery(
                        parcel_loan_amount[i], parcel_home_value[i],
                        is_recession, scenario,
                    )
                    annual_inflows[year] += recovery
                    annual_recovery_losses[year] += parcel_loan_amount[i] - recovery
                    annual_repayments[year] += 1
                    parcel_status[i] = 2
                    continue
                draw -= p_inherited_sale
                if draw < p_inherited_equity:
                    annual_inflows[year] += parcel_loan_amount[i]
                    annual_repayments[year] += 1
                    parcel_status[i] = 2

    # ── Compute required funding ──
    # Pass 1: Find max deficit without investment returns (grant & raw loan)
    cumulative_loan_raw = 0.0
    cumulative_grant = 0.0
    max_deficit_loan_raw = 0.0
    max_deficit_grant = 0.0
    for year in YEARS:
        cumulative_loan_raw += annual_outflows[year] - annual_inflows[year]
        cumulative_grant += annual_outflows[year]
        max_deficit_loan_raw = max(max_deficit_loan_raw, cumulative_loan_raw)
        max_deficit_grant = max(max_deficit_grant, cumulative_grant)

    # Pass 2: Simulate fund with investment returns starting from the raw funding amount
    # The fund starts with max_deficit_loan_raw and earns interest on positive balance
    fund_balance = max_deficit_loan_raw
    total_income = 0.0
    for year in YEARS:
        # Fund earns returns on remaining positive balance
        if fund_balance > 0:
            income = fund_balance * scenario.fund_annual_return
            annual_fund_income[year] = round(income, 2)
            fund_balance += income
            total_income += income
        # Apply this year's net flow
        fund_balance += annual_inflows[year] - annual_outflows[year]

    # Net required funding = raw deficit - income earned
    # (investment returns reduce how much the city actually needs upfront)
    max_deficit_loan = max(0.0, max_deficit_loan_raw - total_income)

    return AgentPathResult(
        annual_outflows=annual_outflows,
        annual_inflows=annual_inflows,
        annual_originations=annual_originations,
        annual_repayments=annual_repayments,
        required_funding_loan=round(max_deficit_loan, 2),
        required_funding_grant=round(max_deficit_grant, 2),
        regime_history=regime_history,
        annual_avg_home_value=annual_avg_home_value,
        annual_fund_income=annual_fund_income,
        annual_capacity_used=annual_capacity_used,
        annual_recovery_losses=annual_recovery_losses,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def run_agent_simulation(
    models: Dict = None,
    scenario: Scenario = None,
    simulation_count: int = 1000,
    seed: int = 20260328,
    verbose: bool = True,
) -> Dict:
    """Run the full enhanced agent-based MC simulation."""
    if scenario is None:
        scenario = BASE_SCENARIO
    if models is None:
        models = load_trained_models()

    parcels = load_v4_parcels()
    predictions = predict_parcel_features(parcels, models)

    materials = load_system_materials()
    target_eligible = materials["eligible_lines"]

    if verbose:
        eligible_count = sum(1 for p in parcels if p["line_material"] in ("lead", "galvanized"))
        print(f"v4 eligible parcels: {eligible_count:,}")
        print(f"Scaling to citywide: {target_eligible:,}")
        print(f"Running {simulation_count:,} enhanced simulation paths...")
        print(f"  Enhancements: regime switching, Weibull hazard, urgency ramp, equity tracking")

    rng = random.Random(seed)

    scaled_parcels, scaled_preds = _scale_parcels_to_citywide(
        parcels, predictions, target_eligible, rng
    )

    paths = []
    for i in range(simulation_count):
        path_rng = random.Random(rng.randint(0, 10**9))
        path = simulate_agent_path(scaled_parcels, scaled_preds, scenario, path_rng)
        paths.append(path)
        if verbose and (i + 1) % 200 == 0:
            print(f"  Completed {i+1}/{simulation_count} paths")

    # Aggregate
    loan_requirements = [p.required_funding_loan for p in paths]
    grant_requirements = [p.required_funding_grant for p in paths]

    def summarize_dist(values):
        ordered = sorted(values)
        return {
            "mean": round(fmean(ordered), 2),
            "median": round(_percentile(ordered, 0.50), 2),
            "p025": round(_percentile(ordered, 0.025), 2),
            "p05": round(_percentile(ordered, 0.05), 2),
            "p25": round(_percentile(ordered, 0.25), 2),
            "p75": round(_percentile(ordered, 0.75), 2),
            "p95": round(_percentile(ordered, 0.95), 2),
            "p975": round(_percentile(ordered, 0.975), 2),
            "minimum": round(ordered[0], 2),
            "maximum": round(ordered[-1], 2),
        }

    def annual_average(field_name):
        return {
            year: fmean(getattr(p, field_name)[year] for p in paths)
            for year in YEARS
        }

    loan_summary = summarize_dist(loan_requirements)
    grant_summary = summarize_dist(grant_requirements)

    # Regime diagnostics
    regime_counts = {REGIME_BOOM: 0, REGIME_NORMAL: 0, REGIME_RECESSION: 0}
    for p in paths:
        for r in p.regime_history:
            regime_counts[r] += 1
    total_regime_years = sum(regime_counts.values())
    regime_fractions = {
        REGIME_NAMES[k]: round(v / total_regime_years, 3)
        for k, v in regime_counts.items()
    }

    # Avg home value trajectory
    avg_home_values = {
        year: fmean(p.annual_avg_home_value[year] for p in paths)
        for year in YEARS
    }

    # Fund income & capacity diagnostics
    avg_fund_income = annual_average("annual_fund_income")
    avg_capacity_used = annual_average("annual_capacity_used")
    avg_recovery_losses = annual_average("annual_recovery_losses")
    total_fund_income = sum(avg_fund_income.values())
    total_recovery_losses = sum(avg_recovery_losses.values())

    if verbose:
        print(f"\n--- Enhanced Agent Simulation Results ({scenario.name}) ---")
        print(f"{'Metric':<30} {'Loan':>15} {'Grant':>15}")
        print("-" * 60)
        print(f"{'Mean funding':<30} ${loan_summary['mean']:>14,.0f} ${grant_summary['mean']:>14,.0f}")
        print(f"{'Median':<30} ${loan_summary['median']:>14,.0f} ${grant_summary['median']:>14,.0f}")
        print(f"{'P95 (recommended)':<30} ${loan_summary['p95']:>14,.0f} ${grant_summary['p95']:>14,.0f}")
        print(f"{'95% CI':<30} ${loan_summary['p025']:>14,.0f} ${grant_summary['p025']:>14,.0f}")
        print(f"{'':30} ${loan_summary['p975']:>14,.0f} ${grant_summary['p975']:>14,.0f}")
        print(f"\nRegime distribution: {regime_fractions}")
        print(f"Avg home value 2026: ${avg_home_values[2026]:,.0f} → 2037: ${avg_home_values[2037]:,.0f}")
        print(f"Total fund investment income: ${total_fund_income:,.0f}")
        print(f"Total recovery losses (LGD): ${total_recovery_losses:,.0f}")
        print(f"Avg capacity used/year: {sum(avg_capacity_used.values())/len(YEARS):.0f} / {scenario.contractor_capacity_base}-{scenario.contractor_capacity_max}")

    return {
        "scenario": scenario.name,
        "simulation_count": simulation_count,
        "eligible_parcels": len(scaled_parcels),
        "loan_summary": loan_summary,
        "grant_summary": grant_summary,
        "loan_requirements": loan_requirements,
        "grant_requirements": grant_requirements,
        "average_outflows": annual_average("annual_outflows"),
        "average_inflows": annual_average("annual_inflows"),
        "average_originations": annual_average("annual_originations"),
        "average_repayments": annual_average("annual_repayments"),
        "regime_distribution": regime_fractions,
        "avg_home_values": {str(k): round(v, 2) for k, v in avg_home_values.items()},
        "total_fund_income": round(total_fund_income, 2),
        "total_recovery_losses": round(total_recovery_losses, 2),
        "avg_capacity_used": {str(k): round(v, 1) for k, v in avg_capacity_used.items()},
        "enhancements": [
            "housing_market_regime_switching",
            "weibull_survival_hazard",
            "deadline_urgency_ramp",
            "home_price_appreciation_equity",
            "contractor_capacity_constraints",
            "fund_investment_returns",
            "loss_given_default",
        ],
    }
