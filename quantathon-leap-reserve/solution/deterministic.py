"""Deterministic closed-form expected cash-flow model.

Computes the expected (mean) cash flows analytically using Markov transition
probabilities instead of Monte Carlo sampling. Used to cross-validate MC results
and illustrate Jensen's inequality gap.
"""

from statistics import fmean
from typing import Dict, List

from .assumptions import BASE_SCENARIO, START_YEAR, YEARS, Scenario
from .data_utils import load_quote_history, load_system_materials
from .markov import _build_transition_matrix


def _weighted_schedule_counts(total: int, weights: List[float]) -> Dict[int, int]:
    """Allocate total lines across YEARS using weights (same logic as model.py)."""
    import math
    total_weight = sum(weights)
    raw = [total * w / total_weight for w in weights]
    counts = [math.floor(v) for v in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return {year: counts[pos] for pos, year in enumerate(YEARS)}


def _annual_trigger_probability(scenario: Scenario, year: int) -> float:
    steps = year - START_YEAR
    p = scenario.annual_trigger_start * ((1.0 + scenario.annual_trigger_growth) ** steps)
    return min(p, scenario.annual_trigger_cap)


def run_deterministic_model(scenario: Scenario = None) -> Dict:
    """Compute expected cash flows deterministically.

    Instead of simulating random paths, we track the *expected* number of loans
    in each state (Active, Inherited, Repaid) using Markov transition probabilities.
    """
    if scenario is None:
        scenario = BASE_SCENARIO

    materials = load_system_materials()
    quotes = load_quote_history()
    eligible = materials["eligible_lines"]
    mean_loan = fmean(quotes["loan_amounts"])
    schedule_counts = _weighted_schedule_counts(eligible, scenario.schedule_weights)

    P = _build_transition_matrix(scenario)
    # Transition probs: Active->Active, Active->Inherited, Active->Repaid
    p_aa, p_ai, p_ar = P[0][0], P[0][1], P[0][2]
    p_ia, p_ii, p_ir = P[1][0], P[1][1], P[1][2]

    # Track remaining at-risk population (expected values)
    remaining = dict(schedule_counts)

    # Track loan cohorts: for each origination year, track (active, inherited, repaid) counts
    # Key: origination year, Value: [active_count, inherited_count, repaid_count]
    cohorts: Dict[int, List[float]] = {}

    annual_outflows: Dict[int, float] = {}
    annual_inflows: Dict[int, float] = {}
    annual_originations: Dict[int, float] = {}
    annual_repayments: Dict[int, float] = {}

    for year in YEARS:
        trigger_prob = _annual_trigger_probability(scenario, year)
        inflation_factor = (1.0 + scenario.annual_cost_inflation) ** (year - START_YEAR)
        inflated_mean_loan = mean_loan * inflation_factor

        # New originations: sum expected originations across all at-risk cohorts
        year_originations = 0.0
        for sched_year in YEARS:
            if sched_year <= year:
                continue
            at_risk = remaining[sched_year]
            if at_risk <= 0:
                continue
            expected_new = at_risk * trigger_prob
            year_originations += expected_new
            remaining[sched_year] -= expected_new

        annual_originations[year] = year_originations
        annual_outflows[year] = year_originations * inflated_mean_loan

        # Add new cohort
        if year_originations > 0:
            cohorts[year] = [year_originations, 0.0, 0.0]  # all start Active

        # Process repayments for existing cohorts (originated before this year)
        year_inflows = 0.0
        year_repayments = 0.0
        for orig_year, state in list(cohorts.items()):
            if orig_year >= year:
                continue
            active, inherited, repaid = state
            orig_inflation = (1.0 + scenario.annual_cost_inflation) ** (orig_year - START_YEAR)
            cohort_loan_amount = mean_loan * orig_inflation

            # Transitions from Active
            new_repaid_from_active = active * p_ar
            new_inherited_from_active = active * p_ai
            new_active = active * p_aa

            # Transitions from Inherited
            new_repaid_from_inherited = inherited * p_ir
            new_inherited = inherited * p_ii + new_inherited_from_active

            # Total new repayments
            new_repaid = new_repaid_from_active + new_repaid_from_inherited

            year_inflows += new_repaid * cohort_loan_amount
            year_repayments += new_repaid

            cohorts[orig_year] = [new_active, new_inherited, repaid + new_repaid]

        annual_inflows[year] = year_inflows
        annual_repayments[year] = year_repayments

    # Compute max cumulative deficit
    cumulative = 0.0
    max_deficit = 0.0
    cumulative_by_year = {}
    for year in YEARS:
        cumulative += annual_outflows[year] - annual_inflows[year]
        cumulative_by_year[year] = cumulative
        max_deficit = max(max_deficit, cumulative)

    total_outflows = sum(annual_outflows.values())
    total_inflows = sum(annual_inflows.values())

    return {
        "deterministic_funding_requirement": round(max_deficit, 2),
        "total_expected_outflows": round(total_outflows, 2),
        "total_expected_inflows": round(total_inflows, 2),
        "recovery_ratio": round(total_inflows / total_outflows, 4) if total_outflows > 0 else 0.0,
        "annual_outflows": {y: round(v, 2) for y, v in annual_outflows.items()},
        "annual_inflows": {y: round(v, 2) for y, v in annual_inflows.items()},
        "annual_originations": {y: round(v, 2) for y, v in annual_originations.items()},
        "cumulative_deficit": {y: round(v, 2) for y, v in cumulative_by_year.items()},
    }


def cross_validate(mc_mean: float, scenario: Scenario = None) -> Dict:
    """Compare deterministic result against MC mean and compute gap."""
    det = run_deterministic_model(scenario)
    det_funding = det["deterministic_funding_requirement"]
    gap = det_funding - mc_mean
    gap_pct = (gap / mc_mean) * 100 if mc_mean != 0 else 0.0

    return {
        "deterministic_funding": round(det_funding, 2),
        "mc_mean_funding": round(mc_mean, 2),
        "absolute_gap": round(gap, 2),
        "percentage_gap": round(gap_pct, 2),
        "detail": det,
    }
