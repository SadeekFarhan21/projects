import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Dict, List, Tuple

from .assumptions import END_YEAR, START_YEAR, YEARS, Scenario
from .data_utils import load_quote_history, load_system_materials


@dataclass
class PathResult:
    annual_outflows: Dict[int, float]
    annual_inflows: Dict[int, float]
    annual_originations: Dict[int, int]
    annual_repayments: Dict[int, int]
    required_funding_loan: float
    required_funding_grant: float
    loan_balances_if_funded: Dict[int, float]
    grant_balances_if_funded: Dict[int, float]


@dataclass
class PortfolioInputs:
    eligible_lines: int
    schedule_counts: Dict[int, int]
    loan_amount_history: List[float]


def _weighted_schedule_counts(total: int, weights: List[float]) -> Dict[int, int]:
    if len(weights) != len(YEARS):
        raise ValueError("Schedule weights must match model years.")
    total_weight = sum(weights)
    raw = [total * weight / total_weight for weight in weights]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return {year: counts[position] for position, year in enumerate(YEARS)}


def build_portfolio_inputs(scenario: Scenario) -> PortfolioInputs:
    materials = load_system_materials()
    quotes = load_quote_history()
    return PortfolioInputs(
        eligible_lines=materials["eligible_lines"],
        schedule_counts=_weighted_schedule_counts(materials["eligible_lines"], scenario.schedule_weights),
        loan_amount_history=quotes["loan_amounts"],
    )


def _annual_trigger_probability(scenario: Scenario, year: int) -> float:
    steps = year - START_YEAR
    probability = scenario.annual_trigger_start * ((1.0 + scenario.annual_trigger_growth) ** steps)
    return min(probability, scenario.annual_trigger_cap)


def _bootstrap_loan_amount(rng: random.Random, history: List[float], year: int, inflation: float) -> float:
    base = history[rng.randrange(len(history))]
    years_from_start = year - START_YEAR
    return round(base * ((1.0 + inflation) ** years_from_start), 2)


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def simulate_path(portfolio: PortfolioInputs, scenario: Scenario, rng: random.Random) -> PathResult:
    remaining_by_schedule = dict(portfolio.schedule_counts)
    loans = []

    annual_outflows = {year: 0.0 for year in YEARS}
    annual_inflows = {year: 0.0 for year in YEARS}
    annual_originations = {year: 0 for year in YEARS}
    annual_repayments = {year: 0 for year in YEARS}

    for year in YEARS:
        trigger_probability = _annual_trigger_probability(scenario, year)
        at_risk_years = [schedule_year for schedule_year in YEARS if schedule_year > year]
        for schedule_year in at_risk_years:
            cohort_remaining = remaining_by_schedule[schedule_year]
            if cohort_remaining <= 0:
                continue
            originations = sum(1 for _ in range(cohort_remaining) if rng.random() < trigger_probability)
            if originations:
                remaining_by_schedule[schedule_year] -= originations
                annual_originations[year] += originations
                for _ in range(originations):
                    principal = _bootstrap_loan_amount(
                        rng=rng,
                        history=portfolio.loan_amount_history,
                        year=year,
                        inflation=scenario.annual_cost_inflation,
                    )
                    annual_outflows[year] += principal
                    loans.append({"principal": principal, "status": "active", "origination_year": year})

        for loan in loans:
            if loan["status"] == "repaid" or loan["origination_year"] >= year:
                continue
            if loan["status"] == "active":
                sale_probability = scenario.annual_sale_probability
                equity_probability = scenario.annual_equity_probability
                inheritance_probability = scenario.annual_inheritance_probability
                draw = rng.random()
                if draw < sale_probability:
                    annual_inflows[year] += loan["principal"]
                    annual_repayments[year] += 1
                    loan["status"] = "repaid"
                    continue
                draw -= sale_probability
                if draw < equity_probability:
                    annual_inflows[year] += loan["principal"]
                    annual_repayments[year] += 1
                    loan["status"] = "repaid"
                    continue
                draw -= equity_probability
                if draw < inheritance_probability:
                    loan["status"] = "inherited"
                    continue
            if loan["status"] == "inherited":
                draw = rng.random()
                if draw < scenario.inherited_sale_probability:
                    annual_inflows[year] += loan["principal"]
                    annual_repayments[year] += 1
                    loan["status"] = "repaid"
                    continue
                draw -= scenario.inherited_sale_probability
                if draw < scenario.inherited_equity_probability:
                    annual_inflows[year] += loan["principal"]
                    annual_repayments[year] += 1
                    loan["status"] = "repaid"

    cumulative_loan = 0.0
    cumulative_grant = 0.0
    max_deficit_loan = 0.0
    max_deficit_grant = 0.0
    for year in YEARS:
        cumulative_loan += annual_outflows[year] - annual_inflows[year]
        cumulative_grant += annual_outflows[year]
        max_deficit_loan = max(max_deficit_loan, cumulative_loan)
        max_deficit_grant = max(max_deficit_grant, cumulative_grant)

    loan_balances_if_funded = {}
    grant_balances_if_funded = {}
    loan_funding = max_deficit_loan
    grant_funding = max_deficit_grant
    running_loan_balance = loan_funding
    running_grant_balance = grant_funding
    for year in YEARS:
        running_loan_balance += annual_inflows[year] - annual_outflows[year]
        running_grant_balance -= annual_outflows[year]
        loan_balances_if_funded[year] = running_loan_balance
        grant_balances_if_funded[year] = running_grant_balance

    return PathResult(
        annual_outflows=annual_outflows,
        annual_inflows=annual_inflows,
        annual_originations=annual_originations,
        annual_repayments=annual_repayments,
        required_funding_loan=round(max_deficit_loan, 2),
        required_funding_grant=round(max_deficit_grant, 2),
        loan_balances_if_funded=loan_balances_if_funded,
        grant_balances_if_funded=grant_balances_if_funded,
    )


def run_simulation(scenario: Scenario, simulation_count: int, seed: int) -> Dict[str, object]:
    portfolio = build_portfolio_inputs(scenario)
    rng = random.Random(seed)
    paths = [simulate_path(portfolio=portfolio, scenario=scenario, rng=random.Random(rng.randint(0, 10**9))) for _ in range(simulation_count)]

    loan_requirements = [path.required_funding_loan for path in paths]
    grant_requirements = [path.required_funding_grant for path in paths]

    def annual_average(field_name: str) -> Dict[int, float]:
        return {
            year: fmean(getattr(path, field_name)[year] for path in paths)
            for year in YEARS
        }

    def annual_band(field_name: str, quantile: float) -> Dict[int, float]:
        return {
            year: _percentile([getattr(path, field_name)[year] for path in paths], quantile)
            for year in YEARS
        }

    return {
        "scenario": scenario,
        "portfolio": portfolio,
        "paths": paths,
        "loan_requirements": loan_requirements,
        "grant_requirements": grant_requirements,
        "loan_summary": summarize_requirement_distribution(loan_requirements),
        "grant_summary": summarize_requirement_distribution(grant_requirements),
        "average_outflows": annual_average("annual_outflows"),
        "average_inflows": annual_average("annual_inflows"),
        "average_originations": annual_average("annual_originations"),
        "average_repayments": annual_average("annual_repayments"),
        "loan_balance_p05": annual_band("loan_balances_if_funded", 0.05),
        "loan_balance_p50": annual_band("loan_balances_if_funded", 0.50),
        "loan_balance_p95": annual_band("loan_balances_if_funded", 0.95),
        "grant_balance_p05": annual_band("grant_balances_if_funded", 0.05),
        "grant_balance_p50": annual_band("grant_balances_if_funded", 0.50),
        "grant_balance_p95": annual_band("grant_balances_if_funded", 0.95),
    }


def summarize_requirement_distribution(values: List[float]) -> Dict[str, float]:
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
