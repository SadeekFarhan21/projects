"""Stress-test scenarios for LEAP funding model.

Defines three named extreme scenarios and runs MC simulations to
quantify tail risk under adverse conditions.
"""

from dataclasses import replace
from typing import Dict

from .assumptions import BASE_SCENARIO, Scenario
from .model import run_simulation

STRESS_SIM_COUNT = 1000
STRESS_SEED = 20260401


def _housing_crash_scenario() -> Scenario:
    """2029 Housing Crash: severe market downturn reduces repayments."""
    return replace(
        BASE_SCENARIO,
        name="stress_housing_crash",
        annual_sale_probability=BASE_SCENARIO.annual_sale_probability * 0.60,
        annual_equity_probability=BASE_SCENARIO.annual_equity_probability * 0.50,
        # Lock into recession-like regime
        regime_transition_matrix=[
            [0.20, 0.30, 0.50],  # From Boom -> likely recession
            [0.05, 0.30, 0.65],  # From Normal -> likely recession
            [0.02, 0.18, 0.80],  # From Recession -> stays recession
        ],
        regime_sale_multipliers=[1.10, 0.80, 0.40],
        regime_equity_multipliers=[1.20, 0.70, 0.25],
    )


def _surge_uptake_scenario() -> Scenario:
    """Surge Uptake: much higher LEAP demand than expected."""
    return replace(
        BASE_SCENARIO,
        name="stress_surge_uptake",
        annual_trigger_start=BASE_SCENARIO.annual_trigger_start * 2.0,
        annual_trigger_cap=BASE_SCENARIO.annual_trigger_cap * 1.5,
        annual_cost_inflation=BASE_SCENARIO.annual_cost_inflation + 0.01,
        contractor_capacity_base=int(BASE_SCENARIO.contractor_capacity_base * 0.8),
    )


def _perfect_storm_scenario() -> Scenario:
    """Perfect Storm: combines housing crash and surge uptake."""
    return replace(
        BASE_SCENARIO,
        name="stress_perfect_storm",
        # Surge uptake
        annual_trigger_start=BASE_SCENARIO.annual_trigger_start * 2.0,
        annual_trigger_cap=BASE_SCENARIO.annual_trigger_cap * 1.5,
        annual_cost_inflation=BASE_SCENARIO.annual_cost_inflation + 0.01,
        contractor_capacity_base=int(BASE_SCENARIO.contractor_capacity_base * 0.8),
        # Housing crash
        annual_sale_probability=BASE_SCENARIO.annual_sale_probability * 0.60,
        annual_equity_probability=BASE_SCENARIO.annual_equity_probability * 0.50,
        regime_transition_matrix=[
            [0.20, 0.30, 0.50],
            [0.05, 0.30, 0.65],
            [0.02, 0.18, 0.80],
        ],
        regime_sale_multipliers=[1.10, 0.80, 0.40],
        regime_equity_multipliers=[1.20, 0.70, 0.25],
    )


STRESS_SCENARIOS = {
    "2029 Housing Crash": _housing_crash_scenario,
    "Surge Uptake": _surge_uptake_scenario,
    "Perfect Storm": _perfect_storm_scenario,
}


def run_all_stress_tests(
    base_p95: float,
    simulation_count: int = STRESS_SIM_COUNT,
) -> Dict:
    """Run all stress-test scenarios and compare to base.

    Args:
        base_p95: Base-case p95 for comparison.
        simulation_count: Number of MC paths per scenario.

    Returns:
        Dict with per-scenario results and comparison metrics.
    """
    results = []
    for name, scenario_fn in STRESS_SCENARIOS.items():
        scenario = scenario_fn()
        sim = run_simulation(scenario, simulation_count=simulation_count, seed=STRESS_SEED)
        p95 = sim["loan_summary"]["p95"]
        mean = sim["loan_summary"]["mean"]
        results.append({
            "scenario_name": name,
            "p95": round(p95, 2),
            "mean": round(mean, 2),
            "p95_increase_pct": round((p95 - base_p95) / base_p95 * 100, 1),
            "mean_increase_pct": round((mean - base_p95) / base_p95 * 100, 1),
            "loan_summary": sim["loan_summary"],
        })

    return {
        "base_p95": round(base_p95, 2),
        "simulation_count": simulation_count,
        "scenarios": results,
    }
