"""One-at-a-time (OAT) tornado sensitivity analysis.

Varies each of 8 key parameters independently between optimistic and
conservative values, running 1,000-path MC simulations per variant.
Records p95 funding swing per parameter and ranks by impact magnitude.
"""

from dataclasses import replace
from typing import Dict, List, Tuple

from .assumptions import BASE_SCENARIO, CONSERVATIVE_SCENARIO, OPTIMISTIC_SCENARIO, Scenario
from .model import run_simulation


# Parameter definitions: (field_name, display_name, optimistic_value, conservative_value)
PARAMETER_RANGES: List[Tuple[str, str, object, object]] = [
    ("annual_trigger_start", "LEAP uptake start rate", OPTIMISTIC_SCENARIO.annual_trigger_start, CONSERVATIVE_SCENARIO.annual_trigger_start),
    ("annual_trigger_cap", "LEAP uptake cap", OPTIMISTIC_SCENARIO.annual_trigger_cap, CONSERVATIVE_SCENARIO.annual_trigger_cap),
    ("annual_sale_probability", "Sale probability", OPTIMISTIC_SCENARIO.annual_sale_probability, CONSERVATIVE_SCENARIO.annual_sale_probability),
    ("annual_equity_probability", "Equity extraction prob.", OPTIMISTIC_SCENARIO.annual_equity_probability, CONSERVATIVE_SCENARIO.annual_equity_probability),
    ("annual_inheritance_probability", "Inheritance probability", OPTIMISTIC_SCENARIO.annual_inheritance_probability, CONSERVATIVE_SCENARIO.annual_inheritance_probability),
    ("annual_cost_inflation", "Cost inflation", OPTIMISTIC_SCENARIO.annual_cost_inflation, CONSERVATIVE_SCENARIO.annual_cost_inflation),
    ("inherited_sale_probability", "Inherited sale prob.", OPTIMISTIC_SCENARIO.inherited_sale_probability, CONSERVATIVE_SCENARIO.inherited_sale_probability),
    ("schedule_weights", "Schedule weighting", OPTIMISTIC_SCENARIO.schedule_weights, CONSERVATIVE_SCENARIO.schedule_weights),
]

TORNADO_SIM_COUNT = 1000
TORNADO_SEED = 20260328


def run_tornado_analysis(base_p95: float = None) -> Dict:
    """Run OAT tornado sensitivity analysis.

    For each parameter, creates two modified scenarios (optimistic and conservative
    value) while holding all other parameters at base. Runs 1,000-path MC per variant.

    Args:
        base_p95: Pre-computed base-case p95 for reference. If None, runs base case.

    Returns:
        Dict with ranked parameter sensitivities and swing magnitudes.
    """
    if base_p95 is None:
        base_result = run_simulation(BASE_SCENARIO, simulation_count=TORNADO_SIM_COUNT, seed=TORNADO_SEED)
        base_p95 = base_result["loan_summary"]["p95"]

    results = []

    for field_name, display_name, opt_value, cons_value in PARAMETER_RANGES:
        # Create optimistic variant
        opt_scenario = replace(BASE_SCENARIO, name=f"tornado_opt_{field_name}", **{field_name: opt_value})
        opt_result = run_simulation(opt_scenario, simulation_count=TORNADO_SIM_COUNT, seed=TORNADO_SEED)
        opt_p95 = opt_result["loan_summary"]["p95"]

        # Create conservative variant
        cons_scenario = replace(BASE_SCENARIO, name=f"tornado_cons_{field_name}", **{field_name: cons_value})
        cons_result = run_simulation(cons_scenario, simulation_count=TORNADO_SIM_COUNT, seed=TORNADO_SEED)
        cons_p95 = cons_result["loan_summary"]["p95"]

        swing = cons_p95 - opt_p95
        results.append({
            "parameter": display_name,
            "field": field_name,
            "optimistic_p95": round(opt_p95, 2),
            "conservative_p95": round(cons_p95, 2),
            "swing": round(swing, 2),
            "abs_swing": round(abs(swing), 2),
            "optimistic_value": opt_value if not isinstance(opt_value, list) else "back-loaded",
            "conservative_value": cons_value if not isinstance(cons_value, list) else "front-loaded",
        })

    # Sort by absolute swing (descending)
    results.sort(key=lambda x: x["abs_swing"], reverse=True)

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {
        "base_p95": round(base_p95, 2),
        "simulation_count_per_variant": TORNADO_SIM_COUNT,
        "parameters": results,
    }
