"""Break-even analysis for LEAP uptake rates.

Sweeps annual trigger start rate to find the value where p95 funding
crosses a given funding level, identifying the break-even uptake rate.
"""

from dataclasses import replace
from typing import Dict, List

from .assumptions import BASE_SCENARIO
from .model import run_simulation

BREAKEVEN_SIM_COUNT = 500
BREAKEVEN_SEED = 20260402


def run_breakeven_analysis(
    scenario=None,
    simulation_count: int = BREAKEVEN_SIM_COUNT,
    funding_level: float = None,
) -> Dict:
    """Sweep trigger start rate to find break-even point.

    Args:
        scenario: Base scenario to vary. Defaults to BASE_SCENARIO.
        simulation_count: MC paths per sweep step.
        funding_level: Threshold to compare p95 against (e.g. base p95).

    Returns:
        Dict with sweep curve data and interpolated break-even rate.
    """
    if scenario is None:
        scenario = BASE_SCENARIO

    # Sweep from 0.0005 to 0.005 in 20 steps
    rates = [0.0005 + i * (0.005 - 0.0005) / 19 for i in range(20)]
    sweep_data: List[Dict] = []

    for rate in rates:
        # Scale cap proportionally to maintain same start/cap ratio
        ratio = scenario.annual_trigger_cap / scenario.annual_trigger_start
        cap = rate * ratio
        variant = replace(
            scenario,
            name=f"breakeven_{rate:.4f}",
            annual_trigger_start=rate,
            annual_trigger_cap=cap,
        )
        sim = run_simulation(variant, simulation_count=simulation_count, seed=BREAKEVEN_SEED)
        sweep_data.append({
            "trigger_rate": round(rate, 5),
            "trigger_cap": round(cap, 5),
            "p95": round(sim["loan_summary"]["p95"], 2),
            "mean": round(sim["loan_summary"]["mean"], 2),
        })

    # Find break-even via linear interpolation
    breakeven_rate = None
    if funding_level is not None:
        for i in range(len(sweep_data) - 1):
            p95_a = sweep_data[i]["p95"]
            p95_b = sweep_data[i + 1]["p95"]
            rate_a = sweep_data[i]["trigger_rate"]
            rate_b = sweep_data[i + 1]["trigger_rate"]

            if p95_a <= funding_level <= p95_b or p95_b <= funding_level <= p95_a:
                # Linear interpolation
                if p95_b != p95_a:
                    t = (funding_level - p95_a) / (p95_b - p95_a)
                    breakeven_rate = rate_a + t * (rate_b - rate_a)
                else:
                    breakeven_rate = rate_a
                break

    return {
        "sweep_data": sweep_data,
        "funding_level": round(funding_level, 2) if funding_level else None,
        "breakeven_rate": round(breakeven_rate, 5) if breakeven_rate else None,
        "simulation_count_per_step": simulation_count,
        "steps": len(sweep_data),
    }
