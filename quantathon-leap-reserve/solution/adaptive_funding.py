"""Adaptive (staged) funding strategy optimization.

Grid-searches over initial reserve, top-up amount, and top-up threshold
to find the cheapest strategy that achieves >= 95% solvency using
existing MC paths.
"""

from typing import Dict, List

from .assumptions import YEARS
from .model import PathResult


def _evaluate_strategy(
    paths: List[PathResult],
    initial_reserve: float,
    topup_amount: float,
    topup_threshold: int,
    topup_year: int = 2029,
) -> Dict:
    """Evaluate a staged funding strategy against MC paths.

    For each path, start with initial_reserve. If cumulative originations
    by topup_year exceed topup_threshold, inject topup_amount.
    Track whether the fund stays solvent (balance >= 0) through 2037.

    Returns solvency rate and expected total capital deployed.
    """
    solvent_count = 0
    topup_triggered_count = 0

    for path in paths:
        balance = initial_reserve
        cumulative_originations = 0
        topup_injected = False

        for year in YEARS:
            # Check topup trigger at topup_year
            cumulative_originations += path.annual_originations[year]
            if year == topup_year and cumulative_originations > topup_threshold:
                balance += topup_amount
                topup_injected = True

            # Apply cash flows
            balance += path.annual_inflows[year] - path.annual_outflows[year]

        if topup_injected:
            topup_triggered_count += 1

        # Check if balance was ever negative
        running = initial_reserve
        cum_orig = 0
        ever_negative = False
        for year in YEARS:
            cum_orig += path.annual_originations[year]
            if year == topup_year and cum_orig > topup_threshold:
                running += topup_amount
            running += path.annual_inflows[year] - path.annual_outflows[year]
            if running < 0:
                ever_negative = True
                break

        if not ever_negative:
            solvent_count += 1

    n = len(paths)
    topup_prob = topup_triggered_count / n if n else 0
    solvency_rate = solvent_count / n if n else 0
    expected_total = initial_reserve + topup_prob * topup_amount

    return {
        "solvency_rate": round(solvency_rate, 4),
        "topup_probability": round(topup_prob, 4),
        "expected_total_capital": round(expected_total, 2),
        "initial_reserve": round(initial_reserve, 2),
        "topup_amount": round(topup_amount, 2),
        "topup_threshold": topup_threshold,
    }


def run_adaptive_analysis(
    base_p95: float,
    paths: List[PathResult],
    simulation_count: int = 500,
) -> Dict:
    """Grid search for optimal adaptive funding strategy.

    Args:
        base_p95: Base-case p95 for reference.
        paths: MC path results to evaluate against.
        simulation_count: Max paths to use (limits compute).

    Returns:
        Dict with grid results and optimal strategy.
    """
    # Use subset of paths for efficiency
    eval_paths = paths[:simulation_count]

    # Grid definition
    initial_reserves = [
        base_p95 * f for f in [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ]
    topup_amounts = [
        base_p95 * f for f in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    ]
    topup_thresholds = [3, 5, 8, 10, 15, 20, 30]

    grid_results: List[Dict] = []
    best_strategy = None
    best_expected_total = float("inf")

    for reserve in initial_reserves:
        for topup in topup_amounts:
            for threshold in topup_thresholds:
                result = _evaluate_strategy(
                    eval_paths, reserve, topup, threshold
                )
                grid_results.append(result)

                # Find optimal: lowest expected total achieving >= 95% solvency
                if (
                    result["solvency_rate"] >= 0.95
                    and result["expected_total_capital"] < best_expected_total
                ):
                    best_expected_total = result["expected_total_capital"]
                    best_strategy = result

    return {
        "base_p95": round(base_p95, 2),
        "paths_evaluated": len(eval_paths),
        "grid_size": len(grid_results),
        "optimal_strategy": best_strategy,
        "savings_vs_base": round(base_p95 - best_expected_total, 2) if best_strategy else 0,
        "savings_pct": round(
            (base_p95 - best_expected_total) / base_p95 * 100, 1
        ) if best_strategy and base_p95 > 0 else 0,
    }
