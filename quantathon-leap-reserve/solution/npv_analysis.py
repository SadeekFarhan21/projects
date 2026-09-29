"""Net Present Value analysis and extended recovery horizon.

Discounts LEAP cash flows at municipal borrowing rates and projects
post-2037 repayments using the Markov chain on the terminal portfolio.
"""

from statistics import fmean
from typing import Dict, List

from .assumptions import BASE_SCENARIO, YEARS, Scenario
from .markov import cumulative_repayment_schedule


def npv_of_cashflows(
    annual_outflows: Dict[int, float],
    annual_inflows: Dict[int, float],
    discount_rate: float,
    base_year: int = 2026,
) -> Dict[str, float]:
    """Compute NPV of cash flows at a given discount rate.

    Outflows are costs (negative), inflows are recoveries (positive).
    Returns NPV of net cost, PV of outflows, PV of inflows.
    """
    pv_outflows = 0.0
    pv_inflows = 0.0
    for year in sorted(set(annual_outflows.keys()) | set(annual_inflows.keys())):
        t = year - base_year
        df = 1.0 / ((1.0 + discount_rate) ** t)
        pv_outflows += annual_outflows.get(year, 0.0) * df
        pv_inflows += annual_inflows.get(year, 0.0) * df

    return {
        "pv_outflows": round(pv_outflows, 2),
        "pv_inflows": round(pv_inflows, 2),
        "npv_net_cost": round(pv_outflows - pv_inflows, 2),
        "pv_recovery_ratio": round(pv_inflows / pv_outflows, 4) if pv_outflows > 0 else 0.0,
    }


def extended_recovery_projection(
    terminal_outstanding_principal: float,
    scenario: Scenario = None,
    max_extra_years: int = 30,
) -> List[Dict]:
    """Project post-2037 cumulative recovery using Markov chain.

    Assumes the terminal portfolio is a mix of Active and Inherited loans.
    For simplicity, we model the average as starting from Active state
    (conservative since some are Inherited and repay faster due to heir sale rate).
    """
    if scenario is None:
        scenario = BASE_SCENARIO

    schedule = cumulative_repayment_schedule(scenario, max_years=max_extra_years)
    projection = []
    for entry in schedule:
        recovered = terminal_outstanding_principal * entry["p_repaid"]
        projection.append({
            "years_after_2037": entry["year"],
            "calendar_year": 2037 + entry["year"],
            "cumulative_recovery_fraction": round(entry["p_repaid"], 4),
            "cumulative_recovery_dollars": round(recovered, 2),
            "still_outstanding": round(terminal_outstanding_principal * (1.0 - entry["p_repaid"]), 2),
        })
    return projection


def find_recovery_milestones(
    total_deployed: float,
    total_recovered_by_2037: float,
    terminal_outstanding: float,
    scenario: Scenario = None,
) -> Dict[str, object]:
    """Find years to reach 80% and 90% cumulative recovery (including post-2037)."""
    if scenario is None:
        scenario = BASE_SCENARIO

    recovery_at_2037 = total_recovered_by_2037 / total_deployed if total_deployed > 0 else 0.0
    schedule = cumulative_repayment_schedule(scenario, max_years=50)

    milestones = {}
    for target_label, target_frac in [("80%", 0.80), ("90%", 0.90)]:
        if recovery_at_2037 >= target_frac:
            milestones[target_label] = {"year": 2037, "years_after_2037": 0}
            continue
        # Need additional recovery from terminal portfolio
        needed_from_terminal = (target_frac * total_deployed) - total_recovered_by_2037
        needed_frac_of_terminal = needed_from_terminal / terminal_outstanding if terminal_outstanding > 0 else float("inf")

        found = False
        for entry in schedule:
            if entry["p_repaid"] >= needed_frac_of_terminal:
                milestones[target_label] = {
                    "year": 2037 + entry["year"],
                    "years_after_2037": entry["year"],
                }
                found = True
                break
        if not found:
            milestones[target_label] = {"year": ">2087", "years_after_2037": ">50"}

    return milestones


def run_npv_analysis(
    average_outflows: Dict[int, float],
    average_inflows: Dict[int, float],
    total_deployed: float,
    total_recovered_by_2037: float,
    scenario: Scenario = None,
) -> Dict:
    """Run the full NPV and extended horizon analysis.

    Args:
        average_outflows: Mean annual outflows from MC.
        average_inflows: Mean annual inflows from MC.
        total_deployed: Total capital deployed (sum of outflows).
        total_recovered_by_2037: Total recovered by end of horizon.
        scenario: Scenario to use for projections.
    """
    if scenario is None:
        scenario = BASE_SCENARIO

    terminal_outstanding = total_deployed - total_recovered_by_2037

    # NPV at multiple discount rates
    discount_results = {}
    for rate in [0.01, 0.02, 0.03, 0.04, 0.05]:
        label = f"{rate:.0%}"
        discount_results[label] = npv_of_cashflows(average_outflows, average_inflows, rate)
        # Also compute NPV for grant (zero inflows)
        grant_npv = npv_of_cashflows(average_outflows, {y: 0.0 for y in YEARS}, rate)
        discount_results[label]["grant_npv_net_cost"] = grant_npv["npv_net_cost"]
        discount_results[label]["loan_savings_pv"] = round(
            grant_npv["npv_net_cost"] - discount_results[label]["npv_net_cost"], 2
        )

    # Extended recovery projection
    projection = extended_recovery_projection(terminal_outstanding, scenario)

    # Recovery milestones
    milestones = find_recovery_milestones(
        total_deployed, total_recovered_by_2037, terminal_outstanding, scenario
    )

    # Discount rate robustness: % change in NPV net cost from base (3%)
    base_npv = discount_results["3%"]["npv_net_cost"]
    robustness = {}
    for label, data in discount_results.items():
        pct_change = ((data["npv_net_cost"] - base_npv) / base_npv * 100) if base_npv != 0 else 0.0
        robustness[label] = {
            "npv_net_cost": data["npv_net_cost"],
            "pct_change_from_base": round(pct_change, 2),
        }

    return {
        "discount_rate_analysis": discount_results,
        "robustness": robustness,
        "terminal_outstanding": round(terminal_outstanding, 2),
        "recovery_at_2037": round(total_recovered_by_2037 / total_deployed, 4) if total_deployed > 0 else 0.0,
        "extended_projection": projection,
        "recovery_milestones": milestones,
    }
