"""Value of Information (VOI) analysis.

Decomposes the overall funding uncertainty into per-parameter contributions
using tornado swing magnitudes and bootstrap CI widths, then ranks parameters
by the dollar value of resolving each one's uncertainty.
"""

from typing import Dict, List


def run_voi_analysis(
    tornado_results: Dict,
    bootstrap_results: Dict,
    base_p95: float,
) -> Dict:
    """Compute dollar-valued VOI for each tornado parameter.

    Args:
        tornado_results: Output from run_tornado_analysis().
        bootstrap_results: Output from run_bootstrap_analysis().
        base_p95: Base-case p95 funding requirement.

    Returns:
        Dict with ranked parameters and dollar VOI values.
    """
    params = tornado_results["parameters"]
    ci_width_p95 = bootstrap_results["loan"]["p95"]["ci_width"]

    # Variance decomposition: share_i = swing_i^2 / sum(swing_j^2)
    swings_sq = [p["swing"] ** 2 for p in params]
    total_sq = sum(swings_sq)

    if total_sq == 0:
        return {
            "parameters": [],
            "total_ci_width_p95": ci_width_p95,
            "total_voi_dollars": 0,
        }

    ranked: List[Dict] = []
    total_voi = 0.0
    for i, p in enumerate(params):
        share = swings_sq[i] / total_sq
        dollar_voi = share * ci_width_p95
        total_voi += dollar_voi
        ranked.append({
            "parameter": p["parameter"],
            "field": p["field"],
            "swing": p["swing"],
            "variance_share": round(share, 4),
            "dollar_voi": round(dollar_voi, 2),
            "rank": i + 1,
        })

    return {
        "parameters": ranked,
        "total_ci_width_p95": round(ci_width_p95, 2),
        "total_voi_dollars": round(total_voi, 2),
        "base_p95": round(base_p95, 2),
    }
