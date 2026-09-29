"""Geographic hotspot analysis on the v4 test set.

Groups Columbus v4 test parcels by zip code and computes composite risk
scores to identify neighborhoods with the highest LEAP funding exposure.
The v4 dataset is the held-out test set; all model training uses v2 data.
"""

from collections import defaultdict
from typing import Dict, List

from .ml_models import load_v4_parcels


def run_geographic_analysis() -> Dict:
    """Analyze LEAP risk by zip code using v4 test set parcels.

    Returns:
        Dict with per-zip metrics and composite risk scores.
    """
    parcels = load_v4_parcels()

    # Group by zip code
    by_zip: Dict[int, List[Dict]] = defaultdict(list)
    for p in parcels:
        by_zip[p["zip_code"]].append(p)

    zip_metrics = []
    for zip_code, group in sorted(by_zip.items()):
        count = len(group)
        eligible_count = sum(
            1 for p in group if p["line_material"] in ("lead", "galvanized")
        )
        avg_lead_risk = sum(p["lead_risk"] for p in group) / count
        avg_assessed_value = sum(p["assessed_value"] for p in group) / count
        avg_replacement_cost = sum(
            p["estimated_replacement_cost"] for p in group
        ) / count
        owner_occupied_count = sum(p["owner_occupied"] for p in group)
        owner_occupied_rate = owner_occupied_count / count

        # Material breakdown
        materials: Dict[str, int] = defaultdict(int)
        for p in group:
            materials[p["line_material"]] += 1

        zip_metrics.append({
            "zip_code": zip_code,
            "parcel_count": count,
            "eligible_count": eligible_count,
            "eligible_pct": round(eligible_count / count, 4) if count else 0,
            "avg_lead_risk": round(avg_lead_risk, 4),
            "avg_assessed_value": round(avg_assessed_value, 2),
            "avg_replacement_cost": round(avg_replacement_cost, 2),
            "owner_occupied_rate": round(owner_occupied_rate, 4),
            "renter_rate": round(1.0 - owner_occupied_rate, 4),
            "material_breakdown": dict(materials),
        })

    # Composite risk score: weighted combination, min-max normalized
    if not zip_metrics:
        return {"zip_metrics": [], "total_parcels": 0}

    # Extract raw component values for normalization
    lead_risks = [z["avg_lead_risk"] for z in zip_metrics]
    values = [z["avg_assessed_value"] for z in zip_metrics]
    eligible_pcts = [z["eligible_pct"] for z in zip_metrics]
    renter_rates = [z["renter_rate"] for z in zip_metrics]
    costs = [z["avg_replacement_cost"] for z in zip_metrics]

    def _min_max(vals):
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi != lo else 1.0
        return [(v - lo) / span for v in vals]

    norm_lead = _min_max(lead_risks)
    # Inverse value: lower value = higher risk
    norm_inv_value = _min_max([-v for v in values])
    norm_eligible = _min_max(eligible_pcts)
    norm_renter = _min_max(renter_rates)
    norm_cost = _min_max(costs)

    for i, z in enumerate(zip_metrics):
        raw_score = (
            0.35 * norm_lead[i]
            + 0.25 * norm_inv_value[i]
            + 0.20 * norm_eligible[i]
            + 0.10 * norm_renter[i]
            + 0.10 * norm_cost[i]
        )
        z["composite_risk_score"] = round(raw_score, 4)

    # Sort by risk score descending
    zip_metrics.sort(key=lambda z: z["composite_risk_score"], reverse=True)

    return {
        "zip_metrics": zip_metrics,
        "total_parcels": len(parcels),
        "zip_count": len(zip_metrics),
        "highest_risk_zip": zip_metrics[0]["zip_code"],
        "lowest_risk_zip": zip_metrics[-1]["zip_code"],
    }
