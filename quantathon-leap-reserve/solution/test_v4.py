"""Out-of-sample testing on the Columbus v4 held-out test set.

The model is trained/calibrated on v2 LEAP data (System Materials, Quotes,
Participation).  This module derives inputs from the v4 synthetic parcel
dataset and compares against the v2-trained model results to validate that
the model generalizes to unseen data.
"""

import json
import random
from collections import Counter
from pathlib import Path
from statistics import fmean, median, stdev

from .assumptions import BASE_SCENARIO, CONSERVATIVE_SCENARIO, OPTIMISTIC_SCENARIO, YEARS
from .data_utils import _read_csv_rows, DATASET_DIR, load_quote_history, load_system_materials
from .model import PortfolioInputs, _weighted_schedule_counts, run_simulation, simulate_path


def load_v4_data():
    """Load and parse the Columbus v4 synthetic test set."""
    rows = _read_csv_rows(DATASET_DIR / "Columbus v4.csv")
    header = rows[1]
    body = rows[2:]
    idx = {name: pos for pos, name in enumerate(header)}

    parcels = []
    for row in body:
        parcels.append({
            "parcel_id": row[idx["parcel_id"]],
            "zip_code": row[idx["zip_code"]],
            "property_type": row[idx["property_type"]],
            "year_built": int(row[idx["year_built"]]),
            "assessed_value": float(row[idx["assessed_value"]]),
            "last_sale_year": int(row[idx["last_sale_year"]]),
            "ownership_length": int(row[idx["ownership_length"]]),
            "owner_occupied": int(row[idx["owner_occupied"]]),
            "lead_risk": float(row[idx["lead_risk"]]),
            "line_material": row[idx["line_material"]],
            "estimated_replacement_cost": float(row[idx["estimated_replacement_cost"]]),
        })
    return parcels


def summarize_v4(parcels):
    """Derive model-relevant statistics from v4 test set."""
    material_counts = Counter(p["line_material"] for p in parcels)
    eligible = [p for p in parcels if p["line_material"] in ("lead", "galvanized")]
    lead_parcels = [p for p in parcels if p["line_material"] == "lead"]
    galv_parcels = [p for p in parcels if p["line_material"] == "galvanized"]

    # Loan amounts: estimated_replacement_cost capped at $10K
    loan_proxies = [min(10000.0, p["estimated_replacement_cost"]) for p in eligible]
    replacement_costs = [p["estimated_replacement_cost"] for p in eligible]

    # Ownership / turnover stats
    ownership_lengths = [p["ownership_length"] for p in eligible]
    assessed_values = [p["assessed_value"] for p in eligible]
    owner_occ_rate = sum(p["owner_occupied"] for p in eligible) / len(eligible)
    lead_risk_scores = [p["lead_risk"] for p in eligible]

    # Implied sale probability from ownership length
    avg_ownership = fmean(ownership_lengths)
    implied_sale_prob = 1.0 / avg_ownership if avg_ownership > 0 else 0.0

    return {
        "total_parcels": len(parcels),
        "material_breakdown": dict(material_counts),
        "eligible_count": len(eligible),
        "lead_count": len(lead_parcels),
        "galvanized_count": len(galv_parcels),
        "lead_share": len(lead_parcels) / len(eligible),
        "galvanized_share": len(galv_parcels) / len(eligible),
        "loan_proxy_mean": round(fmean(loan_proxies), 2),
        "loan_proxy_median": round(median(loan_proxies), 2),
        "loan_proxy_stdev": round(stdev(loan_proxies), 2),
        "replacement_cost_mean": round(fmean(replacement_costs), 2),
        "replacement_cost_median": round(median(replacement_costs), 2),
        "avg_ownership_length": round(avg_ownership, 1),
        "implied_annual_sale_prob": round(implied_sale_prob, 4),
        "avg_assessed_value": round(fmean(assessed_values), 0),
        "owner_occupied_rate": round(owner_occ_rate, 4),
        "avg_lead_risk": round(fmean(lead_risk_scores), 4),
        "loan_proxies": loan_proxies,
    }


def build_v4_portfolio(v4_summary, scenario):
    """Build PortfolioInputs from v4 test data for out-of-sample validation."""
    eligible = v4_summary["eligible_count"]
    # Scale up: v4 has 10K parcels as a sample; v2 system materials show ~34K eligible
    # We test at the v4 sample size directly (no scaling)
    return PortfolioInputs(
        eligible_lines=eligible,
        schedule_counts=_weighted_schedule_counts(eligible, scenario.schedule_weights),
        loan_amount_history=v4_summary["loan_proxies"],
    )


def run_v4_test(simulation_count=1000, seed=42):
    """Run the v2-trained model on v4 test set and compare results."""
    print("=" * 70)
    print("LEAP Model Test: Out-of-Sample Validation on Columbus v4")
    print("=" * 70)

    # --- Load and summarize v4 data ---
    parcels = load_v4_data()
    v4 = summarize_v4(parcels)

    print(f"\n--- v4 Test Set Summary ---")
    print(f"Total parcels:          {v4['total_parcels']:,}")
    print(f"Material breakdown:     {v4['material_breakdown']}")
    print(f"Eligible (lead+galv):   {v4['eligible_count']:,}")
    print(f"  Lead:                 {v4['lead_count']:,} ({v4['lead_share']:.1%})")
    print(f"  Galvanized:           {v4['galvanized_count']:,} ({v4['galvanized_share']:.1%})")
    print(f"Loan proxy (capped):    mean=${v4['loan_proxy_mean']:,.0f}  median=${v4['loan_proxy_median']:,.0f}  stdev=${v4['loan_proxy_stdev']:,.0f}")
    print(f"Replacement cost:       mean=${v4['replacement_cost_mean']:,.0f}  median=${v4['replacement_cost_median']:,.0f}")
    print(f"Avg ownership length:   {v4['avg_ownership_length']} years")
    print(f"Implied sale prob:      {v4['implied_annual_sale_prob']:.4f}")
    print(f"Avg assessed value:     ${v4['avg_assessed_value']:,.0f}")
    print(f"Owner-occupied rate:    {v4['owner_occupied_rate']:.1%}")
    print(f"Avg lead risk score:    {v4['avg_lead_risk']:.4f}")

    # --- Load v2 comparison data ---
    v2_materials = load_system_materials()
    v2_quotes = load_quote_history()
    v2_mean_loan = fmean(v2_quotes["loan_amounts"])

    print(f"\n--- v2 (Training) vs v4 (Test) Comparison ---")
    print(f"{'Metric':<30} {'v2 (train)':>15} {'v4 (test)':>15} {'Delta':>12}")
    print("-" * 72)
    print(f"{'Eligible lines':<30} {v2_materials['eligible_lines']:>15,} {v4['eligible_count']:>15,} {v4['eligible_count'] - v2_materials['eligible_lines']:>+12,}")
    print(f"{'Lead share':<30} {v2_materials['lead_only_share']:>15.1%} {v4['lead_share']:>15.1%} {v4['lead_share'] - v2_materials['lead_only_share']:>+12.1%}")
    print(f"{'Galvanized share':<30} {v2_materials['galvanized_only_share']:>15.1%} {v4['galvanized_share']:>15.1%} {v4['galvanized_share'] - v2_materials['galvanized_only_share']:>+12.1%}")
    print(f"{'Mean loan amount':<30} ${v2_mean_loan:>14,.0f} ${v4['loan_proxy_mean']:>14,.0f} ${v4['loan_proxy_mean'] - v2_mean_loan:>+11,.0f}")
    print(f"{'Sale prob (base scenario)':<30} {BASE_SCENARIO.annual_sale_probability:>15.4f} {v4['implied_annual_sale_prob']:>15.4f} {v4['implied_annual_sale_prob'] - BASE_SCENARIO.annual_sale_probability:>+12.4f}")

    # --- Run simulation with v4 inputs ---
    print(f"\n--- Running Monte Carlo ({simulation_count:,} paths, base scenario) ---")
    portfolio_v4 = build_v4_portfolio(v4, BASE_SCENARIO)
    rng = random.Random(seed)
    paths = [
        simulate_path(portfolio=portfolio_v4, scenario=BASE_SCENARIO, rng=random.Random(rng.randint(0, 10**9)))
        for _ in range(simulation_count)
    ]

    loan_reqs = sorted([p.required_funding_loan for p in paths])
    grant_reqs = sorted([p.required_funding_grant for p in paths])

    def percentile(vals, q):
        idx = min(len(vals) - 1, max(0, int(q * len(vals)) - 1))
        return vals[idx]

    v4_loan_p95 = percentile(loan_reqs, 0.95)
    v4_grant_p95 = percentile(grant_reqs, 0.95)
    v4_loan_mean = fmean(loan_reqs)
    v4_grant_mean = fmean(grant_reqs)

    print(f"\n--- v4 Simulation Results (Base Scenario) ---")
    print(f"{'Metric':<35} {'Loan':>15} {'Grant':>15}")
    print("-" * 65)
    print(f"{'Mean funding requirement':<35} ${v4_loan_mean:>14,.0f} ${v4_grant_mean:>14,.0f}")
    print(f"{'Median':<35} ${percentile(loan_reqs, 0.50):>14,.0f} ${percentile(grant_reqs, 0.50):>14,.0f}")
    print(f"{'P5 (low)':<35} ${percentile(loan_reqs, 0.05):>14,.0f} ${percentile(grant_reqs, 0.05):>14,.0f}")
    print(f"{'P95 (recommended)':<35} ${v4_loan_p95:>14,.0f} ${v4_grant_p95:>14,.0f}")
    print(f"{'Min':<35} ${loan_reqs[0]:>14,.0f} ${grant_reqs[0]:>14,.0f}")
    print(f"{'Max':<35} ${loan_reqs[-1]:>14,.0f} ${grant_reqs[-1]:>14,.0f}")

    # --- Compare to v2-based results ---
    print(f"\n--- Running v2-trained model for comparison ({simulation_count:,} paths) ---")
    v2_results = run_simulation(BASE_SCENARIO, simulation_count, seed)
    v2_loan_p95 = v2_results["loan_summary"]["p95"]
    v2_grant_p95 = v2_results["grant_summary"]["p95"]
    v2_loan_mean = v2_results["loan_summary"]["mean"]

    print(f"\n--- v2 (Train) vs v4 (Test) Results Comparison ---")
    print(f"{'Metric':<35} {'v2 (train)':>15} {'v4 (test)':>15} {'Ratio':>10}")
    print("-" * 75)
    print(f"{'Eligible lines':<35} {v2_materials['eligible_lines']:>15,} {v4['eligible_count']:>15,} {v4['eligible_count']/v2_materials['eligible_lines']:>10.2f}")
    print(f"{'Mean loan amount':<35} ${v2_mean_loan:>14,.0f} ${v4['loan_proxy_mean']:>14,.0f} {v4['loan_proxy_mean']/v2_mean_loan:>10.2f}")
    print(f"{'Loan P95 funding':<35} ${v2_loan_p95:>14,.0f} ${v4_loan_p95:>14,.0f} {v4_loan_p95/v2_loan_p95:>10.2f}")
    print(f"{'Loan mean funding':<35} ${v2_loan_mean:>14,.0f} ${v4_loan_mean:>14,.0f} {v4_loan_mean/v2_loan_mean:>10.2f}")
    print(f"{'Grant P95 funding':<35} ${v2_grant_p95:>14,.0f} ${v4_grant_p95:>14,.0f} {v4_grant_p95/v2_grant_p95:>10.2f}")

    # --- Annual cash flow comparison ---
    avg_outflows_v4 = {}
    avg_inflows_v4 = {}
    for year in YEARS:
        avg_outflows_v4[year] = fmean(p.annual_outflows[year] for p in paths)
        avg_inflows_v4[year] = fmean(p.annual_inflows[year] for p in paths)

    print(f"\n--- Annual Cash Flow Comparison (v4 vs v2, Base Scenario) ---")
    print(f"{'Year':<6} {'v4 Outflow':>14} {'v2 Outflow':>14} {'v4 Inflow':>14} {'v2 Inflow':>14} {'v4 Net':>14}")
    print("-" * 76)
    for year in YEARS:
        v4_out = avg_outflows_v4[year]
        v2_out = v2_results["average_outflows"][year]
        v4_in = avg_inflows_v4[year]
        v2_in = v2_results["average_inflows"][year]
        print(f"{year:<6} ${v4_out:>13,.0f} ${v2_out:>13,.0f} ${v4_in:>13,.0f} ${v2_in:>13,.0f} ${v4_out - v4_in:>13,.0f}")

    # --- Scaling analysis ---
    scale_factor = v2_materials["eligible_lines"] / v4["eligible_count"]
    scaled_v4_p95 = v4_loan_p95 * scale_factor
    print(f"\n--- Scaling Analysis ---")
    print(f"v4 is a sample of {v4['eligible_count']:,} eligible parcels")
    print(f"v2 shows {v2_materials['eligible_lines']:,} eligible lines citywide")
    print(f"Scale factor: {scale_factor:.2f}x")
    print(f"v4 Loan P95 scaled up:    ${scaled_v4_p95:,.0f}")
    print(f"v2 Loan P95 (actual):     ${v2_loan_p95:,.0f}")
    print(f"Ratio (scaled v4 / v2):   {scaled_v4_p95/v2_loan_p95:.2f}")

    print(f"\n{'=' * 70}")
    print("Test complete.")
    return {
        "v4_summary": {k: v for k, v in v4.items() if k != "loan_proxies"},
        "v4_loan_p95": v4_loan_p95,
        "v4_grant_p95": v4_grant_p95,
        "v2_loan_p95": v2_loan_p95,
        "v2_grant_p95": v2_grant_p95,
        "scale_factor": scale_factor,
        "scaled_v4_p95": scaled_v4_p95,
    }


if __name__ == "__main__":
    run_v4_test()
