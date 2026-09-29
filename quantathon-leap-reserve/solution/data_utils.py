import csv
import math
import random as _random
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Dict, List


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def _read_csv_rows(path: Path) -> List[List[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def load_columbus_sample() -> Dict[str, List[Dict[str, float]]]:
    rows = _read_csv_rows(DATASET_DIR / "Columbus v4.csv")
    header = rows[1]
    body = rows[2:]
    idx = {name: position for position, name in enumerate(header)}
    grouped = {"lead": [], "galvanized": []}
    for row in body:
        material = row[idx["line_material"]]
        if material not in grouped:
            continue
        grouped[material].append(
            {
                "assessed_value": float(row[idx["assessed_value"]]),
                "estimated_replacement_cost": float(row[idx["estimated_replacement_cost"]]),
                "loan_proxy": min(10000.0, float(row[idx["estimated_replacement_cost"]])),
            }
        )
    return grouped


def load_quote_history() -> Dict[str, List[float]]:
    rows = _read_csv_rows(DATASET_DIR / "CWP - LEAP Data v2 3.25.26 - LEAP Quotes - No Names.csv")
    loans = []
    chosen = []
    for index, row in enumerate(rows):
        if 8 <= index <= 71 and row[6].strip():
            chosen.append(float(row[5]))
            loans.append(float(row[6]))
    return {"loan_amounts": loans, "chosen_amounts": chosen}


def load_program_snapshot() -> Dict[str, float]:
    rows = _read_csv_rows(DATASET_DIR / "CWP - LEAP Data v2 3.25.26 - LEAP Participation.csv")
    values: Dict[str, float] = {}
    last_label = None
    for row in rows:
        label = row[2].strip() if len(row) > 2 else ""
        value = row[3].strip() if len(row) > 3 else ""
        if label:
            last_label = label
        elif last_label and value:
            values[last_label] = float(value)
            last_label = None
    return values


def load_system_materials() -> Dict[str, int]:
    rows = _read_csv_rows(DATASET_DIR / "CWP - LEAP Data v2 3.25.26 - System Materials.csv")
    total_lines = 0
    whole_lead = 0
    customer_galvanized = 0
    overlap = 0
    for row in rows[4:]:
        if len(row) < 5 or not row[3].strip():
            continue
        customer_material = row[2]
        count = int(row[3])
        whole_designation = row[4]
        total_lines += count
        if whole_designation == "Lead":
            whole_lead += count
        if customer_material == "Galvanized":
            customer_galvanized += count
        if whole_designation == "Lead" and customer_material == "Galvanized":
            overlap += count
    eligible_lines = whole_lead + customer_galvanized - overlap
    return {
        "total_lines": total_lines,
        "whole_lead": whole_lead,
        "customer_galvanized": customer_galvanized,
        "eligible_lines": eligible_lines,
        "lead_only_share": whole_lead / eligible_lines,
        "galvanized_only_share": (customer_galvanized - overlap) / eligible_lines,
    }


def load_contractor_quotes() -> List[float]:
    """Load all contractor quotes from v2 LEAP Quotes as replacement cost distribution."""
    rows = _read_csv_rows(
        DATASET_DIR / "CWP - LEAP Data v2 3.25.26 - LEAP Quotes - No Names.csv"
    )
    costs: List[float] = []
    for index, row in enumerate(rows):
        if 8 <= index <= 71:
            for col in (1, 2, 3):
                val = row[col].strip() if len(row) > col else ""
                if val:
                    try:
                        costs.append(float(val))
                    except ValueError:
                        pass
    return costs


def generate_v2_training_parcels(n: int = 8000, seed: int = 42) -> List[Dict]:
    """Generate synthetic training parcels from v2 LEAP data distributions.

    Uses the System Materials file for material distribution, the LEAP Quotes
    file for replacement cost distribution, and Columbus-representative
    distributions for demographic and property features.  These parcels serve
    as the training set so that all model calibration derives from v2 (real)
    data, while the Columbus v4 synthetic dataset is reserved as a held-out
    test set.
    """
    rng = _random.Random(seed)

    materials = load_system_materials()
    contractor_costs = load_contractor_quotes()

    # Material distribution from v2 System Materials (among known lines)
    total = materials["total_lines"]
    lead_frac = materials["whole_lead"] / total
    galv_frac = (materials["customer_galvanized"]
                 - (materials["whole_lead"] + materials["customer_galvanized"]
                    - materials["eligible_lines"])) / total
    # remainder → copper / other non-lead
    copper_frac = 1.0 - lead_frac - galv_frac

    zip_codes = [43201, 43202, 43206, 43210, 43215, 43220, 43221, 43230]
    property_types = ["single_family", "duplex", "multi_family"]
    property_weights = [0.70, 0.20, 0.10]

    parcels: List[Dict] = []
    for i in range(n):
        # --- material ---
        r = rng.random()
        if r < lead_frac:
            material = "lead"
        elif r < lead_frac + galv_frac:
            material = "galvanized"
        else:
            material = "copper"

        # --- year built (older for lead/galv) ---
        if material == "lead":
            year_built = rng.randint(1920, 1970)
        elif material == "galvanized":
            year_built = rng.randint(1930, 1985)
        else:
            year_built = rng.randint(1950, 2024)

        # --- assessed value (log-normal, lower for lead/galv areas) ---
        base_value = 90000 if material in ("lead", "galvanized") else 130000
        assessed_value = max(15000, base_value * math.exp(rng.gauss(0, 0.5)))

        # --- ownership ---
        ownership_length = rng.randint(1, 33)
        last_sale_year = 2026 - ownership_length
        owner_occupied = 1 if rng.random() < 0.62 else 0

        # --- lead risk (correlated with material) ---
        if material == "lead":
            lead_risk = rng.uniform(0.60, 1.00)
        elif material == "galvanized":
            lead_risk = rng.uniform(0.30, 0.70)
        else:
            lead_risk = rng.uniform(0.00, 0.35)

        # --- replacement cost (bootstrap from v2 contractor quotes) ---
        base_cost = contractor_costs[rng.randrange(len(contractor_costs))]
        replacement_cost = max(4000.0, round(base_cost * (1.0 + rng.gauss(0, 0.10))))

        # --- categorical features ---
        zip_code = rng.choice(zip_codes)
        r2 = rng.random()
        cumul = 0.0
        property_type = property_types[-1]
        for pt, pw in zip(property_types, property_weights):
            cumul += pw
            if r2 < cumul:
                property_type = pt
                break

        parcels.append({
            "parcel_id": str(200000 + i),
            "zip_code": zip_code,
            "property_type": property_type,
            "year_built": year_built,
            "assessed_value": round(assessed_value, 0),
            "last_sale_year": last_sale_year,
            "ownership_length": ownership_length,
            "owner_occupied": owner_occupied,
            "lead_risk": round(lead_risk, 3),
            "line_material": material,
            "estimated_replacement_cost": replacement_cost,
        })

    return parcels


def summarize_inputs() -> Dict[str, float]:
    materials = load_system_materials()
    quotes = load_quote_history()
    program = load_program_snapshot()
    return {
        "eligible_lines": materials["eligible_lines"],
        "lead_share": materials["lead_only_share"],
        "galvanized_share": materials["galvanized_only_share"],
        "mean_loan_amount": fmean(quotes["loan_amounts"]),
        "mean_chosen_amount": fmean(quotes["chosen_amounts"]),
        "active_loans_observed": program["# of completed loans"] + program["# of in progress loans"],
        "application_count_observed": program["# of applications sent to those interested customers"],
        "interest_count_observed": program["# of interested applicants"],
    }
