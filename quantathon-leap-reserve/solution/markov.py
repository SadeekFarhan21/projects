"""Absorbing Markov Chain analysis of the LEAP loan repayment process.

States: Active (0), Inherited (1), Repaid (2 — absorbing).
Derives closed-form expected absorption times, variances, and
cumulative repayment schedules from the transition matrix.
"""

from typing import Dict, List, Tuple

from .assumptions import BASE_SCENARIO, YEARS, Scenario


def _build_transition_matrix(scenario: Scenario) -> List[List[float]]:
    """Return the 3×3 transition matrix P for {Active, Inherited, Repaid}.

    Row i, column j = P(move to j | currently in i).
    """
    s = scenario
    # Active row
    p_a_repaid = s.annual_sale_probability + s.annual_equity_probability
    p_a_inherited = s.annual_inheritance_probability
    p_a_active = 1.0 - p_a_repaid - p_a_inherited

    # Inherited row
    p_i_repaid = s.inherited_sale_probability + s.inherited_equity_probability
    p_i_inherited = 1.0 - p_i_repaid
    p_i_active = 0.0

    # Repaid row (absorbing)
    return [
        [p_a_active, p_a_inherited, p_a_repaid],
        [p_i_active, p_i_inherited, p_i_repaid],
        [0.0, 0.0, 1.0],
    ]


def _invert_2x2(m: List[List[float]]) -> List[List[float]]:
    """Invert a 2×2 matrix [[a,b],[c,d]]."""
    a, b = m[0]
    c, d = m[1]
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ValueError("Singular matrix")
    return [[d / det, -b / det], [-c / det, a / det]]


def _mat_mul_2x2(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply two 2×2 matrices."""
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def compute_fundamental_matrix(scenario: Scenario) -> Dict:
    """Compute the fundamental matrix N = (I - Q)^{-1} and derived quantities.

    Returns dict with:
      - Q: 2×2 transient sub-matrix
      - N: fundamental matrix
      - expected_time: expected steps to absorption from each transient state
      - variance: variance of absorption time from each transient state
      - std_dev: standard deviation of absorption time
    """
    P = _build_transition_matrix(scenario)
    # Q is the 2×2 upper-left (transient-to-transient) block
    Q = [[P[0][0], P[0][1]], [P[1][0], P[1][1]]]

    # I - Q
    I_minus_Q = [[1.0 - Q[0][0], -Q[0][1]], [-Q[1][0], 1.0 - Q[1][1]]]

    # N = (I - Q)^{-1}
    N = _invert_2x2(I_minus_Q)

    # Expected absorption time = N * 1-vector
    expected_from_active = N[0][0] + N[0][1]
    expected_from_inherited = N[1][0] + N[1][1]

    # Variance: Var = (2N_dg - I) * N * 1 - (N * 1)^2 element-wise
    # More precisely: Var_i = (2 * N_dg - I) @ t - t_sq, where t = N*1
    # Using the standard formula: Var = (2*N - I)*t - t_sq
    # where t = N*1 (column vector of expected times)
    t = [expected_from_active, expected_from_inherited]
    # (2N - I)
    two_N_minus_I = [[2 * N[0][0] - 1, 2 * N[0][1]], [2 * N[1][0], 2 * N[1][1] - 1]]
    var_from_active = two_N_minus_I[0][0] * t[0] + two_N_minus_I[0][1] * t[1] - t[0] ** 2
    var_from_inherited = two_N_minus_I[1][0] * t[0] + two_N_minus_I[1][1] * t[1] - t[1] ** 2

    return {
        "P": P,
        "Q": Q,
        "N": N,
        "expected_time": [expected_from_active, expected_from_inherited],
        "variance": [max(0.0, var_from_active), max(0.0, var_from_inherited)],
        "std_dev": [max(0.0, var_from_active) ** 0.5, max(0.0, var_from_inherited) ** 0.5],
    }


def cumulative_repayment_schedule(scenario: Scenario, max_years: int = 30) -> List[Dict]:
    """Compute year-by-year cumulative repayment probability starting from Active.

    Returns list of dicts with year, p_active, p_inherited, p_repaid.
    Uses matrix power of the full P.
    """
    P = _build_transition_matrix(scenario)
    # State vector: starts as [1, 0, 0] (100% Active)
    state = [1.0, 0.0, 0.0]
    schedule = []
    for year in range(1, max_years + 1):
        # Multiply state by P (state = state @ P)
        new_state = [
            state[0] * P[0][0] + state[1] * P[1][0] + state[2] * P[2][0],
            state[0] * P[0][1] + state[1] * P[1][1] + state[2] * P[2][1],
            state[0] * P[0][2] + state[1] * P[1][2] + state[2] * P[2][2],
        ]
        state = new_state
        schedule.append({
            "year": year,
            "p_active": state[0],
            "p_inherited": state[1],
            "p_repaid": state[2],
        })
    return schedule


def cohort_recovery_fractions(scenario: Scenario) -> Dict[int, float]:
    """For each origination cohort year, compute the fraction repaid by 2037.

    A loan originated in year Y has (2037 - Y) years to repay within the program horizon.
    """
    schedule = cumulative_repayment_schedule(scenario, max_years=max(len(YEARS), 30))
    result = {}
    for orig_year in YEARS:
        years_to_repay = 2037 - orig_year
        if years_to_repay <= 0:
            result[orig_year] = 0.0
        else:
            idx = min(years_to_repay, len(schedule)) - 1
            result[orig_year] = schedule[idx]["p_repaid"]
    return result


def run_markov_analysis(scenario: Scenario = None) -> Dict:
    """Run the full Markov chain analysis and return all results."""
    if scenario is None:
        scenario = BASE_SCENARIO
    fundamental = compute_fundamental_matrix(scenario)
    schedule = cumulative_repayment_schedule(scenario, max_years=30)
    cohort_recovery = cohort_recovery_fractions(scenario)

    return {
        "transition_matrix": fundamental["P"],
        "Q_matrix": fundamental["Q"],
        "fundamental_matrix": fundamental["N"],
        "expected_absorption_time": {
            "from_active": round(fundamental["expected_time"][0], 2),
            "from_inherited": round(fundamental["expected_time"][1], 2),
        },
        "absorption_std_dev": {
            "from_active": round(fundamental["std_dev"][0], 2),
            "from_inherited": round(fundamental["std_dev"][1], 2),
        },
        "absorption_variance": {
            "from_active": round(fundamental["variance"][0], 2),
            "from_inherited": round(fundamental["variance"][1], 2),
        },
        "cumulative_schedule": schedule,
        "cohort_recovery_by_2037": {str(k): round(v, 4) for k, v in cohort_recovery.items()},
        "half_life_years": _find_half_life(schedule),
    }


def _find_half_life(schedule: List[Dict]) -> float:
    """Find the year at which cumulative repayment crosses 50%."""
    for entry in schedule:
        if entry["p_repaid"] >= 0.5:
            # Linear interpolation with previous year
            if entry["year"] == 1:
                return float(entry["year"])
            prev = schedule[entry["year"] - 2]
            frac = (0.5 - prev["p_repaid"]) / (entry["p_repaid"] - prev["p_repaid"])
            return round(prev["year"] + frac, 1)
    return float(len(schedule))
