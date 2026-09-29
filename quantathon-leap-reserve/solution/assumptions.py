from dataclasses import dataclass, field
from typing import Dict, List


START_YEAR = 2026
END_YEAR = 2037
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Regime indices
REGIME_BOOM = 0
REGIME_NORMAL = 1
REGIME_RECESSION = 2


@dataclass(frozen=True)
class Scenario:
    name: str
    annual_cost_inflation: float
    annual_sale_probability: float
    annual_equity_probability: float
    annual_inheritance_probability: float
    inherited_sale_probability: float
    inherited_equity_probability: float
    leak_share_of_trigger: float
    annual_trigger_start: float
    annual_trigger_cap: float
    annual_trigger_growth: float
    schedule_weights: List[float]

    # --- Housing market regime switching ---
    # 3x3 Markov transition matrix: [Boom, Normal, Recession]
    regime_transition_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.70, 0.25, 0.05],  # From Boom
        [0.15, 0.70, 0.15],  # From Normal
        [0.05, 0.25, 0.70],  # From Recession
    ])
    regime_sale_multipliers: List[float] = field(default_factory=lambda: [1.30, 1.00, 0.60])
    regime_equity_multipliers: List[float] = field(default_factory=lambda: [1.50, 1.00, 0.40])
    regime_appreciation_rates: List[float] = field(default_factory=lambda: [0.08, 0.03, -0.05])
    initial_regime: int = REGIME_NORMAL

    # --- Weibull survival-based sale hazard ---
    weibull_k: float = 1.3       # shape (>1 = increasing hazard with tenure)
    weibull_lambda: float = 18.0  # scale (calibrated to ~5.6% avg annual hazard)

    # --- Deadline urgency ramp (logistic S-curve) ---
    urgency_A: float = 3.0        # max additional multiplier at saturation
    urgency_B: float = 0.8        # steepness
    urgency_inflection: int = 2033  # year of maximum acceleration

    # --- Home price appreciation & equity tracking ---
    mortgage_amortization_rate: float = 0.0333  # ~1/30 for 30-year mortgage
    equity_baseline_ratio: float = 0.40          # equity ratio where factor = 1.0

    # --- Contractor capacity constraints ---
    contractor_capacity_base: int = 120          # max replacements/year initially
    contractor_capacity_growth: float = 0.10     # annual capacity growth (ramp-up)
    contractor_capacity_max: int = 250           # absolute max replacements/year

    # --- Fund investment returns ---
    fund_annual_return: float = 0.04             # municipal bond / T-bill yield

    # --- Loss given default (partial recovery) ---
    ltv_recovery_floor: float = 0.70             # min recovery fraction even in recession
    recession_recovery_haircut: float = 0.15     # extra loss in recession (homes underwater)


BASE_SCENARIO = Scenario(
    name="base",
    annual_cost_inflation=0.03,
    annual_sale_probability=0.056,
    annual_equity_probability=0.020,
    annual_inheritance_probability=0.008,
    inherited_sale_probability=0.080,
    inherited_equity_probability=0.010,
    leak_share_of_trigger=59.0 / 90.0,
    annual_trigger_start=0.0020,
    annual_trigger_cap=0.0030,
    annual_trigger_growth=0.05,
    schedule_weights=[0.07, 0.075, 0.08, 0.085, 0.09, 0.09, 0.09, 0.09, 0.085, 0.085, 0.08, 0.08],
)


OPTIMISTIC_SCENARIO = Scenario(
    name="optimistic",
    annual_cost_inflation=0.02,
    annual_sale_probability=0.065,
    annual_equity_probability=0.028,
    annual_inheritance_probability=0.006,
    inherited_sale_probability=0.100,
    inherited_equity_probability=0.015,
    leak_share_of_trigger=0.60,
    annual_trigger_start=0.0016,
    annual_trigger_cap=0.0024,
    annual_trigger_growth=0.04,
    schedule_weights=[0.06, 0.065, 0.07, 0.08, 0.085, 0.09, 0.095, 0.1, 0.095, 0.09, 0.085, 0.085],
    # Optimistic: milder regimes, lower urgency, faster repayment
    regime_sale_multipliers=[1.35, 1.00, 0.70],
    regime_equity_multipliers=[1.60, 1.00, 0.50],
    regime_appreciation_rates=[0.10, 0.04, -0.03],
    urgency_A=2.0,
    urgency_B=0.7,
    urgency_inflection=2034,
    weibull_k=1.4,
    weibull_lambda=16.0,
    # Optimistic: more capacity, higher returns, better recovery
    contractor_capacity_base=150,
    contractor_capacity_growth=0.12,
    contractor_capacity_max=300,
    fund_annual_return=0.045,
    ltv_recovery_floor=0.80,
    recession_recovery_haircut=0.10,
)


CONSERVATIVE_SCENARIO = Scenario(
    name="conservative",
    annual_cost_inflation=0.04,
    annual_sale_probability=0.045,
    annual_equity_probability=0.015,
    annual_inheritance_probability=0.012,
    inherited_sale_probability=0.060,
    inherited_equity_probability=0.008,
    leak_share_of_trigger=0.70,
    annual_trigger_start=0.0025,
    annual_trigger_cap=0.0035,
    annual_trigger_growth=0.06,
    schedule_weights=[0.10, 0.095, 0.09, 0.09, 0.085, 0.085, 0.08, 0.08, 0.075, 0.075, 0.075, 0.07],
    # Conservative: harsher recessions, higher urgency, slower repayment
    regime_transition_matrix=[
        [0.65, 0.25, 0.10],
        [0.10, 0.65, 0.25],
        [0.05, 0.25, 0.70],
    ],
    regime_sale_multipliers=[1.20, 1.00, 0.50],
    regime_equity_multipliers=[1.30, 1.00, 0.30],
    regime_appreciation_rates=[0.06, 0.02, -0.07],
    urgency_A=4.0,
    urgency_B=0.9,
    urgency_inflection=2032,
    weibull_k=1.2,
    weibull_lambda=20.0,
    # Conservative: less capacity, lower returns, worse recovery
    contractor_capacity_base=100,
    contractor_capacity_growth=0.08,
    contractor_capacity_max=200,
    fund_annual_return=0.035,
    ltv_recovery_floor=0.60,
    recession_recovery_haircut=0.20,
)


SCENARIOS: Dict[str, Scenario] = {
    "base": BASE_SCENARIO,
    "optimistic": OPTIMISTIC_SCENARIO,
    "conservative": CONSERVATIVE_SCENARIO,
}
