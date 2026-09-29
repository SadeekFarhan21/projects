"""Bootstrap confidence intervals for Monte Carlo simulation statistics.

Resamples from the MC path-level funding requirements to construct
confidence intervals on key statistics (mean, median, p95).
"""

import math
import random
from statistics import fmean
from typing import Dict, List


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for mean, median, and p95.

    Args:
        values: Original MC path-level values (e.g. loan_requirements).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        ci_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        Dict with keys 'mean', 'median', 'p95', each containing
        'point_estimate', 'ci_lower', 'ci_upper', 'ci_width', 'se'.
    """
    rng = random.Random(seed)
    n = len(values)
    alpha = (1.0 - ci_level) / 2.0

    boot_means = []
    boot_medians = []
    boot_p95s = []

    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(fmean(sample))
        boot_medians.append(_percentile(sample, 0.50))
        boot_p95s.append(_percentile(sample, 0.95))

    def _summarize(boot_stats: List[float], point_est: float) -> Dict[str, float]:
        lower = _percentile(boot_stats, alpha)
        upper = _percentile(boot_stats, 1.0 - alpha)
        se = (sum((x - fmean(boot_stats)) ** 2 for x in boot_stats) / (len(boot_stats) - 1)) ** 0.5
        return {
            "point_estimate": round(point_est, 2),
            "ci_lower": round(lower, 2),
            "ci_upper": round(upper, 2),
            "ci_width": round(upper - lower, 2),
            "standard_error": round(se, 2),
        }

    return {
        "mean": _summarize(boot_means, fmean(values)),
        "median": _summarize(boot_medians, _percentile(values, 0.50)),
        "p95": _summarize(boot_p95s, _percentile(values, 0.95)),
    }


def run_bootstrap_analysis(
    loan_requirements: List[float],
    grant_requirements: List[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict:
    """Run bootstrap CI analysis for both loan and grant funding requirements."""
    return {
        "loan": bootstrap_ci(loan_requirements, n_bootstrap=n_bootstrap, seed=seed),
        "grant": bootstrap_ci(grant_requirements, n_bootstrap=n_bootstrap, seed=seed + 1),
        "n_bootstrap": n_bootstrap,
        "n_paths": len(loan_requirements),
    }
