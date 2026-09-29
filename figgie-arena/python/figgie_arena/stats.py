"""Statistics helpers for tournaments: clustered bootstrap intervals."""

from __future__ import annotations

import numpy as np


def cluster_means(values: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Mean of ``values`` within each cluster id (ids are 0..K-1)."""
    k = int(clusters.max()) + 1
    sums = np.bincount(clusters, weights=values, minlength=k)
    counts = np.bincount(clusters, minlength=k)
    return sums / counts


def bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap (1 - alpha) interval.

    When ``clusters`` is given, whole clusters are resampled. In the tournament a
    cluster is one deal replayed in every seat rotation, so the rotations of a
    deal are not treated as independent samples.
    """
    values = np.asarray(values, dtype=np.float64)
    if clusters is None:
        units = values
        weights = np.ones_like(units)
    else:
        units = cluster_means(values, clusters)
        weights = np.bincount(clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n_boot, len(units)))
    boot = (units[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
