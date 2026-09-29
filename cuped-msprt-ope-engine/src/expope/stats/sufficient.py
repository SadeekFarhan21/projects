"""Sufficient statistics: the contract between the SQL metrics layer and the tests.

Every A/B test in :mod:`expope.stats` can be computed from a handful of per-arm sums.
The DuckDB layer emits exactly these sums, so the heavy lifting (grouping millions of
events) happens in SQL and the statistics code only ever sees O(1) numbers per arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeanStats:
    """Per-arm sufficient statistics for a per-user mean metric."""

    n: float
    sum_y: float
    sum_y2: float

    @property
    def mean(self) -> float:
        return self.sum_y / self.n

    @property
    def var(self) -> float:
        """Unbiased sample variance (ddof=1)."""
        n = self.n
        return (self.sum_y2 - self.sum_y * self.sum_y / n) / (n - 1)

    @classmethod
    def from_array(cls, y: np.ndarray) -> "MeanStats":
        y = np.asarray(y, dtype=float)
        return cls(n=float(y.size), sum_y=float(y.sum()), sum_y2=float((y * y).sum()))


@dataclass(frozen=True)
class RatioStats:
    """Per-arm sufficient statistics for a ratio metric sum(num) / sum(den) over users.

    Each user contributes one (num, den) pair, for example (clicks, sessions).
    """

    n: float
    sum_x: float  # numerator
    sum_y: float  # denominator
    sum_x2: float
    sum_y2: float
    sum_xy: float

    @classmethod
    def from_arrays(cls, num: np.ndarray, den: np.ndarray) -> "RatioStats":
        x = np.asarray(num, dtype=float)
        y = np.asarray(den, dtype=float)
        return cls(
            n=float(x.size),
            sum_x=float(x.sum()),
            sum_y=float(y.sum()),
            sum_x2=float((x * x).sum()),
            sum_y2=float((y * y).sum()),
            sum_xy=float((x * y).sum()),
        )


@dataclass(frozen=True)
class CovariateStats:
    """Per-arm sufficient statistics for CUPED: outcome y and pre-period covariate x."""

    n: float
    sum_y: float
    sum_x: float
    sum_y2: float
    sum_x2: float
    sum_xy: float

    @classmethod
    def from_arrays(cls, y: np.ndarray, x: np.ndarray) -> "CovariateStats":
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        return cls(
            n=float(y.size),
            sum_y=float(y.sum()),
            sum_x=float(x.sum()),
            sum_y2=float((y * y).sum()),
            sum_x2=float((x * x).sum()),
            sum_xy=float((x * y).sum()),
        )
