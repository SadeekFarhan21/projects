"""Forecasting models. All take the long (date, symbol) frame of ranked
features and return a forecast Series on the same index."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .features import FEATURES


class SingleFactor:
    """No fitting: the forecast is +/- one ranked feature."""

    def __init__(self, feature: str, sign: float):
        self.feature, self.sign = feature, sign

    def fit(self, train: pd.DataFrame) -> "SingleFactor":
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        return self.sign * test[self.feature]


class _Sklearn:
    def __init__(self, est):
        self.est = est

    # On this machine sklearn's OpenMP pool hung indefinitely inside
    # HistGradientBoosting once the training set passed ~50k rows (see
    # DEVLOG "Problems"). Single-threaded fits take seconds, so pin to 1.
    def fit(self, train: pd.DataFrame):
        t = train.dropna(subset=["y_rank"])
        with threadpool_limits(1):
            self.est.fit(t[FEATURES].to_numpy(np.float32), t["y_rank"].to_numpy(np.float32))
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        with threadpool_limits(1):
            p = self.est.predict(test[FEATURES].to_numpy(np.float32))
        return pd.Series(p, index=test.index)


class RidgeModel(_Sklearn):
    def __init__(self, alpha: float):
        super().__init__(Ridge(alpha=alpha))

    @property
    def coef(self) -> dict[str, float]:
        return dict(zip(FEATURES, self.est.coef_.tolist()))


class GBMModel(_Sklearn):
    def __init__(self, max_depth: int, max_iter: int = 200, learning_rate: float = 0.05,
                 seed: int = 0):
        super().__init__(HistGradientBoostingRegressor(
            max_depth=max_depth, max_iter=max_iter, learning_rate=learning_rate,
            min_samples_leaf=200, early_stopping=False, random_state=seed))


# The pre-registered model list (PREREGISTRATION.md section 7). Order matters
# only for display.
def registry() -> dict[str, callable]:
    return {
        "rev_1d": lambda: SingleFactor("ret_1d", -1.0),
        "rev_5d": lambda: SingleFactor("ret_5d", -1.0),
        "mom_20d": lambda: SingleFactor("ret_20d", +1.0),
        "mom_60d": lambda: SingleFactor("ret_60d", +1.0),
        "ridge_a1": lambda: RidgeModel(1.0),
        "ridge_a10": lambda: RidgeModel(10.0),
        "ridge_a100": lambda: RidgeModel(100.0),
        "gbm_d3": lambda: GBMModel(3),
        "gbm_d6": lambda: GBMModel(6),
    }
