"""Walk-forward evaluation with an embargo between train and test."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEV_START, DEV_END = "2020-01-01", "2024-06-30"
HOLDOUT_START, HOLDOUT_END = "2024-07-01", "2026-08-31"
EMBARGO_DAYS = 5


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_end(self) -> pd.Timestamp:
        # A training row dated d has a label that is known at the close of d+1.
        # Requiring d + 1 <= test_start - EMBARGO_DAYS leaves a gap of
        # EMBARGO_DAYS between the last label and the first test day.
        return self.test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)


def half_year_folds(start: str = DEV_START, end: str = DEV_END) -> list[Fold]:
    folds = []
    s = pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.DateOffset(months=6) - pd.Timedelta(days=1), pd.Timestamp(end))
        folds.append(Fold(f"{s.year}H{1 if s.month <= 6 else 2}", s, e))
        s = e + pd.Timedelta(days=1)
    return folds


def walk_forward(model_factory, long: pd.DataFrame, folds: list[Fold]) -> tuple[pd.Series, list]:
    """Fit a fresh model per fold on data up to fold.train_end and predict the
    fold's test window. Returns the concatenated out-of-sample forecast and the
    fitted models."""
    dates = long.index.get_level_values("date")
    preds, fitted = [], []
    for f in folds:
        train = long[dates <= f.train_end]
        test = long[(dates >= f.test_start) & (dates <= f.test_end)]
        m = model_factory().fit(train)
        preds.append(m.predict(test))
        fitted.append(m)
    return pd.concat(preds).sort_index(), fitted
