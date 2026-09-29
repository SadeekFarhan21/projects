"""Time-series cross-validation with purging and embargo (Lopez de Prado, AFML ch. 7).

Samples are indexed by decision date t (row of the panel). The label for sample t is a
forward return that is realized over rows [t + 1, t + label_horizon], so its information
interval is I(t) = [t, t + label_horizon].

purge     drop training rows whose label interval overlaps a test label interval. Test
          labels span [a, b + label_horizon] for a test block [a, b], so a training row
          t < a is dropped if t + label_horizon >= a, and a row t > b is dropped if
          t <= b + label_horizon.
embargo   additionally drop rows in (b + label_horizon, b + label_horizon + embargo]. Test-set labels
          end at b + label_horizon, and features built on trailing windows right after b
          are correlated with test labels, so training on them leaks through serial
          correlation. Only relevant when training rows follow the test block.

Split kinds:
    walk_forward  test blocks move forward; training uses rows strictly before the block
                  (expanding window, or rolling with train_window). Never trains on the future.
    kfold         purged k-fold: training uses rows on both sides of the test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray   # row indices
    test: np.ndarray    # row indices, a contiguous block

    @property
    def test_range(self) -> tuple[int, int]:
        return int(self.test[0]), int(self.test[-1])


def _test_blocks(n: int, n_splits: int, start: int) -> list[tuple[int, int]]:
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    return [(int(edges[k]), int(edges[k + 1]) - 1) for k in range(n_splits)
            if edges[k + 1] > edges[k]]


def purged_splits(n: int, n_splits: int, label_horizon: int, embargo: int = 0,
                  kind: str = "walk_forward", min_train: int = 0,
                  train_window: int | None = None, purge: bool = True) -> list[Split]:
    """Generate splits over n rows. purge=False exists only to measure what purging buys."""
    if kind not in ("walk_forward", "kfold"):
        raise ValueError(kind)
    rows = np.arange(n)
    first_test = min_train if kind == "walk_forward" else 0
    out = []
    for a, b in _test_blocks(n, n_splits, first_test):
        test = rows[a: b + 1]
        if kind == "walk_forward":
            lo = 0 if train_window is None else max(0, a - train_window)
            train_mask = (rows >= lo) & (rows < a)
        else:
            train_mask = (rows < a) | (rows > b)
        if purge:
            # Rows before the block whose labels reach into it.
            train_mask &= ~((rows < a) & (rows + label_horizon >= a))
            # Rows after the block that start inside the test labels' span.
            train_mask &= ~((rows > b) & (rows <= b + label_horizon))
        if embargo > 0:
            h = label_horizon if purge else 0
            train_mask &= ~((rows > b) & (rows <= b + h + embargo))
        train = rows[train_mask]
        if len(train) == 0:
            continue
        out.append(Split(train, test))
    return out


def check_split(split: Split, label_horizon: int, embargo: int) -> list[str]:
    """Return violations of the purge/embargo contract (empty list means clean)."""
    a, b = split.test_range
    tr = split.train
    errs = []
    if np.intersect1d(tr, split.test).size:
        errs.append("train and test overlap")
    before = tr[tr < a]
    if before.size and (before + label_horizon >= a).any():
        errs.append("training label interval overlaps test block (not purged)")
    after = tr[tr > b]
    if after.size and (after <= b + label_horizon).any():
        errs.append("training row starts inside the test labels' span (not purged)")
    if after.size and (after <= b + label_horizon + embargo).any():
        errs.append("training rows inside embargo window")
    return errs
