"""A single library-wide numpy Generator so runs are reproducible with one seed."""

import numpy as np

_rng = np.random.default_rng(0)


def manual_seed(seed: int) -> None:
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _rng
