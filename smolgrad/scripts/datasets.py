"""Tiny loaders for the two datasets used by the example scripts (not part of the library)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns float32 images flattened to (N, 784) and standardised, plus int64 labels."""
    d = DATA / "mnist"
    if not (d / "train-images-idx3-ubyte.gz").exists():
        raise SystemExit("MNIST not found. Run: uv run python scripts/download_data.py")

    def prep(x):
        return ((x.reshape(len(x), -1).astype(np.float32) / 255.0) - MNIST_MEAN) / MNIST_STD

    xtr = prep(_read_idx(d / "train-images-idx3-ubyte.gz"))
    ytr = _read_idx(d / "train-labels-idx1-ubyte.gz").astype(np.int64)
    xte = prep(_read_idx(d / "t10k-images-idx3-ubyte.gz"))
    yte = _read_idx(d / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return xtr, ytr, xte, yte


def load_text() -> str:
    p = DATA / "tinyshakespeare.txt"
    if not p.exists():
        raise SystemExit("Tiny Shakespeare not found. Run: uv run python scripts/download_data.py")
    return p.read_text()
