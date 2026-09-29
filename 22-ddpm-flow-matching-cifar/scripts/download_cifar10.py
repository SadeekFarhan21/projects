"""Download CIFAR-10 (python version, about 170 MB) into data/ and cache it as a uint8 .npz.

Usage: uv run python scripts/download_cifar10.py
"""
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    out = ROOT / "cifar10.npz"
    if out.exists():
        print(f"already cached at {out}")
        return
    train = CIFAR10(ROOT, train=True, download=True)
    test = CIFAR10(ROOT, train=False, download=True)
    np.savez(
        out,
        x_train=train.data.transpose(0, 3, 1, 2),  # N,3,32,32 uint8
        y_train=np.array(train.targets, dtype=np.int64),
        x_test=test.data.transpose(0, 3, 1, 2),
        y_test=np.array(test.targets, dtype=np.int64),
    )
    print(f"wrote {out}: train {train.data.shape}, test {test.data.shape}")


if __name__ == "__main__":
    main()
