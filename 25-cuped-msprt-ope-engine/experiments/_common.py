"""Shared helpers for experiment scripts: paths, JSON output and a quiet plot style."""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RAW = RESULTS / "raw"
for d in (RESULTS, FIGURES, RAW):
    d.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2f6690", "#d1495b", "#3a7d44", "#edae49", "#6c757d", "#8e5572", "#00798c"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def write_json(name: str, obj: dict) -> Path:
    obj = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": platform.machine(), **obj}
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=float))
    return path


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    for bad in (":", "—", "–"):
        assert bad not in title + xlabel + ylabel, f"no {bad!r} in plot text"
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def save(fig, name: str) -> Path:
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
