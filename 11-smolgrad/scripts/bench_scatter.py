"""Microbenchmark of three ways to compute the embedding backward (scatter-add
of rows): np.add.at, one-hot matmul, and sort plus np.add.reduceat.

Usage: uv run python scripts/bench_scatter.py
Writes results/scatter_add.csv.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from datasets import RESULTS


def add_at(idx, g, V):
    z = np.zeros((V, g.shape[1]), g.dtype)
    np.add.at(z, idx, g)
    return z


def onehot_matmul(idx, g, V):
    oh = np.zeros((len(idx), V), g.dtype)
    oh[np.arange(len(idx)), idx] = 1
    return oh.T @ g


def sort_reduceat(idx, g, V):
    order = np.argsort(idx, kind="stable")
    s = idx[order]
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    z = np.zeros((V, g.shape[1]), g.dtype)
    z[s[starts]] = np.add.reduceat(g[order], starts, axis=0)
    return z


def main():
    rng = np.random.default_rng(0)
    rows = []
    # (V, N, D): the char model's shape, a mid-size vocab, a GPT-2-size vocab.
    for V, N, D in [(65, 4096, 32), (1000, 8192, 64), (50257, 8192, 128)]:
        idx = rng.integers(0, V, N)
        g = rng.standard_normal((N, D)).astype(np.float32)
        ref = add_at(idx, g, V)
        for f in (add_at, onehot_matmul, sort_reduceat):
            if f is onehot_matmul and V * N > 1e8:
                continue  # the one-hot matrix would be > 1.6 GB
            ts = []
            for _ in range(50):
                t = time.perf_counter()
                out = f(idx, g, V)
                ts.append(time.perf_counter() - t)
            row = dict(V=V, N=N, D=D, method=f.__name__, min_us=1e6 * min(ts),
                       median_us=1e6 * float(np.median(ts)), max_abs_err=float(np.abs(out - ref).max()),
                       loadavg_1min=os.getloadavg()[0])
            rows.append(row)
            print(row, flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "scatter_add.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
