"""cProfile of 200 char-model training steps, with the original np.add.at
embedding backward and with the current one.

Usage: uv run python scripts/profile_step.py
Writes results/profile_char_step.txt.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats

import numpy as np

import smolgrad as sg
import smolgrad.functional as F
from datasets import RESULTS
from train_char import CharMLP


def add_at_scatter(idx, g, V):
    z = np.zeros((V, g.shape[1]), g.dtype)
    np.add.at(z, idx, g)
    return z


def profile(label, steps=200):
    sg.manual_seed(0)
    m = CharMLP(65, 16, 32, 512)
    opt = sg.optim.AdamW(m.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)

    def step():
        x, y = rng.integers(0, 65, (256, 16)), rng.integers(0, 65, 256)
        opt.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()

    for _ in range(20):
        step()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(steps):
        step()
    pr.disable()
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("tottime")
    total = st.total_tt
    st.print_stats(10)
    return f"=== {label}: {steps} steps, {1000 * total / steps:.2f} ms/step under the profiler, " \
           f"loadavg {os.getloadavg()[0]:.1f} ===\n" + buf.getvalue()


def main():
    current = F._scatter_add_rows
    F._scatter_add_rows = add_at_scatter
    before = profile("embedding backward with np.add.at")
    F._scatter_add_rows = current
    after = profile("embedding backward with one-hot matmul (current)")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "profile_char_step.txt").write_text(before + "\n" + after)
    print(before + "\n" + after)


if __name__ == "__main__":
    main()
