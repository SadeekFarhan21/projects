"""Step-time benchmark: smolgrad vs a hand-written numpy MLP vs PyTorch eager (CPU).

The hand-written numpy version computes exactly the same forward, backward and
AdamW update with no graph and no Tensor objects. It separates the gap into
  smolgrad - numpy_manual   = cost of the autograd machinery (Python overhead,
                              extra temporaries, the graph walk)
  numpy_manual - torch      = cost of the kernels themselves (numpy vs ATen,
                              fused ops, allocator, threading)

Usage:
  uv run python scripts/bench.py                # default threading
  uv run python scripts/bench.py --threads 1    # single-threaded BLAS and torch

Implementations are interleaved round-robin so background load hits all of them
alike; we report min (best estimate of intrinsic cost) and median.
Writes results/bench_<threads>.csv and results/bench_<threads>.json.
"""

from __future__ import annotations

import argparse
import os
import sys

# Thread limits must be set before numpy / torch are imported.
if "--threads" in sys.argv:
    n = sys.argv[sys.argv.index("--threads") + 1]
    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = n

import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import smolgrad as sg  # noqa: E402
import smolgrad.functional as F  # noqa: E402
import smolgrad.nn as nn  # noqa: E402
from datasets import RESULTS  # noqa: E402
from train_char import CharMLP  # noqa: E402

DIMS = (784, 512, 256, 10)


# ----------------------------------------------------------------- MLP: smolgrad
class SgMLP:
    def __init__(self):
        sg.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                   nn.Linear(256, 10))
        self.opt = sg.optim.AdamW(self.model.parameters(), lr=1e-3)

    def fwd(self, x, y):
        self.opt.zero_grad()
        self.loss = F.cross_entropy(self.model(sg.Tensor(x)), y)

    def bwd(self):
        self.loss.backward()

    def opt_step(self):
        self.opt.step()


# ------------------------------------------------------- MLP: hand-written numpy
class NumpyMLP:
    """Same math, no autograd. Reuses smolgrad's AdamW so the optimizer cost is
    identical and the comparison isolates the forward/backward machinery."""

    def __init__(self):
        sg.manual_seed(0)
        ref = SgMLP().model
        self.W = [ref.layers[i].weight for i in (0, 2, 4)]
        self.b = [ref.layers[i].bias for i in (0, 2, 4)]
        self.opt = sg.optim.AdamW(self.W + self.b, lr=1e-3)

    def fwd(self, x, y):
        W, b = [w.data for w in self.W], [v.data for v in self.b]
        self.x, self.y = x, y
        self.h1 = np.maximum(x @ W[0].T + b[0], 0)
        self.h2 = np.maximum(self.h1 @ W[1].T + b[1], 0)
        z = self.h2 @ W[2].T + b[2]
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        self.p = e / e.sum(1, keepdims=True)
        self.loss = -np.log(self.p[np.arange(len(y)), y]).mean()

    def bwd(self):
        W = [w.data for w in self.W]
        n = len(self.y)
        dz = self.p.copy()
        dz[np.arange(n), self.y] -= 1
        dz /= n
        self.W[2].grad, self.b[2].grad = dz.T @ self.h2, dz.sum(0)
        d2 = (dz @ W[2]) * (self.h2 > 0)
        self.W[1].grad, self.b[1].grad = d2.T @ self.h1, d2.sum(0)
        d1 = (d2 @ W[1]) * (self.h1 > 0)
        self.W[0].grad, self.b[0].grad = d1.T @ self.x, d1.sum(0)

    def opt_step(self):
        self.opt.step()


# ---------------------------------------------------------------- MLP: PyTorch
class TorchMLP:
    def __init__(self):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(torch.nn.Linear(784, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256),
                                         torch.nn.ReLU(), torch.nn.Linear(256, 10))
        # foreach=True is the CPU default when available; set explicitly for clarity.
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, foreach=True)

    def fwd(self, x, y):
        self.opt.zero_grad(set_to_none=True)
        self.loss = torch.nn.functional.cross_entropy(self.model(torch.from_numpy(x)), torch.from_numpy(y))

    def bwd(self):
        self.loss.backward()

    def opt_step(self):
        self.opt.step()


# ---------------------------------------------------------- char model workload
class SgChar:
    def __init__(self):
        sg.manual_seed(0)
        self.model = CharMLP(65, 16, 32, 512)
        self.opt = sg.optim.AdamW(self.model.parameters(), lr=1e-3)

    def fwd(self, x, y):
        self.opt.zero_grad()
        self.loss = F.cross_entropy(self.model(x), y)

    bwd = SgMLP.bwd
    opt_step = SgMLP.opt_step


class TorchChar:
    def __init__(self):
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(65, 32)
        self.net = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.LayerNorm(512), torch.nn.Tanh(),
                                       torch.nn.Linear(512, 65))
        self.opt = torch.optim.AdamW(list(self.emb.parameters()) + list(self.net.parameters()), lr=1e-3)

    def fwd(self, x, y):
        self.opt.zero_grad(set_to_none=True)
        logits = self.net(self.emb(torch.from_numpy(x)).flatten(1))
        self.loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y))

    bwd = TorchMLP.bwd
    opt_step = TorchMLP.opt_step


def time_phases(impl, x, y, iters):
    """Returns per-phase lists of seconds for `iters` steps."""
    out = {"forward": [], "backward": [], "optimizer": [], "step": []}
    for _ in range(iters):
        t0 = time.perf_counter()
        impl.fwd(x, y)
        t1 = time.perf_counter()
        impl.bwd()
        t2 = time.perf_counter()
        impl.opt_step()
        t3 = time.perf_counter()
        out["forward"].append(t1 - t0)
        out["backward"].append(t2 - t1)
        out["optimizer"].append(t3 - t2)
        out["step"].append(t3 - t0)
    return out


def iters_for(batch):
    return 40 if batch <= 128 else 15


def bench_workload(name, impls, make_batch, batches, rounds, rows):
    for bs in batches:
        x, y = make_batch(bs)
        objs = {k: cls() for k, cls in impls.items()}
        for o in objs.values():  # warmup (allocator, BLAS thread pools, lazy init)
            time_phases(o, x, y, 5)
        acc = {k: {p: [] for p in ("forward", "backward", "optimizer", "step")} for k in objs}
        for _ in range(rounds):
            for k, o in objs.items():
                for p, v in time_phases(o, x, y, iters_for(bs)).items():
                    acc[k][p].extend(v)
        for k in objs:
            for p, v in acc[k].items():
                rows.append(dict(workload=name, impl=k, batch=bs, phase=p, n=len(v),
                                 min_ms=1000 * min(v), median_ms=1000 * float(np.median(v))))
        med = {k: 1000 * float(np.median(acc[k]["step"])) for k in objs}
        mn = {k: 1000 * min(acc[k]["step"]) for k in objs}
        print(f"{name:5s} bs={bs:5d} " + "  ".join(f"{k}: med {med[k]:7.3f} ms (min {mn[k]:7.3f})" for k in objs),
              flush=True)


def overhead_microbench(rows, n_ops=2000, reps=15):
    """Per-op cost on 1-element tensors, where the arithmetic is negligible and
    everything measured is framework overhead: one forward op plus its backward."""
    res = {}
    for impl in ("smolgrad", "torch"):
        ts = []
        for _ in range(reps):
            if impl == "smolgrad":
                x = sg.Tensor(np.ones(1, dtype=np.float32), requires_grad=True)
            else:
                x = torch.ones(1, requires_grad=True)
            t0 = time.perf_counter()
            y = x
            for _ in range(n_ops):
                y = y * 1.0001
            y.sum().backward()
            ts.append((time.perf_counter() - t0) / n_ops)
        res[impl] = ts
        rows.append(dict(workload="overhead_scalar_mul", impl=impl, batch=1, phase="fwd+bwd per op", n=reps,
                         min_ms=1000 * min(ts), median_ms=1000 * float(np.median(ts))))
        print(f"per-op overhead {impl}: min {1e6 * min(ts):.2f} us, median {1e6 * float(np.median(ts)):.2f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="default")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    if args.threads != "default":
        torch.set_num_threads(int(args.threads))

    rng = np.random.default_rng(0)
    load_before = os.getloadavg()
    rows: list[dict] = []

    def mnist_batch(bs):
        return rng.standard_normal((bs, 784)).astype(np.float32), rng.integers(0, 10, size=bs)

    def char_batch(bs):
        return rng.integers(0, 65, size=(bs, 16)), rng.integers(0, 65, size=bs)

    bench_workload("mlp", {"smolgrad": SgMLP, "numpy_manual": NumpyMLP, "torch": TorchMLP}, mnist_batch,
                   [1, 32, 128, 512, 2048], args.rounds, rows)
    bench_workload("char", {"smolgrad": SgChar, "torch": TorchChar}, char_batch, [256], args.rounds, rows)
    overhead_microbench(rows)

    meta = dict(threads=args.threads, torch_threads=torch.get_num_threads(), rounds=args.rounds,
                numpy=np.__version__, torch=torch.__version__, python=platform.python_version(),
                machine=platform.machine(), cpu_count=os.cpu_count(),
                loadavg_before=load_before, loadavg_after=os.getloadavg())
    RESULTS.mkdir(exist_ok=True)
    tag = f"bench_threads_{args.threads}"
    with open(RESULTS / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / f"{tag}.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
