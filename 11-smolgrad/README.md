# smolgrad: autograd and a neural network library from scratch

smolgrad is a reverse-mode automatic differentiation engine and a small
PyTorch-style neural network library written in about 900 lines of Python on
top of numpy. It builds a dynamic computation graph on every forward pass,
walks it in reverse topological order to compute gradients, and ships the
pieces needed to train real models: broadcasting-aware ops, a module system,
SGD with momentum and AdamW. Every op is checked against finite differences
and against PyTorch.

It trains a 784-512-256-10 MLP on MNIST to 97.9% test accuracy and a
character-level model on Tiny Shakespeare to 1.79 nats per character on held-out
text. See DEVLOG.md for the full numbers and where each one came from.

## Layout

```
src/smolgrad/
  tensor.py       Tensor, the graph, backward(), all primitive ops
  functional.py   linear, layer_norm (composed); embedding, cross_entropy (fused)
  nn.py           Module, Linear, Embedding, LayerNorm, Sequential, ReLU, Tanh, Flatten
  optim.py        SGD (momentum, weight decay), AdamW
  gradcheck.py    central-difference gradient checker
  random.py       library-wide seeded RNG
tests/
  test_gradcheck.py     finite-difference check of every op (58 cases plus a checker self-test)
  test_engine.py        engine semantics: accumulation, aliasing, no_grad, deep graphs
  test_parity_torch.py  values, grads, modules and optimizer trajectories vs PyTorch
scripts/
  download_data.py      MNIST (public mirror, SHA-256 checked) and Tiny Shakespeare (not checked)
  train_mnist.py        MLP on MNIST, optional PyTorch twin run
  train_char.py         char-level model on Tiny Shakespeare
  divergence.py         how fast smolgrad and PyTorch drift apart in float32 vs float64
  bench.py              step time vs PyTorch CPU and vs hand-written numpy
  plot_bench.py         plots for the benchmark
  bench_scatter.py      embedding backward: np.add.at vs one-hot matmul vs sort+reduceat
  profile_step.py       cProfile of a char-model training step
  run_bench_when_quiet.sh  waits for a low load average, then runs both benchmarks and plots
results/                raw outputs (CSV, JSON, TXT, PNG) of every run quoted in the docs
```

## Build

Needs [uv](https://docs.astral.sh/uv/). Python is pinned to 3.12 in `.python-version`.

```
uv sync
```

The library itself depends only on numpy. PyTorch and matplotlib are dev
dependencies, used as a test oracle, a benchmark baseline and for plots.

## Use

```python
import numpy as np
import smolgrad as sg
from smolgrad import nn, functional as F

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = sg.optim.AdamW(model.parameters(), lr=1e-3)

x = np.random.randn(32, 784).astype(np.float32)
y = np.random.randint(0, 10, size=32)
opt.zero_grad()
loss = F.cross_entropy(model(sg.Tensor(x)), y)
loss.backward()
opt.step()
```

## Test

```
uv run pytest -q
```

## Run the experiments

```
uv run python scripts/download_data.py                 # about 13 MB into data/ (gitignored)
uv run python scripts/train_mnist.py --torch-twin      # 10 epochs; took 321 s under heavy shared load
uv run python scripts/train_char.py                    # 6000 steps; took 478 s under heavy shared load
uv run python scripts/divergence.py                    # float32 vs float64 twin drift
```

## Benchmark

```
uv run python scripts/bench.py --threads 1
uv run python scripts/bench.py
uv run python scripts/plot_bench.py
```

On a shared machine, `scripts/run_bench_when_quiet.sh` waits until the load
average drops below `LOAD_MAX` (default 20) and then runs all three. The
numbers in `results/` were taken at a load average of 135 to 360, so treat the
absolute times as upper bounds; the single-threaded ratios are the useful part.

The benchmark times a full training step (forward, backward, AdamW) for the
MNIST MLP at batch sizes 1 to 2048 in three implementations: smolgrad, the same
network hand-written in numpy with no autograd, and PyTorch eager on CPU. The
hand-written version splits the gap to PyTorch into "autograd overhead" and
"kernel speed". Results and the explanation are in DEVLOG.md (Results).

## Status

| milestone | status |
|---|---|
| v0: Tensor with reverse-mode autodiff over a dynamic graph, broadcasting-aware grads | done |
| v0: ops add, sub, mul, div, pow, matmul, sum/mean/max, reshape, transpose, exp, log, relu, tanh, softmax, log_softmax, indexing, embedding, cross-entropy | done |
| v0: Module, Linear, Embedding, LayerNorm, Sequential; SGD with momentum, AdamW | done |
| v0: finite-difference gradcheck for every op, parity tests against PyTorch | done |
| v0: MNIST MLP and char-level model trained, benchmark vs PyTorch CPU | done |
| v1: higher-order gradients (backward rules written in Tensor ops, `create_graph=True`) | planned |
| v2: lazy graph plus a fusion pass (elementwise chains into one kernel, fused LayerNorm) | planned |
| v3: C++ or Metal backend behind the same Tensor API | planned |
| v4: run project 12 on top of smolgrad | planned |

### Cut from v0

Nothing from the v0 list was cut. Things deliberately left out because they
were never in scope: dropout, convolutions, a DataLoader, a version counter for
in-place safety, `retain_grad` on intermediate tensors, and a float16 path.
These are documented in DESIGN.md (Trade-offs).
