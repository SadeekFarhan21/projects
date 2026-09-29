# DEVLOG: building smolgrad, autograd and a neural network library from scratch

### What I wanted to build

I wanted to understand what actually happens inside `loss.backward()` by
writing it myself, and to hold the result to a real standard rather than a toy
one. The goals for v0:

- a `Tensor` with reverse-mode autodiff over a graph built on the fly each
  forward pass, with broadcasting handled correctly in the backward pass;
- enough ops to train real models: add, mul, matmul, sum and mean over axes,
  reshape, transpose, exp, log, relu, tanh, softmax, log_softmax, indexing and
  embedding gathers, and cross-entropy;
- a PyTorch-shaped module system (Module, Linear, Embedding, LayerNorm,
  Sequential) and two optimizers (SGD with momentum, AdamW);
- proof that it is right: finite-difference gradient checks for every op and
  parity tests against PyTorch, down to whole optimizer trajectories;
- proof that it is useful: an MLP trained on MNIST and a character-level model
  trained on Tiny Shakespeare;
- an honest benchmark against PyTorch on CPU that explains where the gap comes
  from instead of just reporting a ratio.

The only runtime dependency is numpy. PyTorch appears only as a test oracle and
as the benchmark baseline.

### Theory

Reverse-mode autodiff computes the gradient of one scalar output with respect
to every input in a single backward sweep whose cost is a small constant times
the forward cost. Every op `y = f(x1, ..., xk)` only needs to know its
vector-Jacobian product: given `dL/dy`, return `dL/dxi` for each input. The
engine does the bookkeeping:

1. Record each op's inputs and a closure that computes its VJP.
2. Topologically sort the graph from the loss.
3. Walk it in reverse, seeding `dL/dL = 1`, calling each closure and summing
   the gradients that arrive at a node from all of its consumers. The sum is
   the multivariate chain rule: a node used twice gets two contributions.

The VJPs I needed, written for a gradient `G` flowing in:

- `C = A @ B`: `dA = G @ B^T`, `dB = A^T @ G`.
- broadcasting: the forward copies a value along an axis, so the backward sums
  along that axis. Every binary op passes its gradient through
  `_unbroadcast`, which sums away prepended axes and stretched size-1 axes.
- `sum` over axes: broadcast `G` back to the input shape. `mean` is sum times
  a constant.
- `softmax`: `dx = s * (G - sum(G * s))`, which never forms the C by C
  Jacobian.
- `log_softmax`: `dx = G - softmax(x) * sum(G)`.
- fused cross-entropy with mean reduction: `dlogits = (softmax - onehot) / N`.
- gather `W[idx]`: scatter-add `G` back into the rows of `W`. Repeated indices
  must accumulate, which is the whole difficulty of the embedding backward.

Gradient checking uses central differences, `(f(x + h) - f(x - h)) / 2h`, whose
error is `O(h^2)`. I run it in float64 with `h = 1e-6`, and reduce a
non-scalar output to a scalar through a fixed random projection so that every
output element contributes to the check.

### Architecture

```
  scripts (train_mnist, train_char, bench, divergence)
        |                  |                    |
        v                  v                    v
  smolgrad.nn        smolgrad.optim      smolgrad.gradcheck
  Module, Linear,    SGD, AdamW          central differences
  Embedding,         (mutate p.data      under no_grad()
  LayerNorm,          in place)
  Sequential
        |
        v
  smolgrad.functional: linear, layer_norm (composed),
                       embedding, cross_entropy (fused)
        |
        v
  smolgrad.tensor.Tensor: data, grad, requires_grad, _parents, _backward
        |   forward op -> Tensor._make(out, parents, closure)
        |   backward() -> iterative DFS topo sort -> reverse sweep with
        |                 a gradient dict keyed by id(node) -> leaf .grad
        v
  numpy (Accelerate BLAS on macOS)
```

The value and the graph node are the same object. Each op builds its output
through `Tensor._make`, which only records parents and a backward closure when
grad mode is on and some parent requires grad, so evaluation under `no_grad()`
builds no graph at all. During `backward()` intermediate gradients live in a
dict local to that call and are popped as soon as a node is processed; only
leaves get a `.grad`, and it accumulates until zeroed, like PyTorch. DESIGN.md
has the data structures, the seven invariants, and the alternatives I rejected
(a Function class per op, a tape, per-node `.grad`, version counters).

### Implementation

The library is about 900 lines in `src/smolgrad/`:

- `tensor.py`: the Tensor, the engine, and every primitive op. The topological
  sort is an iterative DFS, so graph depth is limited by memory and not by
  Python's recursion limit.
- `functional.py`: `linear` and `layer_norm` are compositions of primitives
  and get their gradients from the engine for free. `cross_entropy` and
  `embedding` are fused primitives with hand-written backward passes, the
  first for numerical stability and speed, the second because a fast
  scatter-add cannot be expressed with the other ops.
- `nn.py`: Module discovers parameters by walking `vars(self)` in insertion
  order, which gives a deterministic parameter order. Layouts and inits match
  PyTorch (`Linear.weight` is `(out, in)`), so tests can copy weights across
  without transposes.
- `optim.py`: SGD with PyTorch's momentum convention and AdamW with decoupled
  weight decay. AdamW uses one preallocated scratch buffer per parameter and
  `out=` numpy calls, so a step allocates nothing.
- `gradcheck.py`: the finite-difference checker, with its own self-test that
  it catches a deliberately wrong backward.

The tests are split three ways: 59 finite-difference checks (every op,
including broadcasting variants, batched matmul, reductions over several
axes, indexing with repeated indices), 19 engine-semantics tests (diamonds,
accumulation, aliasing, `no_grad`, dtype preservation, a 20,000-op chain), and
23 PyTorch parity tests (op values and grads, modules, and multi-step SGD and
AdamW trajectories).

The two models:

- MNIST: a 784-512-256-10 ReLU MLP, 535,818 parameters, AdamW with lr 1e-3
  and weight decay 0.01, batch 128, 10 epochs. `--torch-twin` trains an
  identical PyTorch model from the same initial weights on the same batches.
- Tiny Shakespeare: a character MLP in the style of Bengio et al. 2003. An
  Embedding(65, 32) over a 16-character context, flattened, then
  Linear(512, 512), LayerNorm, tanh, Linear(512, 65). 299,105 parameters,
  AdamW with lr 3e-3 and a linear decay over the last quarter, batch 256,
  6,000 steps.

### Problems

This list is only the issues that left evidence in the code, tests or
results/ files.

1. Leaf gradients aliasing each other. `add` returns the same gradient array
   for both parents, and `sum`'s backward returns a read-only
   `np.broadcast_to` view. Storing those directly as `.grad` meant two
   parameters could share one array, so an in-place update to one would
   corrupt the other, and in-place accumulation into a broadcast view fails
   outright. The fix: a leaf copies an incoming gradient if it is a view, not
   writeable, or was already handed to another leaf in the same call, and
   adopts it without a copy otherwise. The caller's seed gradient is always
   copied. Regression tests: `test_leaf_grads_do_not_alias` and
   `test_seed_gradient_is_not_aliased`.

2. Silent float64 promotion. `Tensor(float32) * 0.5` must stay float32, and
   so must optimizer math. Wrapping Python scalars in a float64 array would
   have promoted whole activations. Constants are now wrapped in the
   tensor's own dtype (`Tensor._const`), and AdamW converts its bias
   correction to a Python float before multiplying. Guarded by
   `test_float32_stays_float32_with_python_scalars`.

3. Recursion limit. A recursive topological sort breaks on long chains
   (Python's default limit is 1,000 frames). I replaced it with an iterative
   DFS with an explicit stack; `test_deep_graph_does_not_recurse` runs a
   20,000-op chain.

4. `np.add.at` was the most expensive call in a char-model step. Profiling
   200 training steps (`results/profile_char_step.txt`) showed `ufunc.at` at
   1.117 s out of 4.757 s total, more than the AdamW step. `np.add.at` is
   correct for repeated indices but runs an unbuffered per-element loop. I
   benchmarked two replacements in `scripts/bench_scatter.py`
   (`results/scatter_add.csv`): a one-hot matrix product
   `onehot(idx)^T @ G`, which is a single BLAS call, and sort plus
   `np.add.reduceat`. For the char model's shape (V=65, N=4,096, D=32) the
   minimum times were 1,272 us for `add.at`, 72 us for one-hot and 484 us for
   sort and reduceat. The one-hot matrix grows as N times V, so for a
   GPT-2-sized vocab (V=50,257) it is skipped, and sort and reduceat wins
   (9,399 us vs 12,530 us for `add.at`). `_scatter_add_rows` now picks one-hot
   when N times V is at most 2^20 and sort and reduceat otherwise, and a test
   checks both paths against `np.add.at`. The profiled step went from 23.78
   ms to 11.51 ms (`results/profile_char_step.txt`).

5. AdamW allocations. Written as numpy expressions, an AdamW step created
   about six parameter-sized temporaries per tensor. The optimizer is a large
   share of a step for a 535k-parameter MLP at small batch, so I rewrote it
   with a scratch buffer and `out=` arguments.

6. float32 runs drift away from PyTorch, which initially looked like a bug.
   After 10 MNIST epochs the max absolute weight difference between smolgrad
   and the PyTorch twin was 0.313 (`results/mnist_train.csv`), even though
   every parity test passed. To tell a bug from rounding, I reran the twin in
   float64 for 3 epochs: max weight difference 5.3e-14
   (`results/mnist_summary_f64_twin.json`). `scripts/divergence.py` then
   measured the drift per step (`results/divergence.csv`): in float64 it stays
   at or below 1.5e-14 for 300 steps with both optimizers, while in float32 it
   starts at the rounding level (3.7e-9 after one SGD step) and grows
   chaotically to 2.6e-2 (SGD) and 6.7e-2 (AdamW) by step 300. Different
   summation orders in the BLAS and reduction kernels are enough; training is
   a chaotic map. AdamW starts noisier (8.3e-6 after one step) because
   `m / sqrt(v)` normalises tiny rounding differences in near-zero gradients
   up to the full step size.

7. The machine was shared. Up to 14 other agents were building projects on
   the same M4 Pro during this session, and the 1-minute load average sat
   between roughly 135 and 360. This affects every timing here. I interleave
   implementations round-robin in the benchmark so the load hits them alike,
   report the minimum over runs as the best estimate of intrinsic cost, and
   record the load average in every results file. The medians are not
   meaningful, and the default-threads benchmark in particular is unreliable
   (see Results).

### Experiments

All commands are run from the project root with `uv run`.

1. Correctness: `uv run pytest -q`. 101 tests (59 gradcheck, 19 engine, 23
   PyTorch parity).
2. MNIST: `scripts/train_mnist.py --torch-twin`, float32, 10 epochs, with a
   PyTorch twin from identical weights. Output: `results/mnist_train.csv`,
   `results/mnist_summary.json`, `results/mnist_curves.png`. A float64 twin
   run for 3 epochs: `results/mnist_train_f64_twin.csv`,
   `results/mnist_summary_f64_twin.json`.
3. Drift: `scripts/divergence.py`, 300 steps for each of {float32, float64}
   x {SGD with momentum, AdamW}. Output: `results/divergence.csv`,
   `results/divergence.png`.
4. Char model: `scripts/train_char.py`, 6,000 steps, compared with uniform,
   unigram and bigram baselines on the same validation split. Output:
   `results/char_train.csv`, `results/char_summary.json`,
   `results/char_samples.txt`, `results/char_curves.png`.
5. Step-time benchmark: `scripts/bench.py --threads 1` and `scripts/bench.py`
   (default threading). A full training step (forward, backward, AdamW) of the
   MNIST MLP at batch 1, 32, 128, 512 and 2048, plus the char model at batch
   256, in three implementations: smolgrad, the same network hand-written in
   numpy with manual gradients, and PyTorch eager on CPU. The hand-written
   version splits the gap: smolgrad minus numpy is the cost of the autograd
   machinery, numpy minus PyTorch is the cost of the kernels. It also times a
   per-op overhead microbenchmark (2,000 chained scalar multiplies, forward
   plus backward). Output: `results/bench_threads_1.{csv,json}`,
   `results/bench_threads_default.{csv,json}`, plots
   `results/bench_step_time.png` and `results/bench_phases.png`.
6. Embedding backward microbenchmark: `scripts/bench_scatter.py`, output
   `results/scatter_add.csv`, and the profile in `scripts/profile_step.py`,
   output `results/profile_char_step.txt`.

### Results

Correctness. All 101 tests pass (`uv run pytest -q`, 16.9 s under load).

MNIST (`results/mnist_summary.json`, `results/mnist_train.csv`):

| | smolgrad | PyTorch twin |
|---|---|---|
| final test accuracy, epoch 10 | 97.89% | 97.81% |
| best test accuracy | 98.11% (epoch 5) | 98.08% (epoch 3) |
| final test loss | 0.0863 | 0.0946 |

The test loss bottoms out at epoch 5 (0.0623) and rises afterwards while train
loss keeps falling to 0.0184, which is ordinary overfitting with no
regularisation beyond weight decay. The float64 twin run reaches 97.75% for
both after 3 epochs with weights identical to 5.3e-14
(`results/mnist_summary_f64_twin.json`), which is the strongest end-to-end
evidence that the gradients and the optimizer are right.

Char model (`results/char_summary.json`, `results/char_train.csv`):

| model | validation loss (nats per char) |
|---|---|
| uniform over 65 chars | 4.174 |
| unigram | 3.347 |
| bigram | 2.482 |
| smolgrad char MLP, step 6,000 | 1.794 |

Train loss at the end was 1.576, so the model is starting to overfit the 16
character windows. Samples (`results/char_samples.txt`) have the play format,
speaker names like "KING RICHARD III:" and mostly real short words, with
invented longer ones, which is about what a 300k-parameter model with a
16-character window should manage.

Step time vs PyTorch CPU, single-threaded (`results/bench_threads_1.csv`,
min over 320 runs for batch up to 128, 120 above):

| MLP batch | smolgrad ms | numpy by hand ms | PyTorch ms | smolgrad / PyTorch |
|---|---|---|---|---|
| 1 | 3.21 | 2.32 | 1.29 | 2.5x |
| 32 | 2.43 | 1.62 | 1.32 | 1.8x |
| 128 | 2.94 | 2.09 | 1.66 | 1.8x |
| 512 | 5.01 | 4.97 | 3.00 | 1.7x |
| 2048 | 21.2 | 38.1 | 11.6 | 1.8x |

The char model step at batch 256 was 3.40 ms for smolgrad and 2.77 ms for
PyTorch (1.2x), same file.

Where the gap comes from, batch 128, single-threaded (`results/bench_threads_1.csv`,
plotted in `results/bench_phases.png`):

| phase | smolgrad ms | numpy by hand ms | PyTorch ms |
|---|---|---|---|
| forward | 0.42 | 0.36 | 0.31 |
| backward | 1.15 | 0.38 | 0.35 |
| AdamW step | 1.33 | 1.32 | 1.00 |

My reading:

- The autograd machinery costs about 0.85 ms per step at batch 128 (2.94
  minus 2.09), and almost all of it is in backward (1.15 vs 0.38 ms). That is
  the graph walk plus extra temporaries: `_unbroadcast` on every bias add, the
  relu mask multiply allocating a new array, the gradient dict, and copies
  where a leaf cannot adopt its gradient. The hand-written version fuses
  these or does them in place.
- The kernel gap is smaller, about 0.43 ms (2.09 minus 1.66), and is mostly
  the optimizer: PyTorch's AdamW uses a fused multi-tensor kernel, while mine
  does about ten numpy passes over each parameter. For a 535k-parameter model
  at batch 128, AdamW is 45% of the smolgrad step, independent of batch size.
- Per-op Python overhead is not the problem. On 1-element tensors, a forward
  plus backward multiply costs 6.2 us in smolgrad vs 12.6 us in PyTorch
  (`results/bench_threads_1.csv`, `overhead_scalar_mul` rows; 6.2 vs 10.0 us in
  `results/bench_threads_default.csv`). PyTorch's dispatcher does more work per
  op than a Python closure does. The gap on real workloads comes from memory
  traffic, not dispatch.
- The batch 2048 row where the hand-written numpy (38.1 ms) is slower than
  smolgrad (21.2 ms) is noise from background load, not a real effect: the two
  run the same matmuls. The default-threads file
  (`results/bench_threads_default.csv`, 40 and 15 runs) is worse, with PyTorch
  slower than hand-written numpy at batch 128, so I do not draw conclusions
  from it beyond the per-op overhead numbers.
- Training wall-clock is dominated by load in this session. The MNIST run
  took 19.2 s for epoch 1 and 4.5 s for epoch 10 as the load changed
  (`results/mnist_train.csv`), and the char run averaged 71 ms per step
  (`results/char_summary.json`) against a benchmark minimum of 3.4 ms.

Embedding backward (`results/scatter_add.csv`, `results/profile_char_step.txt`):
replacing `np.add.at` with the one-hot product made the scatter 17.6x faster
at the char model's shape (1,272 us to 72 us) and halved the profiled step
(23.78 to 11.51 ms).

### What I would change

- Record ops, not closures. Closures made v0 short and readable but make the
  graph opaque, so there is no way to fuse or rewrite it. The lazy fusion
  milestone needs an explicit op record (op type, inputs, attributes), and I
  would switch to that first.
- Fuse the obvious hot spots before writing any backend: a fused
  bias-plus-relu with an in-place mask in backward, a fused LayerNorm
  forward and backward, and a multi-tensor AdamW over one flat parameter
  buffer. The phase table says these are where the 1.8x lives.
- Add a version counter. Invariant 5 in DESIGN.md (do not modify parameters
  between forward and backward) is enforced only by convention.
- Write backward rules in Tensor ops behind `create_graph=True` to get
  higher-order gradients, which is the next milestone.
- Rerun every timing on an idle machine. The benchmark code already records
  the load average and interleaves implementations, but the numbers in this
  log were taken at a load average of 135 to 360 and should be read as upper
  bounds with roughly correct ratios for the single-threaded case.

### Verification

Independent verification on 2026-09-26 (evening, US Eastern, finishing just after midnight), by a reviewer who did not build the project. Threads were capped at 2 (OMP, OpenBLAS, vecLib, MKL) because the machine was shared. All reruns wrote to a scratch folder, so the committed `results/` files were not overwritten.

What I reproduced:

- Clean state: removed `__pycache__` and `.pytest_cache`, kept the uv environment, ran `uv sync` and `uv run pytest -q`. 101 passed (59 gradcheck, 19 engine, 23 PyTorch parity), none skipped.
- `scripts/download_data.py` found all five files cached. The four MNIST files passed their SHA-256 checks.
- MNIST float64 twin (`--torch-twin --dtype float64 --epochs 3`): 97.75% test accuracy for both smolgrad and PyTorch, and every loss agrees with `results/mnist_train_f64_twin.csv` to about 1e-16. The final weight difference was 3.9e-14 against the committed 5.3e-14, both at rounding level.
- Char model, full 6,000 steps: validation loss 1.79406 and train loss 1.57645, matching `results/char_summary.json` to 1e-7. The baselines (4.174, 3.347, 2.482) match exactly.
- Drift (`scripts/divergence.py`, 300 steps): float64 stays at or below 1.5e-14 for both optimizers, and float32 SGD matches `results/divergence.csv` exactly (3.7e-9 after one step, 2.6e-2 at step 300). float32 AdamW matches at step 1 (8.3e-6) but ended at 5.1e-2 instead of 6.7e-2 at step 300. That is expected for the chaotic float32 drift that Problem 6 describes.
- Every number in Experiments and Results appears in a `results/` file.

Discrepancy:

- The float32 MNIST run is deterministic for a fixed thread count (two of my runs agreed exactly for 3 epochs), but it does not reproduce the committed trajectory. Mine reached 98.07% final and 98.17% best test accuracy, with test loss 0.0778, against 97.89%, 98.11% and 0.0863 in `results/mnist_summary.json`. The first epoch already differs (96.96% against 96.77%), so the difference is BLAS summation order under a different thread count, not a bug. The float64 twin shows that the gradients are right. The headline "97.9%" should be read as one draw from a spread of about 0.2 points.

Review of the correctness claims:

- The gradient checks compare each backward rule against central differences of the forward pass in float64 with a random output projection. The checker's own self-test shows it catches a backward that is off by a factor of 2.
- The PyTorch parity tests build separate torch leaves from the same data and compare values and gradients at rtol 1e-10. The module and optimizer trajectory tests copy weights into independent torch modules. PyTorch is a genuinely independent reference here.
- I also checked that the hand-written numpy baseline in `scripts/bench.py` computes the same thing as smolgrad. On one batch of 64 the losses agree and the largest gradient difference is 1.1e-8 (float32), so the autograd-overhead split in Results compares like with like.

What I deferred:

- Timing. I did not rerun the benchmarks for timing. `scripts/bench.py --threads 1 --rounds 1` ran cleanly into a scratch folder. For scale, at a load average near 9 my MNIST epoch took about 1.0 s against 4.5 to 19.2 s in `results/mnist_train.csv`, and the 6,000 step char run took 17 s against 478 s. The step-time table and the `bench_scatter.py` and `profile_step.py` numbers need a quiet machine.

Fixes made:

- README said the download script checksums both datasets. Only the MNIST files have SHA-256 checks, so the layout line now says that.

Open issues:

- The "best test accuracy" row picks the best epoch on the test set. It is labeled as such and the headline uses the final epoch, but a separate validation split would be cleaner.
