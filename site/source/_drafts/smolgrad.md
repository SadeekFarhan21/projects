---
layout: post
title: "smolgrad, a NumPy Autograd Engine Checked Against PyTorch"
code: https://github.com/SadeekFarhan21/projects/tree/main/smolgrad
tags:
  - autograd
  - numpy
  - pytorch
description: >-
  smolgrad is a small reverse-mode autodiff engine and neural network library.
  In float64 it tracks PyTorch to 13 decimal places across three epochs of
  MNIST.
date: 2026-09-29 04:01:45
---

I wrote smolgrad, a reverse-mode automatic differentiation engine<sup>[[1]](#ref-1)</sup> with a small PyTorch-shaped<sup>[[2]](#ref-2)</sup> neural network library on top, in 891 lines of Python over NumPy<sup>[[3]](#ref-3)</sup>. It builds a dynamic graph on every forward pass, walks it in reverse topological order, and ships the pieces needed to train real models, which are broadcasting-aware ops, a module system, SGD with momentum and AdamW<sup>[[4]](#ref-4)</sup>.

The number I trust most is not an accuracy. Trained side by side with a PyTorch twin from identical weights on identical batches, in float64, the two models finish three MNIST<sup>[[5]](#ref-5)</sup> epochs with a maximum weight difference of **5.3e-14**, and an independent rerun landed at 3.9e-14. That is float64 rounding territory, and it leaves no room for a bug in any gradient or optimizer update on that path. In float32 the same experiment ends ten epochs **0.313** apart, and most of this post's middle is about why that is not a bug.

With that established, the models are almost a formality. A 784-512-256-10 MLP reaches **97.89%** MNIST test accuracy after ten float32 epochs (the PyTorch twin reaches 97.81%), which is one draw from a spread of about 0.2 points that depends on the BLAS thread count, and a character-level model reaches **1.794 nats per character** on held-out Tiny Shakespeare<sup>[[6]](#ref-6)</sup> against 2.482 for a bigram baseline. A full training step is **1.2x to 2.5x slower than PyTorch** on one CPU thread, and a hand-written NumPy baseline splits that gap into autograd overhead and kernel speed. Every timing was taken on a machine loaded by other jobs, so the absolute times are upper bounds and the single-threaded ratios are the useful part.

*Reading note.* The argument is that an autograd engine is short, and that almost all of the difficulty is in proving it right and in the handful of places where NumPy's semantics and the chain rule disagree about who owns an array. Skip to [Problems](#problems) for the bugs.

## What I Wanted to Build

I wanted to know what actually happens inside `loss.backward()` by writing it, and to hold the result to a standard that a toy would not survive. A scalar autograd in the style of micrograd<sup>[[7]](#ref-7)</sup> is an afternoon. A tensor autograd that trains the same networks PyTorch does, to the same numbers, is a different object, because broadcasting, reductions over several axes, repeated indices and dtype promotion all have gradient rules that are easy to get almost right.

The first goal for v0 was a `Tensor` with reverse-mode autodiff over a graph rebuilt every forward pass, and enough ops to train real models, from broadcasting arithmetic and batched matmul through reductions over arbitrary axes, indexing, softmax and cross-entropy. On top of it I wanted a module system (`Module`, `Linear`, `Embedding`, `LayerNorm`, `Sequential`) with SGD with momentum and AdamW. Then I wanted proof that it is right, meaning a finite-difference check for every op and parity tests against PyTorch down to whole optimizer trajectories, and proof that it is useful, meaning a trained MNIST MLP and a trained character model. Last, I wanted a benchmark against PyTorch that explains where the gap comes from, not just how big it is.

The only runtime dependency is NumPy. PyTorch appears only as a test oracle and as the benchmark baseline.

## Theory

### Reverse Mode in One Paragraph

A training loss is one scalar $L$ computed from a few million inputs. Forward-mode differentiation would need one pass per input to get $\partial L / \partial \theta$. Reverse mode gets all of them in a single backward sweep whose cost is a small constant multiple of the forward pass<sup>[[1]](#ref-1)</sup>. The trick is that each op $y = f(x_1, \dots, x_k)$ never needs its full Jacobian. It needs only its vector-Jacobian product (VJP), a function that takes $\bar{y} = \partial L / \partial y$ and returns $\bar{x}_i = \partial L / \partial x_i$ for each input.

The engine does three things.

1. During the forward pass, record each op's inputs and a closure that computes its VJP.
2. Topologically sort the graph from the loss.
3. Walk that order in reverse, seed $\bar{L} = 1$, call each closure, and sum every gradient that arrives at a node from its consumers.

The sum in step 3 is the multivariate chain rule. A node used twice (a residual connection, or `x * x`) receives two contributions, and forgetting to add them is the first bug everyone writes.

### The VJPs That Matter

For a gradient $G$ flowing into the output, the rules I needed are short.

| Op | Backward |
|---|---|
| $C = AB$ | $\bar{A} = G B^{\top}$, $\bar{B} = A^{\top} G$ |
| sum over an axis | broadcast $G$ back along that axis |
| $s = \operatorname{softmax}(x)$ | $\bar{x} = s \odot (G - \langle G, s \rangle)$ |
| $y = \log \operatorname{softmax}(x)$ | $\bar{x} = G - \operatorname{softmax}(x) \sum G$ |
| mean cross-entropy over $N$ rows | $\bar{z} = (\operatorname{softmax}(z) - \operatorname{onehot}) / N$ |
| gather $W[\mathrm{idx}]$ | scatter-add $G$ back into the rows of $W$ |

The softmax rule matters because it never forms the $C \times C$ Jacobian $\operatorname{diag}(s) - s s^{\top}$. It contracts the Jacobian with $G$ analytically, which is the whole point of VJPs.

The gather rule is where the real difficulty hides. If index 7 appears three times in a batch, row 7 of the embedding table must receive the sum of three gradient rows. NumPy's `W[idx] += G` keeps only one of the three writes, silently. More on that in [Problems](#problems).

### Broadcasting Has an Adjoint

Broadcasting copies a value along an axis. The adjoint of copying is summing. So every binary op has to take the gradient it received, which has the broadcast output's shape, and sum it back down to each input's shape. There are two cases. Broadcasting can prepend axes (a bias of shape `(512,)` added to activations of shape `(128, 512)`), and it can stretch a size-1 axis (shape `(128, 1)` against `(128, 512)`). Here is the function that undoes both, in full.

```python
def _unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)
```

Every binary op routes both parent gradients through it, and the engine asserts after every VJP that the gradient's shape equals its tensor's shape. That assertion catches an entire class of bug at the op that caused it rather than three layers later.

### Checking Gradients Numerically

The oracle for every op is a central difference.

$$
\frac{\partial L}{\partial x_i} \approx \frac{L(x + h e_i) - L(x - h e_i)}{2h},
$$

whose truncation error is $O(h^2)$. I run it in float64 with $h = 10^{-6}$. For an op with a tensor output, the checker contracts the output with a fixed random tensor $w$ and checks $L = \sum w \odot f(x)$ instead of $\sum f(x)$. With plain `sum()`, every output element gets weight 1, so a backward that permutes or transposes output gradients could pass. A random projection gives each element its own weight and closes that hole.

## Architecture

The value and the graph node are the same object. A `Tensor` wraps one NumPy array (float32 by default, float64 when given float64), and there is no separate node or function class.

<figure class="excal" data-diagram="smolgrad-layers"><a href="/img/diagrams/smolgrad-layers.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/smolgrad-layers.webp" alt="smolgrad package layers: user code calls nn, optim and gradcheck; nn goes through functional into the Tensor class, whose forward ops record a closure and whose backward sweeps them in reverse, while optim reads grad and updates data in place outside the graph; numpy sits underneath." width="2400" height="2200" loading="lazy" decoding="async"></a></figure>

One training step flows like this.

<figure class="excal" data-diagram="smolgrad-step"><a href="/img/diagrams/smolgrad-step.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/smolgrad-step.webp" alt="One smolgrad training step: the input flows through Linear, relu, Linear and cross_entropy to a scalar loss, backward runs the recorded closures in reverse to fill the leaf gradients, step updates the weights in place, and zero_grad clears the gradients for the next batch." width="2400" height="1630" loading="lazy" decoding="async"></a></figure>

### Closures, Not Function Objects

Each op defines its backward as a closure that captures what it needs from the forward pass. `exp` captures its output, softmax captures the probabilities. The closure *returns* parent gradients rather than writing them anywhere, which keeps all accumulation in one place (the engine) and makes each op's backward a pure function that finite differences can test in isolation.

The alternatives were a `Function` class per op, as in PyTorch's `autograd.Function` and earlier versions of tinygrad<sup>[[8]](#ref-8)</sup>, or a tape of `(op, inputs, output)` records. Closures are the least code per op. Their cost is that they are opaque, so you cannot inspect, rewrite or fuse the graph. That trade-off comes back in [What I Would Change](#what-i-would-change).

### Gradients Live in a Local Dict

During `backward()` the gradients of intermediate nodes live in a dict keyed by node identity, local to that call, and each entry is popped as soon as its node is processed. Only leaves get a `.grad`, and it accumulates across calls until zeroed, like PyTorch.

The micrograd approach of storing `.grad` on every node is simpler and has two bugs waiting. A second backward call double counts intermediate gradients unless something resets them, and every intermediate keeps a gradient array alive as long as the graph lives. The local dict avoids both.

### PyTorch Conventions, on Purpose

`Linear.weight` is stored as `(out_features, in_features)`, init is $U(-1/\sqrt{\text{fan\_in}}, 1/\sqrt{\text{fan\_in}})$, the SGD momentum buffer starts at $v = g$ on the first step, the AdamW update order matches, relu's gradient at exactly 0 is 0, and LayerNorm<sup>[[9]](#ref-9)</sup> uses the biased variance. None of these are the only reasonable choice. They are PyTorch's choices, and matching them is what makes step-for-step parity tests possible, which is what makes the 5.3e-14 result possible.

## Implementation

The library is 891 lines across seven files. Almost half of it, 439 lines, is the core that holds the Tensor, the engine and every primitive op. The rest splits into the composed and fused ops, the module system, the optimizers and the gradient checker.

### Recording an Edge

Every op builds its output through one function, and that function is the only place the graph grows.

```python
@staticmethod
def _make(data, parents, backward, op):
    out = Tensor(data, dtype=data.dtype)
    if _GRAD_ENABLED and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._parents = parents
        out._backward = backward
        out._op = op
    return out
```

Under `no_grad()`, or when no parent needs a gradient, nothing is stored, so evaluation builds no graph at all. A typical op then reads like its math.

```python
def __mul__(self, other):
    other = self._const(other)
    a, b = self, other

    def backward(g):
        ga = _unbroadcast(g * b.data, a.shape) if a.requires_grad else None
        gb = _unbroadcast(g * a.data, b.shape) if b.requires_grad else None
        return ga, gb

    return Tensor._make(a.data * b.data, (a, b), backward, "mul")
```

### The Engine

The topological sort is an iterative depth-first search with an explicit stack that remembers whether each node's parents have been visited, emitting a node only after all its parents. The reverse sweep is the part worth reading.

```python
grads = {id(self): grad}
handed_out = set()
for node in reversed(self._toposort()):
    g = grads.pop(id(node), None)
    if g is None:
        continue
    if node._backward is None:  # leaf
        if node.grad is not None:
            node.grad = node.grad + g
            continue
        if id(g) in handed_out or not g.flags.owndata or not g.flags.writeable:
            g = g.copy()
        handed_out.add(id(g))
        node.grad = g
        continue
    for p, pg in zip(node._parents, node._backward(g)):
        if pg is None or not p.requires_grad:
            continue
        assert pg.shape == p.shape, f"{node._op}: grad {pg.shape} vs param {p.shape}"
        k = id(p)
        grads[k] = pg if k not in grads else grads[k] + pg
```

The last line is the chain rule's sum. Note that it allocates a new array rather than adding in place, because the incoming gradient may be the same object another parent also received. The leaf branch with its three-way copy condition looks paranoid. It is the fix for the first bug in [Problems](#problems).

### Fused Where It Pays, Composed Where It Does Not

The linear layer and LayerNorm are compositions of primitives. LayerNorm is about ten graph nodes, slower than a fused kernel, but its gradient comes free from primitives that are already checked, and it matches `torch.nn.LayerNorm` gradients to a relative 1e-8.

Cross-entropy is fused. Composing `log(softmax(x))` produces three graph nodes and computes a log of something that can underflow to zero. The fused version computes a max-shifted log-sum-exp once and has the well-known backward.

```python
def backward(g):
    grad = np.exp(logp)
    grad[np.arange(N), t] -= 1.0
    grad *= g * scale  # g is the upstream 0-d gradient, scale is 1/N for mean
    return (grad.reshape(logits.shape),)
```

A test checks it against the composed version, and another feeds it logits large enough that the naive form overflows.

The embedding lookup is fused because its fast backward cannot be expressed with the other ops. That backward turned out to be the most expensive line in the character model, which is the fourth entry in [Problems](#problems).

### Modules Without Registration

A `Module` finds its parameters by walking its own attributes in insertion order, recursing into sub-modules and lists of them, with no explicit registration step. The deterministic order matters because optimizer state is a list aligned with the parameters, and the parity tests copy weights to PyTorch by zipping the two parameter lists.

### AdamW with No Allocations

AdamW is the decoupled-weight-decay version<sup>[[4]](#ref-4)</sup>, where decay multiplies the weights directly instead of being added to the gradient and then divided by the adaptive denominator. Written as ordinary NumPy expressions, one step created about six parameter-sized temporaries per tensor. The version that shipped uses one preallocated scratch buffer per parameter and `out=` arguments throughout.

```python
np.multiply(g, 1.0 - self.b1, out=tmp)   # m = b1*m + (1-b1)*g
m *= self.b1
m += tmp
np.multiply(g, g, out=tmp)               # v = b2*v + (1-b2)*g^2
tmp *= 1.0 - self.b2
v *= self.b2
v += tmp
np.sqrt(v, out=tmp)                      # denom = sqrt(v)/sqrt(bc2) + eps
tmp *= inv_sqrt_bc2
tmp += self.eps
np.divide(m, tmp, out=tmp)               # p -= lr/bc1 * m/denom
tmp *= step_size
p.data -= tmp
```

It still makes about ten passes over each parameter, which is where most of the remaining optimizer gap to PyTorch comes from.

### The Tests

The suite has 101 tests in three groups.

The first is 58 finite-difference cases covering every op, including broadcasting variants, batched and broadcast-batch matmul, reductions over several axes and indexing with repeated indices, plus a self-test that the checker rejects a deliberately wrong backward. A checker that cannot fail proves nothing.

The second is 19 tests of engine semantics, covering diamonds, accumulation across backward calls, aliasing, `no_grad`, dtype preservation, a 20,000-iteration chain of about 40,000 nodes, and both paths of the embedding scatter against `np.add.at`.

The third is 23 tests comparing op values and gradients, modules, and 20-step SGD and AdamW trajectories against PyTorch. The trajectory tests require losses equal to a relative 1e-9 at every step and final weights equal to 1e-8 relative in float64.

## Problems

This section lists only the issues that left evidence in the code, the tests or the recorded results.

### 1. Leaf Gradients Aliasing Each Other

`add` returns the *same* array object as the gradient for both parents, since $\partial(a + b)/\partial a$ and $\partial(a + b)/\partial b$ are both the identity. And `sum`'s backward returns `np.broadcast_to(g, shape)`, which is a read-only view with zero strides, not a real array. The first version of the engine stored whatever arrived at a leaf directly as its `.grad`.

That caused two failures. Two parameters could end up sharing one gradient array, so an in-place change to one silently changed the other. And the first time a leaf's gradient came from `sum` and was later accumulated into, the in-place add failed outright because the array was read-only.

The regression test is four lines and describes the bug exactly.

```python
a = Tensor(np.zeros(3), requires_grad=True)
b = Tensor(np.zeros(3), requires_grad=True)
(a + b).sum().backward()
a.grad += 100.0
np.testing.assert_allclose(b.grad, [1.0, 1.0, 1.0])
```

The obvious fix is to copy every incoming gradient at every leaf. That works and costs a full copy of every parameter's gradient every step. The shipped fix copies only when it must, which is when the array is a view that does not own its data, is read-only, or has already been handed to another leaf in this same call (tracked in a set). Otherwise the leaf adopts the fresh array the VJP produced. The caller's seed gradient is always copied, since it might be the caller's own array and could otherwise become a leaf's `.grad`, and a second test covers that.

The lesson is that the engine's contract with every VJP needs an ownership rule. Mine is that closures never mutate their input gradient, and the engine makes sure a leaf's gradient is its own.

### 2. Silent float64 Promotion

`Tensor(float32) * 0.5` must stay float32. Wrapping a Python scalar in a plain NumPy array makes it float64, and NumPy's promotion rules then make the whole product float64. Nothing crashes. Activations would quietly double in size and every downstream op would run at float64 speed.

The fix is to wrap every constant in the *calling tensor's* dtype. The optimizer had the same bug in a different place. AdamW's bias correction `1 / sqrt(1 - b2**t)` computed with NumPy is a float64 scalar, and multiplying a float32 buffer by it promotes. It is now converted to a Python float first. A regression test guards the op side, and the float32 parity test asserts the loss dtype.

### 3. The Recursion Limit

A recursive topological sort is five lines and breaks at Python's default recursion limit of 1,000 frames. An MLP never gets near that. A long unrolled recurrence, or any test that chains ops in a loop, does. I replaced it with the iterative DFS above, and a regression test runs `y = y * 1.0 + 0.0` twenty thousand times and checks the gradient is exactly 1.

### 4. np.add.at Was the Most Expensive Call in a Training Step

The embedding backward has to scatter-add gradient rows into the table with repeated indices accumulating. The textbook NumPy answer is `np.add.at(full, idx, g)`, and that is what the generic indexing op still uses. It is correct. It is also an unbuffered per-element loop.

Profiling 200 steps of the character model with cProfile showed it. The top of the profile, taken at a load average of 252, looked like this.

| Function | Total time over 200 steps |
|---|---|
| `ufunc.at` (the scatter) | 1.117 s |
| AdamW step | 1.030 s |
| matmul backward | 0.401 s |
| matmul forward | 0.392 s |

The scatter alone cost more than the whole optimizer. The char model scatters 4,096 rows (a batch of 256 times a 16-character context) into a 65-row table. I benchmarked two replacements.

- The first is a one-hot matmul. Build an $N \times V$ one-hot matrix and compute $\operatorname{onehot}(\mathrm{idx})^{\top} G$. It is a single BLAS call, and wasteful in FLOPs, but at $V = 65$ the waste is irrelevant.
- The second is sort and reduceat. Sort the indices, find the start of each run of equal indices, and sum each run with `np.add.reduceat`. It does no wasted arithmetic, but pays for a sort and a gather.

<figure data-figure="chart:projects/autograd-from-scratch/autograd-from-scratch-scatter-add"></figure>

At the char model's shape the one-hot product took 72 µs against 1,272 µs for `np.add.at`, a 17.6x speedup, with sort and reduceat at 484 µs. The one-hot matrix grows as $N \times V$, though, and at a GPT-2-sized vocabulary of 50,257<sup>[[10]](#ref-10)</sup> with 8,192 rows it would be over 400 million entries, so it is not even attempted there. At that shape sort and reduceat wins, 9,399 µs against 12,530 µs.

The shipped function picks by size.

```python
def _scatter_add_rows(idx, g, num_rows):
    n = idx.shape[0]
    if n * num_rows <= (1 << 20):
        onehot = np.zeros((n, num_rows), dtype=g.dtype)
        onehot[np.arange(n), idx] = 1
        return onehot.T @ g
    order = np.argsort(idx, kind="stable")
    s = idx[order]
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    out = np.zeros((num_rows, g.shape[1]), dtype=g.dtype)
    out[s[starts]] = np.add.reduceat(g[order], starts, axis=0)
    return out
```

Two honest caveats. At the middle shape ($V = 1{,}000$, $N = 8{,}192$), the threshold sends the call to sort and reduceat even though one-hot had the lower minimum (1,794 µs against 1,981 µs). The medians rank them the other way (4,461 µs against 3,226 µs), so under this much load I cannot call it either way, and the threshold is a memory rule as much as a speed rule. Second, sort and reduceat is not bit-identical to `np.add.at`. Its maximum absolute error in float32 was 7.6e-6 at the small shape, because it sums each run in a different order. That is rounding, not a bug, and it foreshadows problem 6.

Rerunning the same profile after the change, the step went from 23.78 ms to 11.51 ms under the profiler and the scatter dropped out of the top ten. Not all of that halving is the scatter, though. The scatter accounted for 5.6 ms per step, and the rest of the 12.3 ms drop came from lines whose code did not change, such as the AdamW step falling from 1.030 s to 0.395 s over 200 steps. That part is load, and the honest claim is a 5.6 ms per step saving.

### 5. AdamW's Temporaries

In the second profile the optimizer is the top line. For a 535,818-parameter MLP at small batch it is a large, batch-independent share of every step, and the expression form of AdamW allocated about six parameter-sized arrays per tensor per step. The `out=` rewrite shown in [AdamW with No Allocations](#adamw-with-no-allocations) removed them. It did not close the gap to PyTorch's multi-tensor AdamW, which I return to in [Results](#results).

### 6. float32 Drifts Away from PyTorch, and It Is Not a Bug

This is the one that looked like a real bug and took the most care to rule out.

The MNIST training run trains a PyTorch twin from the same initial weights on the same batches and logs the maximum absolute weight difference after each epoch. In float32 it was 0.098 after one epoch and **0.313** after ten. That is not rounding error by any normal standard, yet every parity test passed. Either there was a bug that only shows up over thousands of steps, or training amplifies rounding differences into large weight differences.

Those two explanations make different predictions about precision. A real bug in a gradient or an update rule does not care about dtype. It should make the float64 twins diverge just as fast. Amplified rounding should shrink by roughly the ratio of the two machine epsilons, about nine orders of magnitude. So I reran the twin experiment in float64 for three epochs.

| Epoch | float64 max weight difference | smolgrad test accuracy | PyTorch test accuracy |
|---|---|---|---|
| 1 | 2.5e-14 | 96.81% | 96.81% |
| 2 | 2.3e-14 | 97.68% | 97.68% |
| 3 | 5.3e-14 | 97.75% | 97.75% |

After 1,407 AdamW steps on 535,818 parameters, the two implementations agree to 5.3e-14. The epoch train losses agree to 15 significant figures. Test accuracy is identical at every epoch. There is no room for a bug in the forward pass, any gradient, or the optimizer on this path.

To see how the float32 gap grows, I ran the twins for 300 steps in both dtypes with both optimizers and logged the weight difference every step.

| Run | Step 1 | Step 30 | Step 100 | Step 300 |
|---|---|---|---|---|
| float32, SGD with momentum | 3.7e-9 | 7.5e-8 | 4.4e-3 | 2.6e-2 |
| float32, AdamW | 8.3e-6 | 8.4e-6 | 6.8e-4 | 6.7e-2 |
| float64, SGD with momentum | 6.9e-18 | 1.7e-16 | 4.4e-16 | 5.6e-16 |
| float64, AdamW | 9.7e-15 | 9.8e-15 | 9.6e-15 | 1.5e-14 |

In float64 the difference stays at or below 1.5e-14 for 300 steps with either optimizer. In float32 it starts at the rounding level and does not grow smoothly. It jumps. The SGD run sits at 7.5e-8 through step 30 and is at 1.8e-6 one step later, 24 times larger. The AdamW run is flat at about 8.3e-6 for 77 steps, then climbs, and jumps again between steps 91 and 93. Discrete jumps like that are consistent with a ReLU unit whose pre-activation is near zero landing on different sides of zero in the two runs, after which the two networks are computing slightly different functions. I have not traced a specific unit, so that is an interpretation, not a finding.

The implementations round differently because they sum in different orders. NumPy's reductions and Accelerate's GEMM do not accumulate in the same order as PyTorch's kernels, float32 addition is not associative, and training is a chaotic map that amplifies the difference.

AdamW starts noisier than SGD, 8.3e-6 after one step against 3.7e-9, and that has a clean explanation. Adam's update<sup>[[11]](#ref-11)</sup> is $m / (\sqrt{v} + \epsilon)$, which normalizes each coordinate's step toward the full learning rate regardless of the gradient's magnitude. For a weight whose true gradient is near zero, a rounding-level difference in that gradient becomes a difference of order the learning rate in the update. SGD scales the update by the gradient itself, so tiny differences stay tiny.

The practical takeaway is that float32 parity with a reference implementation is only meaningful over a handful of steps, and long-run parity has to be tested in float64. The ten-epoch float32 twins still end at 97.89% and 97.81%, which is the level at which they should agree. The same sensitivity shows up between thread configurations, not just between implementations, as the MNIST results below describe.

### 7. The Machine Was Shared

Every timing in this project was taken while other jobs were running on the same 14-core M4 Pro. I recorded the one-minute load average at each measurement, and it was 338 to 358 for the single-threaded benchmark, 358 to 360 for the default-threads benchmark, 194 to 202 for the scatter benchmark and 252 for the profile. A load of 300 on 14 cores means each runnable process waits most of the time.

The benchmark interleaves the implementations round-robin, so any burst of load hits all three alike, reports the minimum over all runs as the best estimate of intrinsic cost, since load can only add time, and records the load average alongside every result. The medians measure the other jobs more than smolgrad, and I do not quote them. The absolute times below are upper bounds.

How much load distorted things shows in the training runs. The MNIST run's first epoch took 19.2 s and its tenth took 4.5 s with no code change, and the character model averaged 71 ms per step during training against a benchmark minimum of 3.4 ms.

## Experiments

1. Correctness is checked by the 101-test suite.
2. MNIST uses a 784-512-256-10 ReLU MLP, 535,818 parameters, AdamW at learning rate 1e-3 and weight decay 0.01, batch 128, 10 epochs in float32, with a PyTorch twin from identical weights on identical batches. Then the same twin setup in float64 for 3 epochs.
3. The drift experiment runs 300 steps for each of float32 and float64 crossed with SGD with momentum and AdamW, logging the weight difference to the PyTorch twin every step.
4. The character model is a Bengio-style character MLP<sup>[[12]](#ref-12)</sup> on Tiny Shakespeare<sup>[[6]](#ref-6)</sup>. An `Embedding(65, 32)` over a 16-character context is flattened to 512, then `Linear(512, 512)`, `LayerNorm`, tanh, `Linear(512, 65)`. That is 299,105 parameters, trained with AdamW at 3e-3 with a linear decay over the last quarter, batch 256, 6,000 steps, and compared against uniform, unigram and bigram baselines on the same validation split.
5. The step-time benchmark times a full training step (forward, backward, AdamW) of the MNIST MLP at batch 1 to 2048 and the character model at batch 256, single-threaded and with default threading, in three implementations. They are smolgrad, PyTorch eager on CPU, and the same MLP hand-written in NumPy with manual gradients, no graph, and smolgrad's own AdamW. A microbenchmark also times 2,000 chained multiplies on one-element tensors, forward plus backward.

The hand-written NumPy MLP is the design choice that makes the benchmark worth reading. Without it, "1.8x slower than PyTorch" could mean a slow graph engine or slow kernels. With it, smolgrad minus NumPy is the price of autograd, and NumPy minus PyTorch is the price of NumPy.

## Results

### Correctness

All 101 tests pass. The float64 twin result above is the strongest end-to-end evidence, since it exercises every op in the MLP, the fused cross-entropy, the engine's accumulation and AdamW over 1,407 steps at once.

### MNIST

| | smolgrad | PyTorch twin |
|---|---|---|
| final test accuracy, epoch 10 | 97.89% | 97.81% |
| final test loss | 0.0863 | 0.0946 |
| final train loss | 0.0184 | 0.0187 |

I report only the final epoch. The run also recorded a best test accuracy of 98.11% at epoch 5, but there is no validation split, so picking the best epoch means picking it on the test set, and that number is optimistic by construction. A separate validation split would fix this.

The float32 run is deterministic for a fixed thread count but not across thread counts. An independent rerun with BLAS capped at 2 threads matched itself exactly, and reached 98.07% final accuracy with a final test loss of 0.0778, against 97.89% and 0.0863 here. The runs already differ after the first epoch (96.96% against 96.77%), which points to summation order in the matrix multiplies rather than a bug, and the float64 twin rules out a gradient error. So read 97.89% as one draw from a spread of about 0.2 points.

Test loss bottoms out at epoch 5 at 0.0623 and rises afterwards while train loss keeps falling, which is ordinary overfitting with no regularization beyond weight decay. The 0.08-point accuracy difference between the twins is float32 drift, not a property of either implementation. At epoch 8 the ranking was the other way around, 97.51% against 97.88%.

### Tiny Shakespeare

<figure data-figure="chart:projects/autograd-from-scratch/autograd-from-scratch-char-loss"></figure>

The character model reaches 1.794 nats per character on the validation split after 6,000 steps, 0.688 below the bigram baseline. Train loss at the end was 1.576, so the model is starting to overfit its 16-character windows, and validation loss was still falling at the last evaluation, helped by the learning-rate decay. Samples at temperature 0.8 get the play format right, with speaker names and line breaks, mostly real short words and invented longer ones.

```
KING RICHARD III:
Ay, I woonesh thee, I say though it east hear good.
```

That is about what a 300,000-parameter model with a 16-character window should manage.

### Step Time Against PyTorch

<figure data-figure="chart:projects/autograd-from-scratch/autograd-from-scratch-step-ratio"></figure>

Single-threaded, taking the minimum over 320 steps for batches up to 128 and 120 steps above, at a load average of 338 to 358.

| MLP batch | smolgrad | NumPy by hand | PyTorch | smolgrad / PyTorch |
|---|---|---|---|---|
| 1 | 3.21 ms | 2.32 ms | 1.29 ms | 2.5x |
| 32 | 2.43 ms | 1.62 ms | 1.32 ms | 1.8x |
| 128 | 2.94 ms | 2.09 ms | 1.66 ms | 1.8x |
| 512 | 5.01 ms | 4.97 ms | 3.00 ms | 1.7x |
| 2048 | 21.2 ms | 38.1 ms | 11.6 ms | 1.8x |

The character model step at batch 256 was 3.40 ms for smolgrad and 2.77 ms for PyTorch, a ratio of 1.2x.

The batch 2048 row, where hand-written NumPy is 17 ms slower than smolgrad, is load noise. The two run the same matmuls. I keep the row because deleting inconvenient measurements is worse than labelling them, and its other numbers deserve the same suspicion.

### Where the Gap Comes From

<figure data-figure="chart:projects/autograd-from-scratch/autograd-from-scratch-phases"></figure>

At batch 128 the phases split like this.

| Phase | smolgrad | NumPy by hand | PyTorch |
|---|---|---|---|
| forward | 0.42 ms | 0.36 ms | 0.31 ms |
| backward | 1.15 ms | 0.38 ms | 0.35 ms |
| AdamW step | 1.33 ms | 1.32 ms | 1.00 ms |

My reading of it, with the usual hedge that these are minima under heavy load and that phase minima do not add exactly to step minima.

The autograd machinery costs about 0.85 ms per step (2.94 minus 2.09), and almost all of it is in backward, 1.15 ms against 0.38 ms. The forward passes are within 0.06 ms of each other, so recording the graph is cheap. The backward cost is extra memory traffic. The hand-written version computes the relu mask and applies it in one expression, sums the bias gradient directly, and never builds a dict or checks an ownership flag. smolgrad unbroadcasts every bias add, allocates a new array for every relu mask multiply, allocates again when summing gradients that arrive at the same node, and copies wherever a leaf cannot adopt its gradient.

The kernel gap is smaller, about 0.43 ms (2.09 minus 1.66), and most of it is the optimizer. PyTorch's `foreach` AdamW updates all parameters with multi-tensor kernels, while mine makes about ten NumPy passes over each parameter. At batch 128, AdamW is 1.33 of smolgrad's 2.94 ms, or 45% of the step, and because its cost depends only on the parameter count it takes a larger share the smaller the batch.

Per-op Python overhead is not the problem. On one-element tensors, a forward plus backward multiply costs 6.2 µs in smolgrad and 12.6 µs in PyTorch single-threaded, and 6.2 µs against 10.0 µs with default threading. PyTorch's dispatcher does more work per op than a Python closure does. This surprised me, and it means the gap on real workloads is about how many bytes move, not how many Python calls happen.

The default-threads benchmark (40 or 15 steps per point, at a load average near 360) is less reliable. In it PyTorch at batch 128 is slower than hand-written NumPy, 5.24 ms against 2.25 ms, which is what ten PyTorch threads on an oversubscribed machine look like. I use it only for the per-op overhead numbers.

## What I Would Change

### Record Ops, Not Closures

Closures made v0 short, and they make the graph a black box. There is no way to look at a chain of `add`, `relu`, `mul` and fuse it. The planned lazy-graph milestone needs an explicit op record (type, inputs, attributes), and I would switch to that first.

### Fuse the Obvious Hot Spots Before Writing a Backend

The phase table says where the 1.8x lives. A fused bias-plus-relu with an in-place mask in backward attacks the 0.77 ms of backward overhead directly. A multi-tensor AdamW over one flat parameter buffer would replace about ten NumPy passes over each of the MLP's six parameter tensors with about ten passes over one buffer, and should attack most of the 0.33 ms optimizer gap. A fused LayerNorm forward and backward would replace ten graph nodes with one in the character model. All three are possible in NumPy without writing a kernel.

### Add a Version Counter

Closures read their parents' `.data` lazily at backward time, so updating a parameter in place between forward and backward silently corrupts the gradients. PyTorch catches this with a version integer on every tensor. v0 relies on the convention that only the optimizer mutates in place, and it runs after backward.

### Write Backward Rules in Tensor Ops for Higher-Order Gradients

Right now each VJP is raw NumPy, so the backward pass builds no graph and grad-of-grad is impossible. Supporting it means a second set of backward rules written in Tensor ops, behind a `create_graph=True` flag so the first-order path does not pay for it.

### Rerun Every Timing on an Idle Machine

The benchmark already has a wrapper that waits for the load average to drop below a threshold. The numbers here were taken at a one-minute load average of 194 to 360 and are upper bounds, with the single-threaded ratios roughly right. The float64 parity result does not depend on the machine at all, which is why it leads this post.

## References

1. <span id="ref-1"></span>Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, et al. *Automatic Differentiation in Machine Learning: a Survey*. Journal of Machine Learning Research 18(153), 2018. [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
2. <span id="ref-2"></span>Adam Paszke, Sam Gross, Francisco Massa, et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS, 2019. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
3. <span id="ref-3"></span>Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, et al. *Array Programming with NumPy*. Nature 585, 357–362, 2020. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
4. <span id="ref-4"></span>Ilya Loshchilov and Frank Hutter. *Decoupled Weight Decay Regularization*. ICLR, 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
5. <span id="ref-5"></span>Yann LeCun, Léon Bottou, Yoshua Bengio, et al. *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE 86(11), 2278–2324, 1998. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
6. <span id="ref-6"></span>Andrej Karpathy. *char-rnn* (Tiny Shakespeare dataset, `data/tinyshakespeare`). GitHub, 2015. [link](https://github.com/karpathy/char-rnn)
7. <span id="ref-7"></span>Andrej Karpathy. *micrograd: A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API*. GitHub, 2020. [link](https://github.com/karpathy/micrograd)
8. <span id="ref-8"></span>tiny corp. *tinygrad v0.9.0*. GitHub, 2024. [link](https://github.com/tinygrad/tinygrad/blob/v0.9.0/tinygrad/function.py)
9. <span id="ref-9"></span>Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. *Layer Normalization*. arXiv preprint, 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
10. <span id="ref-10"></span>Alec Radford, Jeffrey Wu, Rewon Child, et al. *Language Models are Unsupervised Multitask Learners*. OpenAI technical report, 2019. [link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
11. <span id="ref-11"></span>Diederik P. Kingma and Jimmy Ba. *Adam: A Method for Stochastic Optimization*. ICLR, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
12. <span id="ref-12"></span>Yoshua Bengio, Réjean Ducharme, Pascal Vincent, et al. *A Neural Probabilistic Language Model*. Journal of Machine Learning Research 3, 1137–1155, 2003. [link](https://jmlr.org/papers/v3/bengio03a.html)
