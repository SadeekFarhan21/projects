# smolgrad design

smolgrad is a reverse-mode automatic differentiation engine over a dynamic
computation graph, plus a small neural network library, written in Python on
top of numpy. This document covers the architecture, the data structures, the
invariants the code relies on, and the trade-offs I chose and rejected.

## Architecture

```
                         user code / scripts
                 (train_mnist.py, train_char.py, bench.py)
                                  |
            +---------------------+----------------------+
            |                     |                      |
            v                     v                      v
      smolgrad.nn           smolgrad.optim        smolgrad.gradcheck
  Module, Linear,           SGD (momentum,        central differences,
  Embedding, LayerNorm,     weight decay),        random output projection
  Sequential, ReLU, Tanh    AdamW (decoupled)            |
            |                     |                      |
            | calls               | mutates p.data       | calls f() twice per
            v                     | in place, reads      | input element under
     smolgrad.functional          | p.grad               | no_grad()
  linear, layer_norm (composed)   |                      |
  embedding, cross_entropy (fused)|                      |
            |                     |                      |
            v                     v                      v
  +------------------------------------------------------------------+
  |                        smolgrad.tensor.Tensor                    |
  |  data: np.ndarray   grad: np.ndarray|None   requires_grad: bool  |
  |  _parents: tuple[Tensor]   _backward: g -> tuple[grad|None]      |
  |                                                                  |
  |  forward op  ------>  Tensor._make(out, parents, closure)        |
  |                        (records the edge only if grad is enabled |
  |                         and some parent requires grad)           |
  |                                                                  |
  |  loss.backward():                                                |
  |    1. iterative DFS post-order from loss  -> topo list           |
  |    2. walk topo list in reverse, grads kept in a dict keyed by   |
  |       id(node); each closure returns parent grads; sum into dict |
  |    3. at leaves, store/accumulate into leaf.grad                 |
  +------------------------------------------------------------------+
                                  |
                                  v
                    numpy (Accelerate BLAS on macOS)
```

Data flow for one training step:

```
 x (ndarray) --Tensor--> Linear --> relu --> Linear --> cross_entropy --> loss (0-d)
                  ^ W,b leaves        each op appends one node + closure
 loss.backward():  seed 1.0 --> closures in reverse topo order --> W.grad, b.grad
 opt.step():       reads p.grad, updates p.data in place (outside the graph)
 opt.zero_grad():  p.grad = None
```

## Key data structures

### Tensor

A Tensor is a thin wrapper over one `np.ndarray` (`__slots__`, so no per-object
dict). Graph nodes and values are the same object: there is no separate Node
or Function class. The fields that make up the graph are

| field | meaning |
|---|---|
| `data` | the value, float32 by default, float64 when given float64 |
| `requires_grad` | true for parameters and anything computed from them |
| `_parents` | tuple of input Tensors, empty for leaves and constants |
| `_backward` | closure `g -> tuple(grad or None per parent)`, None for leaves |
| `_op` | op name, only for debugging and assertion messages |
| `grad` | filled only on leaves that require grad |

### Backward closures

Each op defines its backward as a closure that captures what it needs from the
forward pass (for example `exp` captures its output, `softmax` captures the
probabilities). Closures return gradients instead of writing them, which keeps
all accumulation in one place (the engine) and makes each op's backward a pure
function that is easy to test by finite differences.

### Gradient dictionary

During `backward()` the gradients of intermediate nodes live in a dict keyed by
`id(node)`. An entry is popped as soon as the node is processed, so peak memory
holds only the gradients on the current "frontier" of the reverse sweep, and
intermediate tensors never keep a `.grad` around after the call.

### Modules

A Module finds its parameters by walking `vars(self)` in insertion order and
recursing into sub-Modules and lists or tuples of them. There is no
`register_parameter`. Parameter order is deterministic, which the optimizers
(state is a list aligned with the parameter list) and the PyTorch weight
copying in tests both rely on. `Linear.weight` is stored as
`(out_features, in_features)` exactly like PyTorch, so weights copy across
without transposes.

## Invariants

1. A gradient has exactly the shape of the tensor it belongs to. Broadcasting
   is undone by `_unbroadcast`, which sums prepended axes and stretched size-1
   axes. `backward()` asserts this for every parent gradient.
2. Backward closures never mutate their input gradient. The add op hands the
   same array to both parents, so an in-place update in one branch would
   corrupt the other.
3. A leaf owns its `.grad` array. If the incoming array is a view, read-only
   (for example from `np.broadcast_to`), or was already handed to another
   leaf in this call, it is copied first. The seed gradient passed by the
   caller is always copied.
4. Only leaves get `.grad`, and it accumulates across `backward()` calls until
   zeroed, matching PyTorch.
5. The graph must not be modified between forward and backward. Closures read
   parent `.data` lazily, so updating a parameter in place before calling
   `backward()` silently produces wrong gradients. PyTorch catches this with a
   version counter on every tensor; smolgrad does not.
6. Topological order comes from an iterative depth-first search, so graph depth
   is bounded by memory, not by Python's recursion limit (tested with a chain
   of 40,000 nodes).
7. Under `no_grad()` no parents or closures are stored, so evaluation does not
   build a graph at all.

## Ops and where their gradients come from

| op | kind | backward |
|---|---|---|
| add, sub, mul, div, neg, pow(const) | primitive | elementwise rule plus `_unbroadcast` |
| matmul (1-D, 2-D, batched, broadcast batch) | primitive for ndim >= 2, 1-D by reshape composition | `G @ B^T`, `A^T @ G`, then `_unbroadcast` over batch axes |
| sum, mean, max over axes | primitive (mean is sum times constant) | broadcast back; max splits ties evenly |
| reshape, flatten, transpose/permute | primitive | inverse reshape or inverse permutation |
| getitem (basic and advanced) | primitive | `np.add.at` scatter, correct for repeated indices |
| exp, log, sqrt, relu, tanh, sigmoid | primitive | elementwise |
| softmax, log_softmax | primitive, max-shifted forward | `s * (g - sum(g * s))` and `g - softmax * sum(g)` |
| embedding | fused primitive | scatter-add rows (one-hot matmul or sort plus reduceat) |
| cross_entropy | fused primitive | `(softmax - onehot) / N` |
| linear, layer_norm | composed | engine does it |

## Trade-offs

### Chosen: one Tensor type, closures for backward

Alternatives were a `Function` class per op with `forward`/`backward` methods
(PyTorch's `autograd.Function`, and what tinygrad does) or a tape of
`(op, inputs, output)` records. Closures are the least code per op and keep
forward and backward next to each other, which made each op easy to read and
check. The cost is that closures are opaque: you cannot inspect or rewrite the
graph (no fusion pass, no serialization). A future JIT pass would need an
explicit op record, so this is the first thing I would change for the lazy
fusion milestone.

### Chosen: numpy arrays as storage, no own kernels

The library is a graph and gradient engine; all arithmetic is numpy. That
bounds performance at "numpy speed plus Python overhead per op", which the
benchmark measures directly with a hand-written numpy baseline. Writing kernels
(C++ or Metal) is a later milestone.

### Chosen: fused cross_entropy and embedding, composed LayerNorm

Cross-entropy is fused because composing `log(softmax(x))` is numerically worse
and makes three graph nodes where one is enough. Embedding is fused because its
backward is a scatter-add whose fast implementation is not expressible with the
other ops. LayerNorm is composed from about ten primitives. That is slower than
a fused kernel but gets correctness for free from already-checked primitives,
and the parity test against `torch.nn.LayerNorm` holds to 1e-8. A fused
LayerNorm is an easy future win.

### Chosen: gradient dict local to backward, not `.grad` on every node

Storing gradients on every node (micrograd style) is simpler but has two bugs
waiting: calling backward twice double counts intermediate gradients unless
they are reset, and every intermediate keeps a gradient array alive as long as
the graph lives. The local dict avoids both.

### Chosen: PyTorch numerics, conventions and layouts

Same init as PyTorch (`U(-1/sqrt(fan_in), 1/sqrt(fan_in))` for Linear, `N(0,1)`
for Embedding), the same weight layout, the same SGD momentum convention (first
step `v = g`), the same AdamW update order, relu gradient 0 at 0, biased
variance in LayerNorm. This is what makes step-for-step parity tests possible.

### Rejected: higher-order gradients in v0

Supporting grad-of-grad needs backward closures written in terms of Tensor ops
(so the backward pass itself builds a graph) instead of raw numpy. That roughly
doubles the per-op overhead in the common first-order case and was out of scope
for v0. It is on the roadmap, most likely behind a `create_graph=True` flag with
a second set of Tensor-level backward rules.

### Rejected: version counters for in-place safety

Cheap to add (an int per tensor, bumped by optimizer steps and checked in
backward) but no op in v0 mutates in place except the optimizer, which runs
after backward. Documented as invariant 5 instead.

### Rejected: `np.add.at` for embedding backward

It is correct but was the single largest cost in a char-model step. See
DEVLOG.md for the measurements behind the replacement.

### Rejected: a global default-dtype switch

float32 is the default for new tensors, float64 is kept when given. Gradient
checks run in float64 by construction (gradcheck converts inputs), so no global
state is needed.
