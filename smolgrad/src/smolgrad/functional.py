"""Stateless ops built on Tensor. Some are fused primitives with a hand-written
backward (cross_entropy, embedding) and some are compositions of primitives
(linear, layer_norm) whose gradients come for free from the engine."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor, as_numpy_index


def relu(x: Tensor) -> Tensor:
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis)


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """y = x @ W^T + b with W stored as (out_features, in_features) like PyTorch."""
    y = x @ weight.T
    return y + bias if bias is not None else y


def _scatter_add_rows(idx: np.ndarray, g: np.ndarray, num_rows: int) -> np.ndarray:
    """Return Z of shape (num_rows, D) with Z[r] = sum of g[i] over all i where idx[i] == r.

    This is the backward of a row gather. `Z[idx] += g` would be wrong because
    fancy-index assignment keeps only one write per repeated index. np.add.at is
    correct but slow (it is an unbuffered per-element loop), so two faster
    strategies are used, picked by size (timings in DEVLOG.md, Problems):
      * small N*V: one-hot matrix product onehot(idx)^T @ g, a single BLAS call;
      * otherwise: sort idx, then np.add.reduceat over runs of equal indices.
    """
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


def embedding(weight: Tensor, idx) -> Tensor:
    """Row gather weight[idx] for integer idx of any shape -> idx.shape + (D,).
    Backward scatter-adds rows; repeated indices must accumulate."""
    idx = as_numpy_index(idx)
    flat = idx.reshape(-1)
    V, D = weight.shape

    def backward(g):
        return (_scatter_add_rows(flat, g.reshape(-1, D), V),)

    return Tensor._make(weight.data[idx], (weight,), backward, "embedding")


def cross_entropy(logits: Tensor, target, reduction: str = "mean") -> Tensor:
    """Fused log_softmax + negative log likelihood.

    logits: (N, C) or (..., C), target: integer class ids with shape logits.shape[:-1].
    Fusing gives the famous (softmax - onehot) / N gradient in one pass and never
    materialises log(softmax), which is both faster and more stable than
    composing log(softmax(x)).
    """
    target = as_numpy_index(target)
    C = logits.shape[-1]
    x = logits.data.reshape(-1, C)
    t = target.reshape(-1)
    N = x.shape[0]
    if t.shape[0] != N:
        raise ValueError(f"target has {t.shape[0]} entries, logits have {N} rows")

    z = x - x.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - lse
    nll = -logp[np.arange(N), t]
    if reduction == "mean":
        out, scale = nll.mean(), 1.0 / N
    elif reduction == "sum":
        out, scale = nll.sum(), 1.0
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    def backward(g):
        grad = np.exp(logp)
        grad[np.arange(N), t] -= 1.0
        grad *= g * scale  # g is a 0-d array (the upstream scalar gradient)
        return (grad.reshape(logits.shape),)

    return Tensor._make(np.asarray(out, dtype=x.dtype), (logits,), backward, "cross_entropy")


def layer_norm(x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float = 1e-5) -> Tensor:
    """LayerNorm over the last axis, composed from primitives (mean, mul, pow).
    Uses the biased variance, like PyTorch."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    y = xc * (var + eps) ** -0.5
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y
