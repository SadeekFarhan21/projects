"""Finite-difference gradient checking.

For a function f: inputs -> tensor, we contract the output with a fixed random
tensor w and check the scalar L = sum(w * f(x)). Using a random projection
rather than sum() means every output element influences L with a different
weight, so a bug that permutes or mis-scales output gradients cannot cancel out.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .tensor import Tensor, no_grad


def numerical_grad(f: Callable[..., Tensor], inputs: Sequence[np.ndarray], w: np.ndarray,
                   eps: float = 1e-6) -> list[np.ndarray]:
    """Central differences, O(eps^2) truncation error, in float64."""
    xs = [np.array(x, dtype=np.float64) for x in inputs]
    grads = []
    with no_grad():
        for k, x in enumerate(xs):
            g = np.zeros_like(x)
            it = np.nditer(x, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                orig = x[i]
                x[i] = orig + eps
                fp = float((f(*[Tensor(v) for v in xs]).data * w).sum())
                x[i] = orig - eps
                fm = float((f(*[Tensor(v) for v in xs]).data * w).sum())
                x[i] = orig
                g[i] = (fp - fm) / (2 * eps)
            grads.append(g)
    return grads


def gradcheck(f: Callable[..., Tensor], inputs: Sequence[np.ndarray], eps: float = 1e-6,
              atol: float = 1e-6, rtol: float = 1e-5, seed: int = 0) -> float:
    """Raise AssertionError if analytic and numerical gradients disagree.
    Returns the max absolute difference seen (useful for reporting)."""
    ts = [Tensor(np.array(x, dtype=np.float64), requires_grad=True) for x in inputs]
    out = f(*ts)
    w = np.random.default_rng(seed).standard_normal(out.shape)
    (out * Tensor(w)).sum().backward()
    num = numerical_grad(f, inputs, w, eps)
    worst = 0.0
    for k, (t, n) in enumerate(zip(ts, num)):
        a = t.grad if t.grad is not None else np.zeros_like(n)
        worst = max(worst, float(np.max(np.abs(a - n))) if a.size else 0.0)
        if not np.allclose(a, n, atol=atol, rtol=rtol):
            raise AssertionError(
                f"input {k}: analytic and numerical grads differ, max |diff| = {np.max(np.abs(a - n)):.3e}\n"
                f"analytic:\n{a}\nnumerical:\n{n}")
    return worst
