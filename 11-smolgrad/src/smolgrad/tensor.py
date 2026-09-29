"""Tensor with reverse-mode automatic differentiation over a dynamic graph.

Every differentiable op builds a new Tensor that remembers its parents and a
backward closure. The closure maps the gradient of the output to a tuple of
gradients, one per parent (None where a parent needs no gradient). The graph is
rebuilt on every forward pass, so Python control flow just works.

Invariants (checked by tests, relied on by the engine):
  * A gradient always has exactly the shape of the tensor it belongs to.
    Broadcasting in the forward pass is undone by `_unbroadcast` in backward.
  * Backward closures never mutate the incoming gradient array. Several
    parents may receive the very same array object (for example both inputs of
    an add), so in-place updates would corrupt siblings.
  * Only leaf tensors with requires_grad=True get `.grad` filled in, and it
    accumulates across backward calls until zeroed (the PyTorch convention).
    Intermediate gradients live in a dict local to one backward call.
  * Parameters must not be modified in place between forward and backward,
    because closures read `.data` of their parents lazily. There is no version
    counter to catch this (PyTorch has one).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import numpy as np

_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Disable graph construction inside the block (evaluation, optimizer steps)."""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing numpy broadcasting.

    Broadcasting can (a) prepend dimensions and (b) stretch size-1 dimensions.
    The adjoint of "copy along an axis" is "sum along that axis", so we sum the
    prepended axes away and sum (keepdims) along stretched axes.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def _norm_axes(axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


BackwardFn = Callable[[np.ndarray], tuple]


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "_op", "__weakref__")
    # Makes `ndarray <op> Tensor` defer to Tensor's reflected operators.
    __array_priority__ = 100.0

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        if dtype is not None:
            arr = np.asarray(data, dtype=dtype)
        else:
            arr = np.asarray(data)
            # Keep float32/float64 arrays as they are; everything else becomes
            # float32, the default training dtype (same default as PyTorch).
            if arr.dtype not in (np.float32, np.float64):
                arr = arr.astype(np.float32)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = bool(requires_grad)
        self._parents: tuple[Tensor, ...] = ()
        self._backward: BackwardFn | None = None
        self._op = ""

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self.data.item()

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        rg = ", requires_grad=True" if self.requires_grad else ""
        op = f", op={self._op}" if self._op else ""
        return f"Tensor({self.data!r}{rg}{op})"

    # ------------------------------------------------------------- graph glue
    def _const(self, x) -> "Tensor":
        """Wrap a python/numpy constant in this tensor's dtype (avoids float64 promotion)."""
        if isinstance(x, Tensor):
            return x
        return Tensor(np.asarray(x, dtype=self.data.dtype))

    @staticmethod
    def _make(data: np.ndarray, parents: tuple["Tensor", ...], backward: BackwardFn, op: str) -> "Tensor":
        out = Tensor(data, dtype=data.dtype)
        if _GRAD_ENABLED and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out._op = op
        return out

    def _toposort(self) -> list["Tensor"]:
        """Iterative DFS post-order (children before parents). Iterative so deep
        graphs such as long RNN unrolls do not hit Python's recursion limit."""
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in visited:
                    stack.append((p, False))
        return order

    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        # Copy the seed: it may be the caller's array and could end up as a leaf's .grad.
        grad = np.array(grad, dtype=self.data.dtype, copy=True).reshape(self.shape)

        grads: dict[int, np.ndarray] = {id(self): grad}
        handed_out: set[int] = set()  # ids of arrays already stored as some leaf's .grad
        for node in reversed(self._toposort()):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf
                if node.grad is not None:
                    node.grad = node.grad + g
                    continue
                # A leaf must own its .grad: copy views (broadcast_to results are
                # read-only views) and arrays already given to another leaf (add
                # passes one array to both parents). Fresh arrays are adopted as is,
                # which saves a full copy per parameter per step.
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

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------ elementwise binary
    def __add__(self, other) -> "Tensor":
        other = self._const(other)
        a, b = self, other
        return Tensor._make(a.data + b.data, (a, b),
                            lambda g: (_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)), "add")

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        return Tensor._make(-self.data, (self,), lambda g: (-g,), "neg")

    def __sub__(self, other) -> "Tensor":
        other = self._const(other)
        a, b = self, other
        return Tensor._make(a.data - b.data, (a, b),
                            lambda g: (_unbroadcast(g, a.shape), _unbroadcast(-g, b.shape)), "sub")

    def __rsub__(self, other) -> "Tensor":
        return self._const(other) - self

    def __mul__(self, other) -> "Tensor":
        other = self._const(other)
        a, b = self, other

        def backward(g):
            ga = _unbroadcast(g * b.data, a.shape) if a.requires_grad else None
            gb = _unbroadcast(g * a.data, b.shape) if b.requires_grad else None
            return ga, gb

        return Tensor._make(a.data * b.data, (a, b), backward, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Tensor":
        other = self._const(other)
        a, b = self, other

        def backward(g):
            ga = _unbroadcast(g / b.data, a.shape) if a.requires_grad else None
            gb = _unbroadcast(-g * a.data / (b.data * b.data), b.shape) if b.requires_grad else None
            return ga, gb

        return Tensor._make(a.data / b.data, (a, b), backward, "div")

    def __rtruediv__(self, other) -> "Tensor":
        return self._const(other) / self

    def __pow__(self, p) -> "Tensor":
        if isinstance(p, Tensor):
            raise NotImplementedError("only constant exponents are supported")
        a = self
        p = float(p)
        return Tensor._make(a.data ** p, (a,), lambda g: (g * p * a.data ** (p - 1),), "pow")

    # ------------------------------------------------------------------ matmul
    def matmul(self, other) -> "Tensor":
        other = self._const(other)
        a, b = self, other
        # 1-D operands are promoted to matrices and the extra axis removed
        # afterwards, which is exactly numpy's rule. Doing it by composition
        # keeps the primitive below to the >= 2-D case.
        if a.ndim == 1:
            out = a.reshape((1, -1)).matmul(b)
            return out.reshape(out.shape[:-2] + out.shape[-1:])
        if b.ndim == 1:
            out = a.matmul(b.reshape((-1, 1)))
            return out.reshape(out.shape[:-1])

        def backward(g):
            # d(A@B)/dA = G @ B^T and d/dB = A^T @ G, then undo batch broadcasting.
            ga = _unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.shape) if a.requires_grad else None
            gb = _unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape) if b.requires_grad else None
            return ga, gb

        return Tensor._make(a.data @ b.data, (a, b), backward, "matmul")

    __matmul__ = matmul

    def __rmatmul__(self, other) -> "Tensor":
        return self._const(other).matmul(self)

    # -------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        a = self
        axes = _norm_axes(axis, a.ndim)

        def backward(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (np.broadcast_to(g, a.shape),)

        return Tensor._make(a.data.sum(axis=axes, keepdims=keepdims), (a,), backward, "sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        axes = _norm_axes(axis, self.ndim)
        count = int(np.prod([self.shape[i] for i in axes])) if axes else 1
        return self.sum(axis=axes, keepdims=keepdims) * (1.0 / count)

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Max reduction. Ties split the gradient evenly (subgradient choice)."""
        a = self
        axes = _norm_axes(axis, a.ndim)
        m = a.data.max(axis=axes, keepdims=True)

        def backward(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            mask = (a.data == m).astype(a.data.dtype)
            mask /= mask.sum(axis=axes, keepdims=True)
            return (mask * g,)

        out = m if keepdims else np.squeeze(m, axis=axes)
        return Tensor._make(out, (a,), backward, "max")

    # ------------------------------------------------------------ shape ops
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return Tensor._make(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),), "reshape")

    view = reshape

    def flatten(self, start_dim: int = 1) -> "Tensor":
        return self.reshape(self.shape[:start_dim] + (-1,))

    def transpose(self, *axes) -> "Tensor":
        """transpose() reverses axes; transpose(i, j) swaps two axes; transpose(perm) permutes."""
        a = self
        if len(axes) == 0:
            perm = tuple(reversed(range(a.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(ax % a.ndim for ax in axes[0])
        elif len(axes) == 2:
            i, j = axes[0] % a.ndim, axes[1] % a.ndim
            p = list(range(a.ndim))
            p[i], p[j] = p[j], p[i]
            perm = tuple(p)
        else:
            perm = tuple(ax % a.ndim for ax in axes)
        inv = tuple(np.argsort(perm))
        return Tensor._make(a.data.transpose(perm), (a,), lambda g: (g.transpose(inv),), "transpose")

    permute = transpose

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __getitem__(self, idx) -> "Tensor":
        """Basic and advanced indexing. Backward scatters with np.add.at so that
        repeated indices (the embedding case) accumulate instead of overwrite."""
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)
        elif isinstance(idx, tuple):
            idx = tuple(i.data.astype(np.int64) if isinstance(i, Tensor) else i for i in idx)
        a = self

        def backward(g):
            full = np.zeros_like(a.data)
            np.add.at(full, idx, g)
            return (full,)

        return Tensor._make(a.data[idx], (a,), backward, "getitem")

    # ------------------------------------------------------ elementwise unary
    def exp(self) -> "Tensor":
        a = self
        out = np.exp(a.data)
        return Tensor._make(out, (a,), lambda g: (g * out,), "exp")

    def log(self) -> "Tensor":
        a = self
        return Tensor._make(np.log(a.data), (a,), lambda g: (g / a.data,), "log")

    def sqrt(self) -> "Tensor":
        a = self
        out = np.sqrt(a.data)
        return Tensor._make(out, (a,), lambda g: (g * 0.5 / out,), "sqrt")

    def relu(self) -> "Tensor":
        a = self
        # Gradient at exactly 0 is taken as 0, matching PyTorch.
        return Tensor._make(np.maximum(a.data, 0), (a,), lambda g: (g * (a.data > 0),), "relu")

    def tanh(self) -> "Tensor":
        a = self
        out = np.tanh(a.data)
        return Tensor._make(out, (a,), lambda g: (g * (1 - out * out),), "tanh")

    def sigmoid(self) -> "Tensor":
        a = self
        out = 1.0 / (1.0 + np.exp(-a.data))
        return Tensor._make(out, (a,), lambda g: (g * out * (1 - out),), "sigmoid")

    # --------------------------------------------------- softmax family
    def softmax(self, axis: int = -1) -> "Tensor":
        a = self
        z = a.data - a.data.max(axis=axis, keepdims=True)  # stability shift
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)

        def backward(g):
            # Jacobian-vector product without forming the (C x C) Jacobian:
            # dx = s * (g - <g, s>)
            return (s * (g - (g * s).sum(axis=axis, keepdims=True)),)

        return Tensor._make(s, (a,), backward, "softmax")

    def log_softmax(self, axis: int = -1) -> "Tensor":
        a = self
        z = a.data - a.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
        out = z - lse

        def backward(g):
            # y = x - logsumexp(x)  =>  dx = g - softmax(x) * sum(g)
            return (g - np.exp(out) * g.sum(axis=axis, keepdims=True),)

        return Tensor._make(out, (a,), backward, "log_softmax")


def tensor(data, requires_grad: bool = False, dtype=None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def as_numpy_index(x) -> np.ndarray:
    return x.data.astype(np.int64) if isinstance(x, Tensor) else np.asarray(x, dtype=np.int64)

