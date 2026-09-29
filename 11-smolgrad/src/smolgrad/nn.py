"""Module system. A Module is any object whose attributes may hold parameters
(Tensors with requires_grad=True), sub-Modules, or lists/tuples of either.
Parameters are discovered by walking attributes in definition order, so there
is no registration boilerplate and the order is deterministic (which matters
for optimizer state and for copying weights to and from PyTorch)."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from . import functional as F
from .random import get_rng
from .tensor import Tensor


class Module:
    training: bool = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _children(self) -> Iterator[tuple[str, object]]:
        for name, value in vars(self).items():
            if isinstance(value, (Tensor, Module)):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (Tensor, Module)):
                        yield f"{name}.{i}", v

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        seen: set[int] = set()
        for name, value in self._children():
            full = f"{prefix}{name}"
            if isinstance(value, Tensor):
                if value.requires_grad and id(value) not in seen:
                    seen.add(id(value))
                    yield full, value
            else:
                for n, p in value.named_parameters(prefix=full + "."):
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield n, p

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def modules(self) -> Iterator["Module"]:
        yield self
        for _, v in self._children():
            if isinstance(v, Module):
                yield from v.modules()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {n: p.data.copy() for n, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        if missing:
            raise KeyError(f"missing keys: {sorted(missing)}")
        for n, p in params.items():
            arr = np.asarray(state[n], dtype=p.data.dtype)
            if arr.shape != p.shape:
                raise ValueError(f"{n}: shape {arr.shape} != {p.shape}")
            p.data[...] = arr

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())


def _uniform(shape, bound, dtype):
    return get_rng().uniform(-bound, bound, size=shape).astype(dtype)


class Linear(Module):
    """y = x W^T + b. Init matches PyTorch's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=np.float32):
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(_uniform((out_features, in_features), bound, dtype), requires_grad=True)
        self.bias = Tensor(_uniform((out_features,), bound, dtype), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class Embedding(Module):
    """Lookup table, initialised N(0, 1) like PyTorch."""

    def __init__(self, num_embeddings: int, dim: int, dtype=np.float32):
        w = get_rng().standard_normal((num_embeddings, dim)).astype(dtype)
        self.weight = Tensor(w, requires_grad=True)

    def forward(self, idx) -> Tensor:
        return F.embedding(self.weight, idx)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype=np.float32):
        self.eps = eps
        self.weight = Tensor(np.ones(dim, dtype=dtype), requires_grad=True)
        self.bias = Tensor(np.zeros(dim, dtype=dtype), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i: int) -> Module:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)
