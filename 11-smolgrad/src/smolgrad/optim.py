"""Optimizers. Updates are done in place on `p.data` outside the graph, with
the exact update rules of torch.optim.SGD and torch.optim.AdamW so the two can
be compared step for step in tests."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params, lr: float):
        self.params: list[Tensor] = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """v <- mu * v + (g + wd * p);  p <- p - lr * v   (PyTorch's convention: the
    first step initialises v = g rather than (1 - dampening) * g)."""

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: list[np.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v = self.velocity[i]
                if v is None:
                    v = self.velocity[i] = g.copy()
                else:
                    v *= self.momentum
                    v += g
                g = v
            p.data -= self.lr * g


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019):
    the decay multiplies the weights directly instead of being added to the
    gradient, so it is not rescaled by the adaptive denominator."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        # One scratch buffer per parameter so a step allocates nothing. The
        # naive expression form allocated about six parameter-sized temporaries
        # per tensor per step (see DEVLOG.md, Problems).
        self._tmp = [np.empty_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        inv_sqrt_bc2 = float(1.0 / np.sqrt(1.0 - self.b2 ** self.t))  # python float keeps float32 math
        step_size = self.lr / bc1
        for p, m, v, tmp in zip(self.params, self.m, self.v, self._tmp):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                p.data *= 1.0 - self.lr * self.weight_decay
            np.multiply(g, 1.0 - self.b1, out=tmp)      # m = b1*m + (1-b1)*g
            m *= self.b1
            m += tmp
            np.multiply(g, g, out=tmp)                  # v = b2*v + (1-b2)*g^2
            tmp *= 1.0 - self.b2
            v *= self.b2
            v += tmp
            np.sqrt(v, out=tmp)                         # denom = sqrt(v)/sqrt(bc2) + eps
            tmp *= inv_sqrt_bc2
            tmp += self.eps
            np.divide(m, tmp, out=tmp)                  # p -= lr/bc1 * m/denom
            tmp *= step_size
            p.data -= tmp
