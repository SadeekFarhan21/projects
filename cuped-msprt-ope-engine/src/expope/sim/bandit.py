"""Synthetic contextual bandit with a known evaluation-policy value.

q(x, a) = sigmoid(x . theta_a + b_a + interaction term), x ~ N(0, I_d).
Behaviour and evaluation policies are softmax policies over the same true logits with
different inverse temperatures, so the evaluation policy is shifted toward good actions and
the importance weights have a realistic spread. Ground truth is E_x[ sum_a pi_e(a|x) q(x,a) ],
estimated on a large independent context sample without reward noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class SyntheticBandit:
    n_actions: int = 10
    dim: int = 5
    beta_behavior: float = 0.5
    beta_eval: float = 3.0
    eval_epsilon: float = 0.1
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.theta = rng.normal(0, 0.5, size=(self.dim, self.n_actions))
        self.bias = rng.normal(-1.0, 0.5, size=self.n_actions)
        self.inter = rng.normal(0, 0.5, size=(self.n_actions,))

    def q(self, x: np.ndarray) -> np.ndarray:
        """True expected reward, shape (n, A). Includes a nonlinear term a GBM must learn."""
        lin = x @ self.theta + self.bias
        nonlin = np.sin(x[:, [0]] * 2.0) * self.inter[None, :]
        return _sigmoid(lin + nonlin)

    def _logits(self, x):
        return np.log(self.q(x))

    def pi_b(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.beta_behavior * self._logits(x))

    def pi_e(self, x: np.ndarray) -> np.ndarray:
        soft = _softmax(self.beta_eval * self._logits(x))
        return (1 - self.eval_epsilon) * soft + self.eval_epsilon / self.n_actions

    def true_value(self, n: int = 1_000_000, seed: int = 12345) -> float:
        rng = np.random.default_rng(seed)
        total, done = 0.0, 0
        while done < n:
            m = min(200_000, n - done)
            x = rng.normal(size=(m, self.dim))
            total += float((self.pi_e(x) * self.q(x)).sum())
            done += m
        return total / n

    def sample_logs(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        x = rng.normal(size=(n, self.dim))
        pb = self.pi_b(x)
        u = rng.random(n)[:, None]
        action = (pb.cumsum(axis=1) < u).sum(axis=1)
        action = np.minimum(action, self.n_actions - 1)
        q = self.q(x)
        reward = (rng.random(n) < q[np.arange(n), action]).astype(float)
        return {
            "context": x,
            "action": action,
            "reward": reward,
            "pscore": pb[np.arange(n), action],
            "action_dist": self.pi_e(x)[:, :, None],
            "q_true": q[:, :, None],
        }
