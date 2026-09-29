"""Glue between the C++ batched search engines and a PyTorch network."""
from __future__ import annotations

import numpy as np
import torch

from az import c4core


class NetEvaluator:
    """Callable that maps a (B,3,6,7) float32 array to (policy probs, value)."""

    def __init__(self, net: torch.nn.Module | None, device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self.calls = 0
        self.rows = 0

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        self.rows += len(obs)
        if self.net is None:  # uniform prior, zero value
            return np.ones((len(obs), 7), np.float32), np.zeros(len(obs), np.float32)
        x = torch.from_numpy(obs).to(self.device)
        logits, v = self.net(x)
        p = torch.softmax(logits.float(), dim=1)
        return p.cpu().numpy().astype(np.float32), v.float().cpu().numpy().astype(np.float32)


def drive(engine, evaluator: NetEvaluator) -> None:
    """Runs a BatchedGames or BatchedAnalysis engine to completion."""
    while True:
        obs = engine.gather()
        if len(obs) == 0:
            if engine.done():
                return
            continue
        p, v = evaluator(obs)
        engine.scatter(p, v)


def analyze_positions(positions, evaluator: NetEvaluator, num_sims: int, seed: int = 0):
    """PUCT (no noise) on each position. Returns (best_moves, policies)."""
    cfg = c4core.PuctConfig(num_sims=num_sims, dirichlet_eps=0.0)
    eng = c4core.BatchedAnalysis(list(positions), cfg, seed)
    drive(eng, evaluator)
    return np.array(eng.best_moves()), np.array(eng.policies(), dtype=np.float32)
