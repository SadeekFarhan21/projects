"""Matches against baselines, confidence intervals and Elo estimates."""
from __future__ import annotations

import math
import time

import numpy as np

from az import c4core
from az.search import NetEvaluator, drive


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (k may be fractional)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def elo_from_score(s: float) -> float:
    """Elo difference implied by an expected score s in (0, 1)."""
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / s - 1.0)


def play_vs_mcts(
    evaluator: NetEvaluator,
    rollouts: int,
    games: int,
    num_sims: int,
    seed: int,
    slots: int | None = None,
    openings: list[str] | None = None,
) -> dict:
    """Network+PUCT (argmax move, no noise) vs PureMCTS(rollouts).

    The network plays first in even-numbered games and second in odd ones.
    """
    cfg = c4core.PuctConfig(num_sims=num_sims, dirichlet_eps=0.0, temp_moves=0)
    eng = c4core.BatchedGames(slots or min(games, 256), games, cfg, seed, rollouts, openings or [])
    t0 = time.time()
    drive(eng, evaluator)
    recs = eng.drain_games()
    w = sum(r.winner == r.net_side for r in recs)
    d = sum(r.winner == -1 for r in recs)
    losses = len(recs) - w - d
    score = (w + 0.5 * d) / len(recs)
    lo, hi = wilson(w + 0.5 * d, len(recs))
    wlo, whi = wilson(w, len(recs))
    first = [r for r in recs if r.net_side == 0]
    second = [r for r in recs if r.net_side == 1]
    return {
        "opponent": f"mcts{rollouts}",
        "rollouts": rollouts,
        "net_sims": num_sims,
        "games": len(recs),
        "wins": w,
        "draws": d,
        "losses": losses,
        "win_rate": w / len(recs),
        "win_rate_lo": wlo,
        "win_rate_hi": whi,
        "score": score,
        "score_lo": lo,
        "score_hi": hi,
        "elo_diff": elo_from_score(score),
        "elo_diff_lo": elo_from_score(lo),
        "elo_diff_hi": elo_from_score(hi),
        "win_rate_as_first": float(np.mean([r.winner == 0 for r in first])) if first else float("nan"),
        "win_rate_as_second": float(np.mean([r.winner == 1 for r in second])) if second else float("nan"),
        "seconds": time.time() - t0,
    }


def bradley_terry(names: list[str], results: list[tuple[int, int, float, int]], anchor: str, iters: int = 2000):
    """Fits Elo ratings from pairwise results.

    results: (i, j, score_of_i, n_games). Draws count as half points. Returns
    ratings with `anchor` fixed at 0. Uses gradient ascent on the log
    likelihood with a tiny ridge term so perfect scores stay finite.
    """
    k = len(names)
    r = np.zeros(k)
    scale = math.log(10) / 400.0
    for _ in range(iters):
        g = -1e-4 * r
        for i, j, s, n in results:
            p = 1.0 / (1.0 + math.exp(-scale * (r[i] - r[j])))
            g[i] += n * (s - p)
            g[j] -= n * (s - p)
        r += 20.0 * g / max(1, sum(n for *_, n in results)) * k
    r -= r[names.index(anchor)]
    return dict(zip(names, r.tolist()))
