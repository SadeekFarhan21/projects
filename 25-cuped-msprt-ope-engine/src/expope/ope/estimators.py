"""From-scratch off-policy estimators for contextual bandits with optional slate positions.

Array conventions (chosen to match Open Bandit Pipeline so inputs are interchangeable):
    reward       (n,)        observed reward r_t
    action       (n,)        logged action a_t, integer in [0, A)
    pscore       (n,)        behaviour propensity pi_b(a_t | x_t) (per position for slates)
    action_dist  (n, A, L)   evaluation policy pi_e(a | x_t) at each of L positions
    position     (n,)        slot index in [0, L); None means L = 1
    q_hat        (n, A, L)   reward-model prediction q(x_t, a) at each position

Every estimator here is a mean of per-round terms, or (SNIPS) a ratio of two means. The
per-round terms are exposed so that the bootstrap can resample rounds without recomputing
the (n, A, L) tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RowTerms:
    """Per-round quantities that every estimator is built from."""

    reward: np.ndarray  # r_t
    w: np.ndarray  # importance weight pi_e(a_t|x_t) / pi_b(a_t|x_t)
    dm: np.ndarray | None = None  # sum_a pi_e(a|x_t) q(x_t, a)
    q_factual: np.ndarray | None = None  # q(x_t, a_t)


def _position(action_dist: np.ndarray, position: np.ndarray | None) -> np.ndarray:
    if position is None:
        return np.zeros(action_dist.shape[0], dtype=int)
    return np.asarray(position, dtype=int)


def _validate(reward, action, pscore, action_dist, position):
    n = reward.shape[0]
    if action_dist.ndim != 3:
        raise ValueError("action_dist must have shape (n, A, L)")
    if not (action.shape[0] == pscore.shape[0] == action_dist.shape[0] == n):
        raise ValueError("reward, action, pscore and action_dist must have the same n")
    if np.any(pscore <= 0):
        raise ValueError("pscore must be strictly positive")
    if position is not None and (position.min() < 0 or position.max() >= action_dist.shape[2]):
        raise ValueError("position out of range")
    if action.min() < 0 or action.max() >= action_dist.shape[1]:
        raise ValueError("action out of range")


def row_terms(
    reward: np.ndarray,
    action: np.ndarray,
    pscore: np.ndarray,
    action_dist: np.ndarray,
    position: np.ndarray | None = None,
    q_hat: np.ndarray | None = None,
) -> RowTerms:
    reward = np.asarray(reward, dtype=float)
    action = np.asarray(action, dtype=int)
    pscore = np.asarray(pscore, dtype=float)
    _validate(reward, action, pscore, action_dist, position)
    pos = _position(action_dist, position)
    idx = np.arange(reward.shape[0])
    w = action_dist[idx, action, pos] / pscore
    if q_hat is None:
        return RowTerms(reward, w)
    pi_at_pos = action_dist[idx, :, pos]  # (n, A)
    q_at_pos = q_hat[idx, :, pos]  # (n, A)
    dm = (q_at_pos * pi_at_pos).sum(axis=1) / pi_at_pos.sum(axis=1)
    return RowTerms(reward, w, dm, q_hat[idx, action, pos])


# ---- per-round contributions. Each returns (numerator_terms, denominator_terms or None).


def ips_terms(t: RowTerms):
    return t.w * t.reward, None


def snips_terms(t: RowTerms):
    return t.w * t.reward, t.w


def dm_terms(t: RowTerms):
    _need_q(t)
    return t.dm, None


def dr_terms(t: RowTerms):
    _need_q(t)
    return t.dm + t.w * (t.reward - t.q_factual), None


def switch_dr_terms(t: RowTerms, tau: float):
    _need_q(t)
    keep = (t.w <= tau).astype(float)
    return t.dm + keep * t.w * (t.reward - t.q_factual), None


def _need_q(t: RowTerms) -> None:
    if t.dm is None:
        raise ValueError("this estimator needs q_hat")


def _value(num: np.ndarray, den: np.ndarray | None) -> float:
    if den is None:
        return float(num.mean())
    return float(num.mean() / den.mean())


# ---- public one-shot estimators


def ips(reward, action, pscore, action_dist, position=None) -> float:
    return _value(*ips_terms(row_terms(reward, action, pscore, action_dist, position)))


def snips(reward, action, pscore, action_dist, position=None) -> float:
    return _value(*snips_terms(row_terms(reward, action, pscore, action_dist, position)))


def direct_method(action_dist, q_hat, position=None) -> float:
    n = action_dist.shape[0]
    pos = _position(action_dist, position)
    idx = np.arange(n)
    pi = action_dist[idx, :, pos]
    return float(((q_hat[idx, :, pos] * pi).sum(axis=1) / pi.sum(axis=1)).mean())


def doubly_robust(reward, action, pscore, action_dist, q_hat, position=None) -> float:
    return _value(*dr_terms(row_terms(reward, action, pscore, action_dist, position, q_hat)))


def switch_dr(reward, action, pscore, action_dist, q_hat, position=None, tau: float = 1.0) -> float:
    return _value(*switch_dr_terms(row_terms(reward, action, pscore, action_dist, position, q_hat), tau))


ESTIMATORS = ("ips", "snips", "dm", "dr", "switch_dr")


def estimator_terms(name: str, t: RowTerms, tau: float = 1.0):
    if name == "ips":
        return ips_terms(t)
    if name == "snips":
        return snips_terms(t)
    if name == "dm":
        return dm_terms(t)
    if name == "dr":
        return dr_terms(t)
    if name == "switch_dr":
        return switch_dr_terms(t, tau)
    raise KeyError(name)


def estimate_all(t: RowTerms, tau: float = 1.0) -> dict[str, float]:
    names = ESTIMATORS if t.dm is not None else ("ips", "snips")
    return {k: _value(*estimator_terms(k, t, tau)) for k in names}


# ---- bootstrap


def bootstrap_ci_obp_compatible(
    samples: np.ndarray, alpha: float = 0.05, n_bootstrap_samples: int = 10000, random_state: int | None = None
) -> dict[str, float]:
    """Percentile bootstrap of the mean of per-round terms, bit-compatible with obp.

    obp draws with ``RandomState(seed).choice(samples, size=n)`` once per replicate. Legacy
    ``choice`` with replacement and uniform p draws ``randint(0, n, size=n)`` indices, which we
    reproduce here so the two libraries agree to floating-point rounding on identical inputs.
    Note that for SNIPS this resamples already-normalised terms and ignores the variability of
    the normaliser; :func:`bootstrap_ci` below resamples rounds and recomputes the ratio.
    """
    samples = np.asarray(samples, dtype=float)
    rs = np.random.RandomState(random_state)
    n = samples.shape[0]
    boot = np.empty(n_bootstrap_samples)
    for b in range(n_bootstrap_samples):
        boot[b] = samples[rs.randint(0, n, size=n)].mean()
    return {
        "mean": float(boot.mean()),
        "lower": float(np.percentile(boot, 100 * (alpha / 2))),
        "upper": float(np.percentile(boot, 100 * (1.0 - alpha / 2))),
    }


def bootstrap_ci(
    num: np.ndarray,
    den: np.ndarray | None = None,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int | None = None,
    chunk: int = 250,
) -> dict[str, float]:
    """Percentile bootstrap CI for mean(num) or mean(num) / mean(den) by resampling rounds.

    Resampling with replacement is represented by multinomial count vectors, so each replicate
    is a dot product with the per-round terms and the ratio estimators are recomputed exactly.
    """
    num = np.asarray(num, dtype=float)
    n = num.shape[0]
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    p = np.full(n, 1.0 / n)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        counts = rng.multinomial(n, p, size=b).astype(float)
        top = counts @ num
        reps[s : s + b] = top / (counts @ den) if den is not None else top / n
    return {
        "estimate": _value(num, den),
        "lower": float(np.percentile(reps, 100 * alpha / 2)),
        "upper": float(np.percentile(reps, 100 * (1 - alpha / 2))),
        "se": float(reps.std(ddof=1)),
    }
