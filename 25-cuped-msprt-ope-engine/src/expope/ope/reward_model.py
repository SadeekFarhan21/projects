"""Gradient-boosted reward model q(x, a, position) with K-fold cross-fitting.

Cross-fitting: rows in fold k are predicted by a model trained on the other folds, so the
reward model never scores a round it was trained on. This removes the own-observation
overfitting bias that otherwise leaks into DM and DR (Chernozhukov et al. 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np


@dataclass
class RewardModelConfig:
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 50
    reg_lambda: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    n_folds: int = 3
    n_jobs: int = 2
    seed: int = 0


def _features(context: np.ndarray, action: np.ndarray, position: np.ndarray, action_context: np.ndarray | None) -> np.ndarray:
    cols = [context, action[:, None].astype(float), position[:, None].astype(float)]
    if action_context is not None:
        cols.append(action_context[action])
    return np.hstack(cols).astype(np.float32)


def fit_predict_q_hat(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    n_actions: int,
    len_list: int = 1,
    position: np.ndarray | None = None,
    action_context: np.ndarray | None = None,
    cfg: RewardModelConfig = RewardModelConfig(),
) -> np.ndarray:
    """Return cross-fitted q_hat with shape (n, n_actions, len_list).

    The action index is a LightGBM categorical feature (column ``d``), the position is numeric.
    Binary rewards use the logistic objective; anything else uses squared error.
    """
    n = context.shape[0]
    position = np.zeros(n, dtype=int) if position is None else np.asarray(position, dtype=int)
    action = np.asarray(action, dtype=int)
    reward = np.asarray(reward, dtype=float)
    binary = bool(np.isin(reward, (0.0, 1.0)).all())
    d = context.shape[1]
    rng = np.random.default_rng(cfg.seed)
    folds = rng.permutation(n) % cfg.n_folds
    q_hat = np.zeros((n, n_actions, len_list))
    params = dict(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        reg_lambda=cfg.reg_lambda,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )
    X_all = _features(context, action, position, action_context)
    for k in range(cfg.n_folds):
        tr, te = folds != k, folds == k
        model = lgb.LGBMClassifier(**params) if binary else lgb.LGBMRegressor(**params)
        y = reward[tr].astype(int) if binary else reward[tr]
        if binary and np.unique(y).size < 2:
            q_hat[te] = float(y.mean())
            continue
        model.fit(X_all[tr], y, categorical_feature=[d])
        ctx_te = context[te]
        m = ctx_te.shape[0]
        for a in range(n_actions):
            for p in range(len_list):
                X = _features(ctx_te, np.full(m, a), np.full(m, p), action_context)
                q_hat[te, a, p] = model.predict_proba(X)[:, 1] if binary else model.predict(X)
    return q_hat


def per_action_mean_q_hat(action: np.ndarray, reward: np.ndarray, n: int, n_actions: int, len_list: int = 1,
                          position: np.ndarray | None = None) -> np.ndarray:
    """A deliberately weak, context-free reward model: the mean reward of each (action, position).

    Used to show that DM inherits reward-model bias while DR stays unbiased when propensities are known.
    """
    position = np.zeros(action.shape[0], dtype=int) if position is None else position
    overall = reward.mean()
    table = np.full((n_actions, len_list), overall)
    for a in range(n_actions):
        for p in range(len_list):
            m = (action == a) & (position == p)
            if m.any():
                table[a, p] = reward[m].mean()
    return np.broadcast_to(table, (n, n_actions, len_list)).copy()
