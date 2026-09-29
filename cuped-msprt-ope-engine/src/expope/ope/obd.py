"""Open Bandit Dataset (ZOZOTOWN) loading and the Bernoulli Thompson Sampling evaluation policy.

The loader reads the raw CSVs with polars and builds features itself, so the from-scratch
pipeline does not depend on obp. See scripts/fetch_obd.py for how data/obd is populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

N_ACTIONS = 80
LEN_LIST = 3


@dataclass
class OBDLog:
    behavior_policy: str
    campaign: str
    context: np.ndarray  # (n, d) float
    action: np.ndarray  # (n,) item index
    position: np.ndarray  # (n,) 0-based slot
    reward: np.ndarray  # (n,) click
    pscore: np.ndarray  # (n,) logged propensity
    n_actions: int
    len_list: int

    @property
    def n(self) -> int:
        return self.reward.shape[0]


def load_obd(data_dir: Path, behavior_policy: str = "random", campaign: str = "all") -> OBDLog:
    path = Path(data_dir) / behavior_policy / campaign / f"{campaign}.csv"
    df = pl.read_csv(path)
    df = df.sort("timestamp")
    user_cols = [c for c in df.columns if c.startswith("user_feature")]
    aff_cols = [c for c in df.columns if c.startswith("user-item_affinity")]
    # One-hot user features with a fixed, sorted category order (drop the first level like obp).
    blocks = []
    for c in user_cols:
        levels = sorted(df[c].unique().to_list())[1:]
        col = df[c].to_numpy()
        blocks.append(np.stack([(col == lv) for lv in levels], axis=1).astype(float) if levels else np.zeros((df.height, 0)))
    blocks.append(df.select(aff_cols).to_numpy().astype(float))
    context = np.hstack(blocks)
    pos_raw = df["position"].to_numpy()
    position = np.searchsorted(np.unique(pos_raw), pos_raw)
    return OBDLog(
        behavior_policy=behavior_policy,
        campaign=campaign,
        context=context,
        action=df["item_id"].to_numpy().astype(int),
        position=position.astype(int),
        reward=df["click"].to_numpy().astype(float),
        pscore=df["propensity_score"].to_numpy().astype(float),
        n_actions=N_ACTIONS,
        len_list=LEN_LIST,
    )


def load_bts_prior(path: Path, campaign: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """Parse the ZOZOTOWN production Beta prior (a tiny YAML: campaign -> alpha/beta lists).

    Parsed by hand to avoid a YAML dependency in the core package.
    """
    alpha: dict[str, list[float]] = {}
    cur_campaign = cur_key = None
    for raw in Path(path).read_text().splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            cur_campaign = line[:-1]
        elif line.endswith(":"):
            cur_key = f"{cur_campaign}.{line[:-1]}"
            alpha[cur_key] = []
        elif line.startswith("-") and cur_key:
            alpha[cur_key].append(float(line[1:].strip()))
    return np.array(alpha[f"{campaign}.alpha"]), np.array(alpha[f"{campaign}.beta"])


def bts_action_dist(alpha: np.ndarray, beta: np.ndarray, len_list: int = LEN_LIST, n_sim: int = 100_000,
                    seed: int = 12345, chunk: int = 20_000) -> np.ndarray:
    """Monte Carlo slate distribution of Bernoulli Thompson Sampling.

    Each simulation draws theta_a ~ Beta(alpha_a, beta_a) and ranks items by theta descending;
    returns P(item a shown at slot k), shape (A, L). Vectorised version of obp's loop.
    """
    rng = np.random.default_rng(seed)
    a = alpha.shape[0]
    counts = np.zeros((a, len_list))
    done = 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        theta = rng.beta(alpha, beta, size=(m, a))
        top = np.argsort(-theta, axis=1, kind="stable")[:, :len_list]
        for k in range(len_list):
            counts[:, k] += np.bincount(top[:, k], minlength=a)
        done += m
    return counts / n_sim
