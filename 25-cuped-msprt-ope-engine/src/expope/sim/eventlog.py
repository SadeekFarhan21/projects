"""Synthetic event logs in the metrics-layer schema, with a known treatment effect.

Each user has a latent activity rate lambda ~ Gamma(shape, scale). Sessions in the pre period
and in the experiment window are Poisson(lambda * days), so pre-period behaviour is a real
covariate for CUPED. Every session can produce a click (per-user click probability) and a
purchase (global probability) whose revenue is log-normal, which gives heavy tails.

A treatment multiplies the post-period session rate by (1 + session_lift). With
session_lift = 0 the generated data is an A/A test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

EPOCH = datetime(2026, 1, 1)


@dataclass(frozen=True)
class EventLogConfig:
    n_users: int = 2000
    treatment_share: float = 0.5
    exposure_rate: float = 0.9
    pre_days: int = 14
    days: int = 14
    gamma_shape: float = 2.0
    gamma_scale: float = 0.25  # mean lambda 0.5 sessions per day
    click_beta: tuple[float, float] = (2.0, 8.0)
    purchase_prob: float = 0.03
    revenue_lognorm: tuple[float, float] = (3.0, 1.0)
    session_lift: float = 0.0


def _events_for(user_ids: np.ndarray, sessions: np.ndarray, pclick: np.ndarray, t0: np.ndarray, span_s: np.ndarray,
                cfg: EventLogConfig, rng: np.random.Generator) -> list[pl.DataFrame]:
    uid = np.repeat(user_ids, sessions)
    start = np.repeat(t0, sessions)
    span = np.repeat(span_s, sessions)
    pc = np.repeat(pclick, sessions)
    ts = start + rng.random(uid.size) * span
    clicked = rng.random(uid.size) < pc
    bought = rng.random(uid.size) < cfg.purchase_prob
    rev = rng.lognormal(cfg.revenue_lognorm[0], cfg.revenue_lognorm[1], size=int(bought.sum()))
    frames = [
        pl.DataFrame({"user_id": uid, "event_type": "session", "ts_s": ts, "value": np.full(uid.size, np.nan)}),
        pl.DataFrame({"user_id": uid[clicked], "event_type": "click", "ts_s": ts[clicked] + 1.0, "value": np.full(int(clicked.sum()), np.nan)}),
        pl.DataFrame({"user_id": uid[bought], "event_type": "purchase", "ts_s": ts[bought] + 2.0, "value": rev}),
    ]
    return frames


def generate_event_log(n_experiments: int, cfg: EventLogConfig = EventLogConfig(), seed: int = 0) -> dict[str, pl.DataFrame]:
    """Generate ``n_experiments`` independent experiments, each on its own population of users.

    Returns polars frames keyed by table name, ready for :meth:`MetricsEngine.load`.
    """
    rng = np.random.default_rng(seed)
    n = cfg.n_users
    total = n_experiments * n
    exp_idx = np.repeat(np.arange(n_experiments), n)
    user_id = np.arange(total, dtype=np.int64)
    treated = rng.random(total) < cfg.treatment_share
    exposed = rng.random(total) < cfg.exposure_rate
    lam = rng.gamma(cfg.gamma_shape, cfg.gamma_scale, size=total)
    pclick = rng.beta(*cfg.click_beta, size=total)

    day = 86400.0
    start_s = cfg.pre_days * day
    end_s = start_s + cfg.days * day
    # First exposure uniformly in the first half of the window.
    exposure_s = start_s + rng.random(total) * (cfg.days * day / 2)

    pre_sessions = rng.poisson(lam * cfg.pre_days)
    post_rate = lam * np.where(treated, 1.0 + cfg.session_lift, 1.0)
    # Post-period sessions only occur after exposure (for exposed users) to keep the effect clean.
    post_span = np.where(exposed, end_s - exposure_s, end_s - start_s)
    post_start = np.where(exposed, exposure_s, start_s)
    post_sessions = rng.poisson(post_rate * post_span / day)

    ev = _events_for(user_id, pre_sessions, pclick, np.zeros(total), np.full(total, start_s), cfg, rng)
    ev += _events_for(user_id, post_sessions, pclick, post_start, post_span, cfg, rng)
    events = pl.concat(ev).with_columns(
        (pl.lit(EPOCH) + pl.duration(microseconds=(pl.col("ts_s") * 1e6).cast(pl.Int64))).alias("ts"),
        pl.col("value").fill_nan(None),
    ).select("user_id", "event_type", "ts", "value")

    exp_ids = np.array([f"exp{i:05d}" for i in range(n_experiments)])
    start_at = EPOCH + timedelta(days=cfg.pre_days)
    experiments = pl.DataFrame(
        {
            "experiment_id": exp_ids,
            "control_variant": "control",
            "treatment_variant": "treatment",
            "treatment_share": cfg.treatment_share,
            "start_at": start_at,
            "end_at": start_at + timedelta(days=cfg.days),
            "pre_period_days": cfg.pre_days,
        }
    )
    assignments = pl.DataFrame(
        {
            "experiment_id": exp_ids[exp_idx],
            "user_id": user_id,
            "variant": np.where(treated, "treatment", "control"),
            "assigned_at": start_at,
        }
    )
    exposures = pl.DataFrame(
        {
            "experiment_id": exp_ids[exp_idx][exposed],
            "user_id": user_id[exposed],
            "exposed_s": exposure_s[exposed],
        }
    ).with_columns(
        (pl.lit(EPOCH) + pl.duration(microseconds=(pl.col("exposed_s") * 1e6).cast(pl.Int64))).alias("exposed_at")
    ).select("experiment_id", "user_id", "exposed_at")
    return {"experiments": experiments, "assignments": assignments, "exposures": exposures, "events": events}
