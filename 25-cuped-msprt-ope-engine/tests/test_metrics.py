from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from expope.metrics import DEFAULT_METRICS, MeanMetric, MetricsEngine, RatioMetric
from expope.sim import EventLogConfig, generate_event_log
from expope.stats import MeanStats, RatioStats, delta_method_ratio, welch_from_stats

T0 = datetime(2026, 3, 1)


def _tiny_engine():
    e = MetricsEngine()
    e.load(
        experiments=pl.DataFrame(
            {
                "experiment_id": ["e1"],
                "control_variant": ["c"],
                "treatment_variant": ["t"],
                "treatment_share": [0.5],
                "start_at": [T0],
                "end_at": [T0 + timedelta(days=7)],
                "pre_period_days": [7],
            }
        ),
        assignments=pl.DataFrame(
            {"experiment_id": ["e1"] * 4, "user_id": [1, 2, 3, 4], "variant": ["c", "c", "t", "t"], "assigned_at": [T0] * 4}
        ),
        exposures=pl.DataFrame(
            {
                "experiment_id": ["e1"] * 5,
                # user 1 exposed twice (first exposure wins), user 4 never exposed.
                "user_id": [1, 1, 2, 3, 3],
                "exposed_at": [T0 + timedelta(days=2), T0 + timedelta(days=1), T0, T0 + timedelta(days=3), T0 - timedelta(days=1)],
            }
        ),
        events=pl.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 2, 3, 3, 3, 4],
                "event_type": ["session", "click", "purchase", "session", "session", "session", "purchase", "session", "session"],
                "ts": [
                    T0 + timedelta(days=1, hours=1),  # post (after first exposure day 1)
                    T0 + timedelta(days=1, hours=2),  # post
                    T0 + timedelta(days=1, hours=3),  # post, revenue 10
                    T0 - timedelta(days=3),  # pre
                    T0 + timedelta(days=8),  # after end, ignored
                    T0 + timedelta(days=3, hours=1),  # post for user 3? first exposure is before start -> excluded unit
                    T0 - timedelta(days=2),
                    T0 - timedelta(days=20),  # before pre window, ignored
                    T0 + timedelta(days=1),
                ],
                "value": [None, None, 10.0, None, None, None, 5.0, None, None],
            }
        ),
    )
    e.build_user_metrics()
    return e


def test_user_metrics_edge_cases():
    e = _tiny_engine()
    um = e.con.execute("SELECT * FROM user_metrics ORDER BY user_id").pl()
    # user 3's first exposure is before the start, so it is not a valid unit; user 4 never exposed.
    assert um["user_id"].to_list() == [1, 2]
    u1 = um.row(0, named=True)
    assert (u1["sessions"], u1["clicks"], u1["purchases"], u1["revenue"]) == (1, 1, 1, 10.0)
    assert u1["pre_sessions"] == 1 and u1["pre_revenue"] == 0.0
    u2 = um.row(1, named=True)  # exposed, but no in-window events: zeros not NULLs
    assert (u2["sessions"], u2["revenue"], u2["pre_sessions"]) == (0, 0.0, 0)
    counts = e.assignment_counts().sort("variant")
    assert counts["assigned"].to_list() == [2, 2]
    assert counts["exposed"].to_list() == [2, 0]


def _reference_user_metrics(d: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Independent polars implementation of the user_metrics stage."""
    x = d["experiments"]
    fe = d["exposures"].group_by("experiment_id", "user_id").agg(pl.col("exposed_at").min())
    units = (
        d["assignments"].join(fe, on=["experiment_id", "user_id"]).join(x, on="experiment_id")
        .filter((pl.col("exposed_at") >= pl.col("start_at")) & (pl.col("exposed_at") < pl.col("end_at")))
        .with_columns((pl.col("start_at") - pl.duration(days=pl.col("pre_period_days"))).alias("pre_start"))
    )
    ev = units.select("experiment_id", "user_id", "exposed_at", "start_at", "end_at", "pre_start").join(d["events"], on="user_id")

    def agg(df, prefix):
        return df.group_by("experiment_id", "user_id").agg(
            (pl.col("event_type") == "session").sum().alias(f"{prefix}sessions"),
            (pl.col("event_type") == "click").sum().alias(f"{prefix}clicks"),
            (pl.col("event_type") == "purchase").sum().alias(f"{prefix}purchases"),
            pl.col("value").filter(pl.col("event_type") == "purchase").sum().alias(f"{prefix}revenue"),
        )

    post = agg(ev.filter((pl.col("ts") >= pl.col("exposed_at")) & (pl.col("ts") < pl.col("end_at"))), "")
    pre = agg(ev.filter((pl.col("ts") >= pl.col("pre_start")) & (pl.col("ts") < pl.col("start_at"))), "pre_")
    out = units.select("experiment_id", "user_id", "variant").join(post, on=["experiment_id", "user_id"], how="left")
    out = out.join(pre, on=["experiment_id", "user_id"], how="left").fill_null(0)
    return out.sort("experiment_id", "user_id")


def test_user_metrics_matches_polars_reference():
    d = generate_event_log(3, EventLogConfig(n_users=400, session_lift=0.2), seed=7)
    e = MetricsEngine()
    e.load(**d)
    e.build_user_metrics()
    got = e.con.execute("SELECT * FROM user_metrics ORDER BY experiment_id, user_id").pl()
    ref = _reference_user_metrics(d)
    assert got.height == ref.height > 0
    for c in ["user_id", "sessions", "clicks", "purchases", "pre_sessions", "pre_clicks", "pre_purchases"]:
        assert got[c].cast(pl.Int64).to_list() == ref[c].cast(pl.Int64).to_list(), c
    np.testing.assert_allclose(got["revenue"].to_numpy(), ref["revenue"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(got["pre_revenue"].to_numpy(), ref["pre_revenue"].to_numpy(), rtol=1e-12)


def test_sql_sufficient_stats_give_same_tests_as_arrays():
    d = generate_event_log(2, EventLogConfig(n_users=600, session_lift=0.3), seed=3)
    e = MetricsEngine()
    e.load(**d)
    e.build_user_metrics()
    res = e.analyze()
    um = e.con.execute("SELECT * FROM user_metrics").pl()
    for eid in ["exp00000", "exp00001"]:
        g = um.filter(pl.col("experiment_id") == eid)
        c, t = g.filter(pl.col("variant") == "control"), g.filter(pl.col("variant") == "treatment")
        w = welch_from_stats(MeanStats.from_array(c["revenue"].to_numpy()), MeanStats.from_array(t["revenue"].to_numpy()))
        row = res.filter((pl.col("experiment_id") == eid) & (pl.col("metric") == "revenue_per_user") & (pl.col("method") == "welch"))
        assert row["p_value"][0] == pytest.approx(w.p_value, rel=1e-9)
        dm = delta_method_ratio(
            RatioStats.from_arrays(c["clicks"].to_numpy(), c["sessions"].to_numpy()),
            RatioStats.from_arrays(t["clicks"].to_numpy(), t["sessions"].to_numpy()),
        )
        row = res.filter((pl.col("experiment_id") == eid) & (pl.col("metric") == "ctr_per_session"))
        assert row["estimate"][0] == pytest.approx(dm.estimate, rel=1e-9)
        assert row["se"][0] == pytest.approx(dm.se, rel=1e-9)
    # A 30 percent session lift is detected on sessions_per_user in both experiments.
    s = res.filter((pl.col("metric") == "sessions_per_user") & (pl.col("method") == "cuped"))
    assert (s["p_value"] < 0.001).all() and (s["estimate"] > 0).all()


def test_custom_metric_definition():
    d = generate_event_log(1, EventLogConfig(n_users=300), seed=1)
    e = MetricsEngine()
    e.load(**d)
    e.build_user_metrics()
    metrics = [MeanMetric("any_click", "CAST(clicks > 0 AS DOUBLE)"), RatioMetric("aov", "revenue", "purchases")]
    r = e.analyze(metrics)
    assert set(r["metric"].to_list()) == {"any_click", "aov"}
    assert len(DEFAULT_METRICS) == 5


def test_rejects_unknown_table():
    with pytest.raises(ValueError):
        MetricsEngine().load(bogus=pl.DataFrame({"a": [1]}))
