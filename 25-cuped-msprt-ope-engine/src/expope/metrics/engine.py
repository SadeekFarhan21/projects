"""DuckDB metrics layer.

Pipeline
    raw tables (experiments, assignments, exposures, events)
      -> user_metrics   one row per (experiment, exposed user) with post and pre-period aggregates
      -> suff_stats     one row per (experiment, variant) with the sums each test needs
      -> analyze()      Python statistics on O(1) numbers per arm

Metric definitions are SQL expressions over the columns of ``user_metrics``. They are trusted
configuration written by the analyst, never user input, so they are spliced into SQL text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable, Sequence

import duckdb
import polars as pl

from ..stats import (
    CovariateStats,
    MeanStats,
    RatioStats,
    cuped_from_stats,
    delta_method_ratio,
    srm_check,
    welch_from_stats,
)

# ------------------------------------------------------------------ definitions


@dataclass(frozen=True)
class UserAggregate:
    """A per-user aggregate over events. Computed for both the post and the pre period.

    ``sql`` is an aggregate expression over the ``events`` columns (event_type, ts, value).
    The pre-period copy is exposed as ``pre_<name>``.
    """

    name: str
    sql: str


DEFAULT_AGGREGATES: tuple[UserAggregate, ...] = (
    UserAggregate("sessions", "COUNT(*) FILTER (WHERE event_type = 'session')"),
    UserAggregate("clicks", "COUNT(*) FILTER (WHERE event_type = 'click')"),
    UserAggregate("purchases", "COUNT(*) FILTER (WHERE event_type = 'purchase')"),
    UserAggregate("revenue", "COALESCE(SUM(value) FILTER (WHERE event_type = 'purchase'), 0.0)"),
)


@dataclass(frozen=True)
class MeanMetric:
    """Per-user mean metric, for example revenue per user or conversion rate.

    If ``covariate`` is given the metric is also analysed with CUPED using that pre-period expression.
    """

    name: str
    value: str
    covariate: str | None = None
    kind: str = field(default="mean", init=False)


@dataclass(frozen=True)
class RatioMetric:
    """Ratio of sums across users, for example clicks per session. Analysed with the delta method."""

    name: str
    numerator: str
    denominator: str
    kind: str = field(default="ratio", init=False)


Metric = MeanMetric | RatioMetric

DEFAULT_METRICS: tuple[Metric, ...] = (
    MeanMetric("revenue_per_user", "revenue", covariate="pre_revenue"),
    MeanMetric("conversion", "CAST(purchases > 0 AS DOUBLE)", covariate="CAST(pre_purchases > 0 AS DOUBLE)"),
    MeanMetric("sessions_per_user", "sessions", covariate="pre_sessions"),
    RatioMetric("ctr_per_session", "clicks", "sessions"),
    RatioMetric("revenue_per_session", "revenue", "sessions"),
)


def schema_sql() -> str:
    return resources.files("expope.metrics").joinpath("schema.sql").read_text()


# ---------------------------------------------------------------------- engine


class MetricsEngine:
    """Thin wrapper around a DuckDB connection holding the event log."""

    def __init__(self, con: duckdb.DuckDBPyConnection | None = None, aggregates: Sequence[UserAggregate] = DEFAULT_AGGREGATES):
        self.con = con if con is not None else duckdb.connect()
        self.aggregates = tuple(aggregates)
        self.con.execute(schema_sql())

    # -- loading
    def load(self, **tables: pl.DataFrame) -> None:
        """Append polars frames into the named tables (experiments, assignments, exposures, events)."""
        for name, df in tables.items():
            if name not in {"experiments", "assignments", "exposures", "events"}:
                raise ValueError(f"unknown table {name}")
            self.con.register("_incoming", df.to_arrow())
            cols = ", ".join(df.columns)
            self.con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _incoming")
            self.con.unregister("_incoming")

    # -- stage 1
    def user_metrics_sql(self) -> str:
        post_cols = ",\n        ".join(f"{a.sql} AS {a.name}" for a in self.aggregates)
        pre_cols = ",\n        ".join(f"{a.sql} AS pre_{a.name}" for a in self.aggregates)
        final_cols = ",\n    ".join(
            [f"COALESCE(post.{a.name}, 0) AS {a.name}" for a in self.aggregates]
            + [f"COALESCE(pre.pre_{a.name}, 0) AS pre_{a.name}" for a in self.aggregates]
        )
        return f"""
WITH first_exposure AS (
    SELECT experiment_id, user_id, MIN(exposed_at) AS exposed_at
    FROM exposures GROUP BY experiment_id, user_id
),
units AS (
    SELECT a.experiment_id, a.user_id, a.variant, f.exposed_at,
           x.start_at, x.end_at,
           x.start_at - to_days(x.pre_period_days) AS pre_start
    FROM assignments a
    JOIN first_exposure f USING (experiment_id, user_id)
    JOIN experiments x USING (experiment_id)
    WHERE f.exposed_at >= x.start_at AND f.exposed_at < x.end_at
),
post AS (
    SELECT u.experiment_id, u.user_id,
        {post_cols}
    FROM units u JOIN events ev
      ON ev.user_id = u.user_id AND ev.ts >= u.exposed_at AND ev.ts < u.end_at
    GROUP BY u.experiment_id, u.user_id
),
pre AS (
    SELECT u.experiment_id, u.user_id,
        {pre_cols}
    FROM units u JOIN events ev
      ON ev.user_id = u.user_id AND ev.ts >= u.pre_start AND ev.ts < u.start_at
    GROUP BY u.experiment_id, u.user_id
)
SELECT u.experiment_id, u.user_id, u.variant,
    {final_cols}
FROM units u
LEFT JOIN post USING (experiment_id, user_id)
LEFT JOIN pre USING (experiment_id, user_id)
"""

    def build_user_metrics(self) -> None:
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE user_metrics AS {self.user_metrics_sql()}")

    # -- stage 2
    @staticmethod
    def suff_stats_sql(metrics: Sequence[Metric]) -> str:
        parts: list[str] = []
        for m in metrics:
            p = m.name
            if isinstance(m, MeanMetric):
                v = f"({m.value})"
                parts += [f"SUM({v})::DOUBLE AS {p}__sum_y", f"SUM({v}*{v})::DOUBLE AS {p}__sum_y2"]
                if m.covariate:
                    x = f"({m.covariate})"
                    parts += [
                        f"SUM({x})::DOUBLE AS {p}__sum_x",
                        f"SUM({x}*{x})::DOUBLE AS {p}__sum_x2",
                        f"SUM({x}*{v})::DOUBLE AS {p}__sum_xy",
                    ]
            else:
                x, y = f"({m.numerator})", f"({m.denominator})"
                parts += [
                    f"SUM({x})::DOUBLE AS {p}__sum_x",
                    f"SUM({y})::DOUBLE AS {p}__sum_y",
                    f"SUM({x}*{x})::DOUBLE AS {p}__sum_x2",
                    f"SUM({y}*{y})::DOUBLE AS {p}__sum_y2",
                    f"SUM({x}*{y})::DOUBLE AS {p}__sum_xy",
                ]
        cols = ",\n    ".join(parts)
        return f"""
SELECT experiment_id, variant, COUNT(*)::DOUBLE AS n,
    {cols}
FROM user_metrics
GROUP BY experiment_id, variant
ORDER BY experiment_id, variant
"""

    def sufficient_stats(self, metrics: Sequence[Metric] = DEFAULT_METRICS) -> pl.DataFrame:
        return self.con.execute(self.suff_stats_sql(metrics)).pl()

    def assignment_counts(self) -> pl.DataFrame:
        """Assigned and exposed counts per (experiment, variant) for SRM checks."""
        return self.con.execute(
            """
            SELECT a.experiment_id, a.variant,
                   COUNT(*)::BIGINT AS assigned,
                   COUNT(u.user_id)::BIGINT AS exposed
            FROM assignments a
            LEFT JOIN (SELECT experiment_id, user_id FROM user_metrics) u USING (experiment_id, user_id)
            GROUP BY a.experiment_id, a.variant
            ORDER BY a.experiment_id, a.variant
            """
        ).pl()

    def experiments(self) -> pl.DataFrame:
        return self.con.execute("SELECT * FROM experiments ORDER BY experiment_id").pl()

    # -- stage 3
    def analyze(self, metrics: Sequence[Metric] = DEFAULT_METRICS, alpha: float = 0.05) -> pl.DataFrame:
        return analyze(self.sufficient_stats(metrics), self.assignment_counts(), self.experiments(), metrics, alpha)


# -------------------------------------------------------------------- analysis


def _row_stats(row: dict, m: Metric, cuped: bool = False):
    p = m.name
    if isinstance(m, RatioMetric):
        return RatioStats(row["n"], row[f"{p}__sum_x"], row[f"{p}__sum_y"], row[f"{p}__sum_x2"], row[f"{p}__sum_y2"], row[f"{p}__sum_xy"])
    if cuped:
        return CovariateStats(row["n"], row[f"{p}__sum_y"], row[f"{p}__sum_x"], row[f"{p}__sum_y2"], row[f"{p}__sum_x2"], row[f"{p}__sum_xy"])
    return MeanStats(row["n"], row[f"{p}__sum_y"], row[f"{p}__sum_y2"])


def analyze(
    stats: pl.DataFrame,
    counts: pl.DataFrame,
    experiments: pl.DataFrame,
    metrics: Iterable[Metric] = DEFAULT_METRICS,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Turn per-arm sufficient statistics into one result row per (experiment, metric, method)."""
    metrics = tuple(metrics)
    by_key = {(r["experiment_id"], r["variant"]): r for r in stats.iter_rows(named=True)}
    cnt = {(r["experiment_id"], r["variant"]): r for r in counts.iter_rows(named=True)}
    out: list[dict] = []
    for exp in experiments.iter_rows(named=True):
        eid, cv, tv = exp["experiment_id"], exp["control_variant"], exp["treatment_variant"]
        share = exp["treatment_share"]
        cc, ct = cnt.get((eid, cv)), cnt.get((eid, tv))
        srm_assigned = srm_exposed = math.nan
        if cc and ct:
            srm_assigned = srm_check([cc["assigned"], ct["assigned"]], [1 - share, share]).p_value
            srm_exposed = srm_check([cc["exposed"], ct["exposed"]], [1 - share, share]).p_value
        rc, rt = by_key.get((eid, cv)), by_key.get((eid, tv))
        if rc is None or rt is None:
            continue
        for m in metrics:
            results = []
            if isinstance(m, RatioMetric):
                results.append(("delta", delta_method_ratio(_row_stats(rc, m), _row_stats(rt, m), alpha), math.nan))
            else:
                results.append(("welch", welch_from_stats(_row_stats(rc, m), _row_stats(rt, m), alpha), math.nan))
                if m.covariate:
                    r, theta = cuped_from_stats(_row_stats(rc, m, True), _row_stats(rt, m, True), alpha)
                    results.append(("cuped", r, theta))
            for method, r, theta in results:
                out.append(
                    dict(
                        experiment_id=eid,
                        metric=m.name,
                        method=method,
                        n_control=rc["n"],
                        n_treatment=rt["n"],
                        control=r.control_mean,
                        treatment=r.treatment_mean,
                        estimate=r.estimate,
                        se=r.se,
                        p_value=r.p_value,
                        ci_low=r.ci_low,
                        ci_high=r.ci_high,
                        theta=theta,
                        srm_p_assigned=srm_assigned,
                        srm_p_exposed=srm_exposed,
                    )
                )
    return pl.DataFrame(out)
