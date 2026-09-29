"""Metrics over JSONL traces.

* pass@1 is the mean over tasks of c_i / n_i.
* pass@k uses the unbiased estimator from Chen et al. 2021,
  1 - C(n - c, k) / C(n, k), averaged over tasks.
* 95 percent intervals come from a percentile bootstrap that resamples tasks
  (not episodes), because samples of one task are correlated.
* Cost per solved task is total tokens (or wall seconds) spent on all episodes
  in the group divided by the number of solved episodes, so failures count.
* The failure taxonomy is rule based on the trace alone, one primary label
  per failed episode, first matching rule wins.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError("k > n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load_traces(paths: Iterable[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


# ---- failure taxonomy ---------------------------------------------------------

EDIT_TOOLS = {"write_file", "edit_file"}


def classify_failure(tr: dict) -> str | None:
    if tr.get("passed"):
        return None
    stop = tr.get("stop_reason")
    steps = tr.get("steps", [])
    v = tr.get("verifier", {}) or {}
    vmeta = v.get("meta", {}) or {}
    if stop == "harness_error":
        return "harness_error"
    if stop == "parse_errors":
        return "format_errors"
    if stop in ("token_budget", "context_overflow", "episode_timeout"):
        return stop
    tool_steps = [s for s in steps if "tool" in s]
    n_tool_err = sum(1 for s in tool_steps if not s.get("tool_ok", True))
    if not tr.get("submitted"):
        if tool_steps and n_tool_err >= max(2, len(tool_steps) // 2):
            return "stuck_on_tool_errors"
        if len(tool_steps) >= 3 and len({json.dumps([s["tool"], s.get("args")], sort_keys=True)
                                        for s in tool_steps}) <= len(tool_steps) // 2:
            return "looping_same_call"
        return "no_submit_step_limit"
    if tr.get("family") == "sql":
        ok_sql = [s for s in tool_steps if s.get("tool") == "run_sql" and s.get("tool_ok")]
        if vmeta.get("reason") == "unparseable" or tr.get("submission") in (None, ""):
            return "empty_or_unparseable_answer"
        if not ok_sql:
            return "answered_without_query"
        return "wrong_answer"
    # bugfix and other command-verified families
    edits = [s for s in tool_steps if s.get("tool") in EDIT_TOOLS and s.get("tool_ok")]
    if not edits:
        return "submitted_without_edit"
    if vmeta.get("timed_out"):
        return "verifier_timeout"
    detail = v.get("detail", "")
    if vmeta.get("errors") or "SyntaxError" in detail or "ImportError" in detail \
            or "IndentationError" in detail:
        return "broke_import_or_syntax"
    return "tests_still_fail"


# ---- tables ------------------------------------------------------------------------

def episodes_frame(traces: list[dict]) -> pd.DataFrame:
    rows = []
    for tr in traces:
        rows.append({
            "task_id": tr["task_id"], "family": tr["family"], "difficulty": tr["difficulty"],
            "sample": tr.get("sample", 0), "seed": tr.get("seed"), "adapter": tr.get("adapter"),
            "passed": bool(tr["passed"]), "submitted": bool(tr.get("submitted")),
            "stop_reason": tr.get("stop_reason"), "n_steps": tr.get("n_steps", 0),
            "prompt_tokens": tr.get("prompt_tokens", 0), "completion_tokens": tr.get("completion_tokens", 0),
            "total_tokens": tr.get("total_tokens", 0), "wall_s": tr.get("wall_s", 0.0),
            "model_latency_s": tr.get("model_latency_s", 0.0),
            "failure": classify_failure(tr),
        })
    return pd.DataFrame(rows)


def _task_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("task_id").agg(n=("passed", "size"), c=("passed", "sum")).reset_index()


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = stat(values[idx], axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def summarize(df: pd.DataFrame, ks: Iterable[int] = (1,), group_cols=("family", "difficulty"),
              n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per group (plus an 'all' row per family and overall)."""
    groups: list[tuple[dict, pd.DataFrame]] = [({"family": "all", "difficulty": "all"}, df)]
    for fam, g in df.groupby("family"):
        groups.append(({"family": fam, "difficulty": "all"}, g))
        if "difficulty" in group_cols:
            for diff, gg in g.groupby("difficulty"):
                groups.append(({"family": fam, "difficulty": diff}, gg))
    rows = []
    for key, g in groups:
        tc = _task_counts(g)
        n_min = int(tc["n"].min())
        row = dict(key)
        row.update(n_tasks=len(tc), samples_per_task=n_min, episodes=len(g))
        for k in ks:
            if k > n_min:
                continue
            per_task = np.array([pass_at_k(int(n), int(c), k) for n, c in zip(tc["n"], tc["c"])])
            lo, hi = bootstrap_ci(per_task, n_boot=n_boot, seed=seed)
            row[f"pass@{k}"] = float(per_task.mean())
            row[f"pass@{k}_lo"] = lo
            row[f"pass@{k}_hi"] = hi
        solved = int(g["passed"].sum())
        row["solved_episodes"] = solved
        row["tokens_per_solved"] = float(g["total_tokens"].sum() / solved) if solved else float("nan")
        row["wall_s_per_solved"] = float(g["wall_s"].sum() / solved) if solved else float("nan")
        row["mean_tokens_per_episode"] = float(g["total_tokens"].mean())
        row["mean_wall_s_per_episode"] = float(g["wall_s"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def failure_table(df: pd.DataFrame) -> pd.DataFrame:
    f = df[~df["passed"]]
    t = f.groupby(["family", "failure"]).size().rename("count").reset_index()
    t["share_of_family_failures"] = t["count"] / t.groupby("family")["count"].transform("sum")
    return t.sort_values(["family", "count"], ascending=[True, False])
