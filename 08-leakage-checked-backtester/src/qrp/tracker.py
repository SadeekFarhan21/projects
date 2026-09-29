"""Local experiment tracker: one directory per run, plain files only.

runs/<run_id>/
    config.json     the fully resolved config (after CLI overrides)
    meta.json       timestamps, git sha, python/numpy versions, data snapshot id, duration
    metrics.json    flat dict of scalar metrics
    artifacts/      anything else (equity curve parquet, plots, audit reports)

No database and no server: `ls`, `cat` and `jq` work, and runs diff cleanly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _git_sha(cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


class Run:
    def __init__(self, root: str | Path, name: str, config: dict):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self.id = f"{stamp}-{name}-{config_hash(config)}"
        self.dir = Path(root) / self.id
        self.art = self.dir / "artifacts"
        self.art.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.meta = {"run_id": self.id, "name": name, "started": datetime.now().isoformat(),
                     "git_sha": _git_sha(Path.cwd()), "python": platform.python_version(),
                     "numpy": np.__version__, "machine": platform.machine()}
        self._t0 = time.time()
        self._write("config.json", config)

    def _write(self, fname: str, obj) -> None:
        (self.dir / fname).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))

    def log_meta(self, **kw) -> None:
        self.meta.update(kw)

    def log_metrics(self, metrics: dict) -> None:
        self._write("metrics.json", metrics)

    def artifact_path(self, name: str) -> Path:
        return self.art / name

    def finish(self, status: str = "ok") -> None:
        self.meta.update(status=status, seconds=round(time.time() - self._t0, 3),
                         finished=datetime.now().isoformat())
        self._write("meta.json", self.meta)


def load_runs(root: str | Path) -> list[dict]:
    out = []
    for d in sorted(Path(root).glob("*/")):
        if not (d / "config.json").exists():
            continue
        rec = {"id": d.name, "dir": str(d)}
        for f in ("config", "meta", "metrics"):
            p = d / f"{f}.json"
            rec[f] = json.loads(p.read_text()) if p.exists() else {}
        out.append(rec)
    return out


def find_run(root: str | Path, key: str) -> dict:
    runs = load_runs(root)
    hits = [r for r in runs if r["id"] == key] or [r for r in runs if key in r["id"]]
    if len(hits) != 1:
        raise SystemExit(f"run key {key!r} matched {len(hits)} runs")
    return hits[0]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[key] = ",".join(str(x.get("name", x)) for x in v)
        else:
            out[key] = v
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def format_table(rows: list[list[str]]) -> str:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def list_table(runs: list[dict], metrics=("sharpe", "ann_return", "max_drawdown",
                                          "avg_daily_turnover")) -> str:
    rows = [["run_id", *metrics, "status"]]
    for r in runs:
        rows.append([r["id"], *(_fmt(r["metrics"].get(m)) for m in metrics),
                     _fmt(r["meta"].get("status"))])
    return format_table(rows)


def compare_table(runs: list[dict]) -> str:
    """Metrics side by side, then only the config keys that differ between runs."""
    head = ["key", *(r["id"][-24:] for r in runs)]
    mkeys = sorted({k for r in runs for k in r["metrics"]})
    rows = [head] + [[f"metric.{k}", *(_fmt(r["metrics"].get(k)) for r in runs)] for k in mkeys]
    flat = [flatten(r["config"]) for r in runs]
    ckeys = sorted({k for f in flat for k in f})
    diff = [k for k in ckeys if len({json.dumps(f.get(k), default=str) for f in flat}) > 1]
    rows += [[f"config.{k}", *(_fmt(f.get(k)) for f in flat)] for k in diff]
    return format_table(rows)
