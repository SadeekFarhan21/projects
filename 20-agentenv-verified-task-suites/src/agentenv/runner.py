"""Parallel episode runner with fixed seeds and JSONL traces.

Each (task, sample) job gets a seed derived from (base_seed, task_id, sample)
with SHA-256, so the seed for a job does not depend on worker count,
scheduling order or which other tasks are in the run. Workers are processes;
each builds its own adapter once (for MLX that means one model copy per
worker). The parent writes one JSON line per finished episode.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .adapters import make_adapter
from .agent import EpisodeConfig, run_episode
from .task import Task

_ADAPTER = None


def episode_seed(base_seed: int, task_id: str, sample: int) -> int:
    h = hashlib.sha256(f"{base_seed}:{task_id}:{sample}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _init_worker(spec: str) -> None:
    global _ADAPTER
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _ADAPTER = make_adapter(spec)


def _run_job(task_dict: dict[str, Any], sample: int, seed: int, cfg: EpisodeConfig) -> dict[str, Any]:
    task = Task.from_dict(task_dict)
    return run_episode(task, _ADAPTER, seed, cfg, sample=sample)


def run_eval(tasks: Iterable[Task], adapter_spec: str, out_path: str | Path, n_samples: int = 1,
             base_seed: int = 0, workers: int = 4, cfg: EpisodeConfig = EpisodeConfig(),
             progress: bool = True) -> dict[str, Any]:
    tasks = list(tasks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(t, s, episode_seed(base_seed, t.id, s)) for t in tasks for s in range(n_samples)]
    t0 = time.monotonic()
    n_done = n_pass = 0
    with open(out_path, "w") as f, ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(adapter_spec,)
    ) as pool:
        futs = [pool.submit(_run_job, t.to_dict(), s, seed, cfg) for t, s, seed in jobs]
        for fut in as_completed(futs):
            tr = fut.result()
            f.write(json.dumps(tr, default=str) + "\n")
            f.flush()
            n_done += 1
            n_pass += int(tr["passed"])
            if progress and (n_done % 10 == 0 or n_done == len(jobs)):
                el = time.monotonic() - t0
                print(f"[{adapter_spec}] {n_done}/{len(jobs)} done, {n_pass} passed, "
                      f"{el:.0f}s elapsed, {3600 * n_done / el:.0f} episodes/h", flush=True)
    wall = time.monotonic() - t0
    meta = {
        "adapter": adapter_spec, "n_tasks": len(tasks), "n_samples": n_samples,
        "episodes": len(jobs), "base_seed": base_seed, "workers": workers,
        "config": {**asdict(cfg), "sandbox": asdict(cfg.sandbox)},
        "wall_s": round(wall, 2), "episodes_per_hour": round(3600 * len(jobs) / wall, 1),
        "passed": n_pass, "traces": str(out_path),
    }
    Path(str(out_path).replace(".jsonl", ".meta.json")).write_text(json.dumps(meta, indent=2))
    return meta
