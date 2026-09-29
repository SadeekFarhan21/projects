"""Grid-search autotuner with a JSON cache.

For every kernel of a Program the tuner enumerates candidate configs for its
kernel family, compiles each variant, times it in isolation with GPU
timestamps and keeps the fastest. Results are cached by the kernel's tune_key
(family, shape, dtype and operand strides), so a second compile of the same
shapes performs no search.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

from . import _metal
from .codegen import generate, mm_config_ok

DEFAULT_CACHE = Path(os.environ.get("TCM_TUNE_CACHE", Path.cwd() / ".tcm_cache" / "autotune.json"))


def matmul_candidates() -> list[dict]:
    out = []
    for BM, BN, BK, TM, TN in itertools.product((32, 64, 128), (32, 64, 128), (8, 16, 32), (2, 4, 8), (2, 4, 8)):
        c = {"BM": BM, "BN": BN, "BK": BK, "TM": TM, "TN": TN}
        if not mm_config_ok(c):
            continue
        # keep register tiles sane: at most 64 accumulators per thread
        if TM * TN > 64 or TM * TN < 4:
            continue
        out.append(c)
    return out


def row_candidates(n: int, vec: int) -> list[dict]:
    out = []
    for tpr in (32, 64, 128, 256, 512, 1024):
        if tpr > 32 and tpr * vec > 2 * n:
            continue
        rpgs = (1, 2, 4, 8) if tpr == 32 else (1,)
        for rpg in rpgs:
            out.append({"tpr": tpr, "rpg": rpg})
    return out


def ew_candidates() -> list[dict]:
    return [{"tg": t} for t in (64, 128, 256, 512, 1024)]


class Autotuner:
    def __init__(self, cache_path: str | os.PathLike | None = None, budget_s: float = 60.0, verbose: bool = False):
        self.path = Path(cache_path) if cache_path else DEFAULT_CACHE
        self.budget_s = budget_s
        self.verbose = verbose
        self.cache: dict[str, dict] = {}
        self.log: list[dict] = []  # every timed candidate
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.cache = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=1, sort_keys=True))

    def candidates(self, prog, idx: int) -> list[dict]:
        spec, grp = prog.kernels[idx], prog.groups[idx]
        if spec.kind == "matmul":
            return matmul_candidates()
        if spec.kind == "row":
            return row_candidates(grp.domain[-1], spec.config["vec"])
        return ew_candidates()

    def time_spec(self, prog, spec, reps: int | None = None) -> float:
        dev = prog.dev
        p = dev.compile(spec.source, spec.name)
        if spec.threads[0] > p.max_threads:
            return float("inf")
        d = _metal.Dispatch(p, [prog.buffer_of[v] for v in spec.args], spec.groups, spec.threads)
        dev.run([d], repeat=2)
        if reps is None:
            t1 = dev.run([d], repeat=1)["gpu_s"]
            reps = int(min(200, max(3, 0.02 / max(t1, 1e-7))))
        best = float("inf")
        for _ in range(3):
            r = dev.run([d], repeat=reps)
            best = min(best, r["gpu_s"] / reps)
        return best

    def tune_kernel(self, prog, idx: int, force: bool = False) -> dict:
        spec, grp = prog.kernels[idx], prog.groups[idx]
        key = spec.tune_key
        if not force and key in self.cache:
            return self.cache[key]["config"]
        t_start = time.perf_counter()
        best_cfg, best_t = dict(spec.config), self.time_spec(prog, spec)
        default_t = best_t
        for cfg in self.candidates(prog, idx):
            if time.perf_counter() - t_start > self.budget_s:
                break
            try:
                cand = generate(prog.graph, grp, spec.name, cfg)
                t = self.time_spec(prog, cand)
            except (RuntimeError, ValueError) as e:  # compile or resource failure: skip
                if self.verbose:
                    print("skip", cfg, e)
                continue
            self.log.append({"key": key, **{k: v for k, v in cand.config.items()}, "time_s": t})
            if t < best_t:
                best_cfg, best_t = dict(cand.config), t
        self.cache[key] = {"config": best_cfg, "time_s": best_t, "default_time_s": default_t}
        if self.verbose:
            print(f"tuned {key}: {best_cfg} {best_t * 1e6:.1f}us (default {default_t * 1e6:.1f}us)")
        return best_cfg

    def tune_program(self, prog, force: bool = False) -> None:
        for i in range(len(prog.kernels)):
            cfg = self.tune_kernel(prog, i, force=force)
            spec = generate(prog.graph, prog.groups[i], prog.kernels[i].name, cfg)
            prog.kernels[i] = spec
        prog._make_dispatches()
        self.save()
