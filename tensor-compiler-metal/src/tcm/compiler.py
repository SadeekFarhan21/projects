"""Compiler driver and executable Program.

    prog = tcm.compile(graph, fuse=True, autotune=False)
    outs = prog(x, w)            # numpy in, numpy out
    t = prog.time(repeat=50)     # GPU timestamp timing of the whole schedule
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .codegen import KernelSpec, generate
from .ir import NP_DTYPE, Graph
from .passes import optimize
from .schedule import BufferPlan, fuse_groups, legalize, plan_buffers

_DEVICE = None


def device():
    """Process-wide Metal device (lazy so the pure-Python parts import without it)."""
    global _DEVICE
    if _DEVICE is None:
        from . import _metal

        _DEVICE = _metal.Device()
    return _DEVICE


@dataclass
class CompileOptions:
    optimize: bool = True
    fuse: bool = True
    reuse_buffers: bool = True
    autotune: bool = False
    tune_cache: str | None = None
    configs: dict = field(default_factory=dict)  # tune_key -> config override


def lower(g: Graph, opts: CompileOptions) -> tuple[Graph, list, list[KernelSpec], BufferPlan]:
    if opts.optimize:
        g = optimize(g)
    g = legalize(g)
    groups = fuse_groups(g, fuse=opts.fuse)
    kernels = []
    for i, grp in enumerate(groups):
        spec = generate(g, grp, f"k{i}_{grp.kind}")
        if spec.tune_key in opts.configs:
            spec = generate(g, grp, spec.name, opts.configs[spec.tune_key])
        kernels.append(spec)
    plan = plan_buffers(g, groups)
    return g, groups, kernels, plan


class Program:
    def __init__(self, graph: Graph, opts: CompileOptions | None = None) -> None:
        self.opts = opts or CompileOptions()
        self.source_graph = graph
        self.graph, self.groups, self.kernels, self.plan = lower(graph, self.opts)
        self.dev = device()
        self._build()
        if self.opts.autotune:
            from .autotune import Autotuner

            self.tuner = Autotuner(self.opts.tune_cache)
            self.tuner.tune_program(self)

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        g, plan = self.graph, self.plan
        if self.opts.reuse_buffers:
            self.slot_buffers = [self.dev.alloc(nb) for nb in plan.slot_bytes]
            self.buffer_of = {v: self.slot_buffers[s] for v, s in plan.slot_of.items()}
        else:
            self.slot_buffers = []
            self.buffer_of = {v: self.dev.alloc(g[v].nbytes) for v in plan.slot_of}
        # upload constants once
        for v, s in plan.slot_of.items():
            if plan.kind_of_slot[s] == "const":
                arr = np.ascontiguousarray(g[v].attrs["value"], dtype=NP_DTYPE[g[v].dtype])
                self.dev.upload(self.buffer_of[v], arr)
        self._make_dispatches()

    def _make_dispatches(self) -> None:
        from . import _metal

        self.pipelines = [self.dev.compile(k.source, k.name) for k in self.kernels]
        self.dispatches = []
        for k, p in zip(self.kernels, self.pipelines):
            if k.threads[0] > p.max_threads:
                raise RuntimeError(f"{k.name}: {k.threads[0]} threads exceeds pipeline limit {p.max_threads}")
            bufs = [self.buffer_of[v] for v in k.args]
            self.dispatches.append(_metal.Dispatch(p, bufs, k.groups, k.threads))

    def replace_kernel(self, idx: int, spec: KernelSpec) -> None:
        self.kernels[idx] = spec
        self._make_dispatches()

    # ------------------------------------------------------------------ run
    @property
    def num_kernels(self) -> int:
        return len(self.kernels)

    def set_inputs(self, *arrays: np.ndarray) -> None:
        g = self.graph
        if len(arrays) != len(g.inputs):
            raise ValueError(f"expected {len(g.inputs)} inputs, got {len(arrays)}")
        for nid, a in zip(g.inputs, arrays):
            n = g[nid]
            a = np.ascontiguousarray(a, dtype=NP_DTYPE[n.dtype])
            if a.shape != n.shape:
                raise ValueError(f"input {n.attrs['name']}: shape {a.shape} != {n.shape}")
            self.dev.upload(self.buffer_of[nid], a)

    def get_outputs(self) -> list[np.ndarray]:
        outs = []
        for o in self.graph.outputs:
            n = self.graph[o]
            arr = np.empty(n.shape, dtype=NP_DTYPE[n.dtype])
            self.dev.download(self.buffer_of[o], arr)
            outs.append(arr)
        return outs

    def run(self, repeat: int = 1, mode: str = "batched", profile: bool = False) -> dict:
        return self.dev.run(self.dispatches, repeat=repeat, mode=mode, profile=profile)

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        self.set_inputs(*arrays)
        self.run()
        return self.get_outputs()

    def time(self, repeat: int = 20, warmup: int = 3, mode: str = "batched") -> dict:
        """GPU time per execution of the whole schedule, from command buffer timestamps."""
        self.run(repeat=warmup, mode=mode)
        r = self.run(repeat=repeat, mode=mode)
        return {"gpu_s": r["gpu_s"] / repeat, "wall_s": r["wall_s"] / repeat}

    def profile(self, repeat: int = 10) -> list[float]:
        """Per-kernel GPU seconds from stage-boundary timestamp counters."""
        self.run(repeat=2)
        r = self.run(repeat=repeat, profile=True)
        return list(r.get("per_kernel_s", []))

    def summary(self) -> str:
        lines = [f"{len(self.kernels)} kernels"]
        for k in self.kernels:
            ops = ",".join(self.graph[n].op for n in k.nodes)
            lines.append(f"  {k.name}: {k.kind} grid={k.groups} tg={k.threads} ops=[{ops}] cfg={k.config}")
        p = self.plan
        lines.append(f"  temp bytes naive={p.naive_temp_bytes} pooled={p.pooled_temp_bytes}")
        return "\n".join(lines)


def compile(graph: Graph, **kw) -> Program:  # noqa: A001 (mirrors torch.compile naming)
    return Program(graph, CompileOptions(**kw))


def run_graph(graph: Graph, inputs: Sequence[np.ndarray], **kw) -> list[np.ndarray]:
    return compile(graph, **kw)(*inputs)
