"""Declarative feature specs, a small DAG engine, and the empirical leakage audit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qrp.features.ops import ALL_OPS
from qrp.panel import Panel


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    op: str
    inputs: tuple[str, ...] = ()
    params: dict = field(default_factory=dict, hash=False, compare=True)

    @staticmethod
    def from_dict(d: dict) -> "FeatureSpec":
        d = dict(d)
        name, op = d.pop("name"), d.pop("op")
        if op not in ALL_OPS:
            raise ValueError(f"unknown op {op!r} for feature {name!r}")
        inputs = d.pop("inputs", None)
        if inputs is None:
            inputs = ALL_OPS[op][1]
            if inputs is None:
                raise ValueError(f"op {op!r} needs explicit inputs (feature {name!r})")
        return FeatureSpec(name, op, tuple(inputs), d)


def _toposort(specs: list[FeatureSpec], fields: set[str]) -> list[FeatureSpec]:
    by_name = {s.name: s for s in specs}
    if len(by_name) != len(specs):
        raise ValueError("duplicate feature names")
    order, state = [], {}

    def visit(s: FeatureSpec):
        if state.get(s.name) == 1:
            raise ValueError(f"feature cycle through {s.name!r}")
        if state.get(s.name) == 2:
            return
        state[s.name] = 1
        for i in s.inputs:
            if i in by_name:
                visit(by_name[i])
            elif i not in fields:
                raise ValueError(f"feature {s.name!r} input {i!r} is not a field or feature")
        state[s.name] = 2
        order.append(s)

    for s in specs:
        visit(s)
    return order


def compute_features(panel: Panel, specs: list[FeatureSpec]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in _toposort(specs, set(panel.fields)):
            fn = ALL_OPS[s.op][0]
            args = [out[i] if i in out else panel[i] for i in s.inputs]
            out[s.name] = fn(*args, **s.params)
    return out


# ---------------------------------------------------------------- leakage audit
def _same(a: np.ndarray, b: np.ndarray, rtol=1e-9, atol=1e-12) -> np.ndarray:
    """Elementwise equality that treats NaN == NaN and inf == inf as equal."""
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        close = np.abs(a - b) <= atol + rtol * np.abs(b)
    return both_nan | close | (a == b)


@dataclass
class LeakReport:
    feature: str
    causal: bool
    first_bad_row: int | None   # earliest row t whose value changed when only data after the cut changed
    cut: int | None
    method: str | None


def audit_causality(panel: Panel, specs: list[FeatureSpec], n_cuts: int = 4,
                    seed: int = 0) -> list[LeakReport]:
    """Empirically test that every feature at row t depends only on rows <= t.

    Two probes per cut point c:
      truncation    recompute on panel[:c+1]; rows <= c must be unchanged.
                    Catches full-sample statistics (global z-scores, demeaning).
      perturbation  scramble every field after c (random positive multiplicative noise);
                    rows <= c must be unchanged. Catches shift(-k) and centered windows,
                    and would still catch them if a length check hid the truncation probe.
    """
    rng = np.random.default_rng(seed)
    T = panel.shape[0]
    base = compute_features(panel, specs)
    cuts = sorted(set(np.linspace(T // 4, T - 2, n_cuts).astype(int)))
    worst: dict[str, LeakReport] = {s.name: LeakReport(s.name, True, None, None, None) for s in specs}

    def record(name, rows_ok, c, method):
        bad = np.where(~rows_ok)[0]
        if len(bad):
            r = worst[name]
            if r.first_bad_row is None or bad[0] < r.first_bad_row:
                worst[name] = LeakReport(name, False, int(bad[0]), c, method)

    for c in cuts:
        trunc = compute_features(panel.slice_time(c + 1), specs)
        pert_panel = panel.copy()
        for k, v in pert_panel.fields.items():
            v[c + 1:] *= np.exp(rng.normal(0, 0.5, v[c + 1:].shape))
        pert = compute_features(pert_panel, specs)
        for s in specs:
            record(s.name, _same(trunc[s.name], base[s.name][: c + 1]).all(axis=1), c, "truncation")
            record(s.name, _same(pert[s.name][: c + 1], base[s.name][: c + 1]).all(axis=1), c,
                   "perturbation")
    return [worst[s.name] for s in specs]


class LookAheadError(RuntimeError):
    pass


def assert_causal(panel: Panel, specs: list[FeatureSpec], **kw) -> None:
    bad = [r for r in audit_causality(panel, specs, **kw) if not r.causal]
    if bad:
        msg = "; ".join(f"{r.feature} (row {r.first_bad_row} changed after cut {r.cut}, {r.method})"
                        for r in bad)
        raise LookAheadError(f"look-ahead detected in features: {msg}")
