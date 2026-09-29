"""Shared helpers. Worker functions live at module top level so spawn can import them."""

from __future__ import annotations

import dataclasses
import functools

import torch

from dtrain.dist_utils import launch
from dtrain.model import GPTConfig
from dtrain.tensor_parallel import merge_state_dicts
from dtrain.trainer import TrainConfig, train

SMALL = GPTConfig(vocab_size=64, seq_len=16, d_model=32, n_head=4, n_layer=4)


def tc(**kw) -> TrainConfig:
    kw.setdefault("model", SMALL)
    return TrainConfig(**kw)


def run(world: int, cfg: TrainConfig) -> list[dict]:
    return launch(train, world, cfg)


@functools.lru_cache(maxsize=None)
def _reference_cached(key: str) -> dict:
    cfg = TrainConfig(**eval(key, {"GPTConfig": GPTConfig, "TrainConfig": TrainConfig}))
    return run(1, cfg)[0]


def reference(cfg: TrainConfig) -> dict:
    """Single-process run with the same hyperparameters (mode forced to single)."""
    fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    fields.update(mode="single", ckpt_dir=None, ckpt_every=0, resume=False)
    return _reference_cached(repr(fields))


def full_state(mode: str, results: list[dict]) -> dict:
    if mode == "tp":
        return merge_state_dicts([r["state"] for r in results])
    if mode == "pp":
        out = {}
        for r in results:
            out.update(r["state"])
        return out
    return results[0]["state"]


def max_param_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys(), (sorted(set(a) ^ set(b)))
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def assert_states_close(a: dict, b: dict, atol: float) -> None:
    d = max_param_diff(a, b)
    assert d <= atol, f"max param diff {d:.3e} > {atol:.1e}"
