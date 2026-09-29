"""Sharded checkpoints with an atomic commit.

Layout:
  ckpt_dir/
    step_000003/
      replicated.pt   state identical on all ranks (written once, by rank 0)
      rank00000.pt    state owned by rank r (TP shard, pipeline stage, ZeRO shard)
      meta.json       world size, mode, step, config; written LAST = commit marker
    latest            text file naming the newest committed step directory

Write protocol: every rank writes its shard to a temp file and renames it (rename is
atomic on POSIX), barrier, then rank 0 writes meta.json and updates `latest`. A crash
before meta.json leaves an uncommitted directory that load() ignores because `latest`
still points at the previous step.

v0 restriction: resume requires the same world size and mode (no resharding).
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(text: str, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def save(ckpt_dir: str, step: int, shard: dict, replicated: dict | None, meta: dict) -> str:
    rank, world = dist.get_rank(), dist.get_world_size()
    d = os.path.join(ckpt_dir, f"step_{step:06d}")
    if rank == 0:
        os.makedirs(d, exist_ok=True)
    dist.barrier()
    _atomic_save(shard, os.path.join(d, f"rank{rank:05d}.pt"))
    if rank == 0 and replicated is not None:
        _atomic_save(replicated, os.path.join(d, "replicated.pt"))
    dist.barrier()
    if rank == 0:
        full_meta = dict(meta, step=step, world_size=world)
        _atomic_write_text(json.dumps(full_meta, indent=2), os.path.join(d, "meta.json"))
        _atomic_write_text(os.path.basename(d), os.path.join(ckpt_dir, "latest"))
    dist.barrier()
    return d


def latest(ckpt_dir: str) -> str | None:
    p = os.path.join(ckpt_dir, "latest")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = os.path.join(ckpt_dir, f.read().strip())
    return d if os.path.exists(os.path.join(d, "meta.json")) else None


def load(ckpt_dir: str, expect_mode: str) -> tuple[dict, dict | None, dict] | None:
    d = latest(ckpt_dir)
    if d is None:
        return None
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    world, rank = dist.get_world_size(), dist.get_rank()
    if meta["world_size"] != world or meta.get("mode") != expect_mode:
        raise ValueError(f"checkpoint is {meta.get('mode')} at world {meta['world_size']}, "
                         f"run is {expect_mode} at world {world}; resharding is not in v0")
    shard = torch.load(os.path.join(d, f"rank{rank:05d}.pt"), weights_only=False)
    rp = os.path.join(d, "replicated.pt")
    replicated = torch.load(rp, weights_only=False) if os.path.exists(rp) else None
    return shard, replicated, meta
