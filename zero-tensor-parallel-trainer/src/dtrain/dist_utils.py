"""Process-group setup and a tiny multi-process launcher.

On this Mac there is no NVIDIA GPU, so everything runs on CPU processes talking over
gloo. The same code path picks NCCL and one CUDA device per rank when CUDA exists, so
the modules above this file never branch on the backend.
"""

from __future__ import annotations

import os
import socket
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass
class DistEnv:
    rank: int
    world_size: int
    device: torch.device
    backend: str


def pick_backend() -> tuple[str, bool]:
    """nccl + cuda when available, otherwise gloo on CPU."""
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl", True
    return "gloo", False


def init_distributed(rank: int, world_size: int, port: int, backend: str | None = None) -> DistEnv:
    auto_backend, use_cuda = pick_backend()
    backend = backend or auto_backend
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    if use_cuda and backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return DistEnv(rank, world_size, device, backend)


def init_from_env() -> DistEnv:
    """For torchrun-style launches (RANK / WORLD_SIZE / MASTER_PORT set by the launcher)."""
    return init_distributed(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                            int(os.environ.get("MASTER_PORT", "29500")))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, fn: Callable, args: tuple,
            out_dir: str, threads: int) -> None:
    torch.set_num_threads(threads)
    env = init_distributed(rank, world_size, port)
    try:
        result = fn(env, *args)
        torch.save({"ok": True, "result": result}, os.path.join(out_dir, f"rank{rank}.pt"))
    except Exception:
        torch.save({"ok": False, "error": traceback.format_exc()},
                   os.path.join(out_dir, f"rank{rank}.pt"))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(fn: Callable[..., Any], world_size: int, *args: Any, threads: int = 1) -> list[Any]:
    """Run fn(env, *args) on world_size spawned processes and return each rank's result.

    fn must be importable (defined at module top level) because we use the spawn start
    method. Results travel back through torch.save files, which keeps tensors intact and
    avoids multiprocessing queue pickling surprises.
    """
    port = free_port()
    with tempfile.TemporaryDirectory() as out_dir:
        mp.spawn(_worker, args=(world_size, port, fn, args, out_dir, threads),
                 nprocs=world_size, join=True)
        results = []
        for r in range(world_size):
            payload = torch.load(os.path.join(out_dir, f"rank{r}.pt"), weights_only=False)
            if not payload["ok"]:
                raise RuntimeError(f"rank {r} failed:\n{payload['error']}")
            results.append(payload["result"])
        return results
