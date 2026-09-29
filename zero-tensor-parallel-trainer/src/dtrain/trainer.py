"""One training loop for every parallel mode. Tests, benchmarks and the CLI all call
train(env, TrainConfig) through dist_utils.launch.

Modes
  single    one process, the reference
  dp_naive  data parallel, one blocking all_reduce per parameter after backward
  dp        data parallel, bucketed async all_reduce overlapped with backward
  zero1     data parallel with ZeRO-1 sharded AdamW (reduce_scatter + all_gather)
  tp        Megatron tensor parallel over the whole world (every rank sees the full batch)
  pp        GPipe pipeline over the whole world (one stage per rank)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch
import torch.distributed as dist

from . import checkpoint, comm
from .data import SyntheticMarkov, CharText, shakespeare_path, shard_batch
from .data_parallel import BucketedDDP, allreduce_grads_naive
from .dist_utils import DistEnv
from .model import GPTConfig, build_reference
from .pipeline import build_stage, gpipe_step
from .precision import DynamicLossScaler, autocast_ctx, DTYPES
from .tensor_parallel import build_tp_model
from .zero import ZeroAdamW

MODES = ("single", "dp_naive", "dp", "zero1", "tp", "pp")


@dataclass
class TrainConfig:
    mode: str = "single"
    model: GPTConfig = field(default_factory=GPTConfig)
    global_batch: int = 8
    steps: int = 5
    lr: float = 3e-3
    weight_decay: float = 0.01
    precision: str = "fp32"
    bucket_cap_mb: float = 1.0
    overlap: bool = True
    comm_dtype: str | None = None      # "bf16" to compress DP gradients on the wire
    n_micro: int = 4
    grad_clip: float | None = None
    seed: int = 0
    data: str = "synthetic"
    ckpt_dir: str | None = None
    ckpt_every: int = 0
    resume: bool = False
    scaler_init: float = 2.0 ** 16
    scaler_growth_interval: int = 200
    inject_inf_at: int | None = None   # test hook: poison one gradient at this step


def make_data(tc: TrainConfig):
    if tc.data == "synthetic":
        return SyntheticMarkov(tc.model.vocab_size, tc.model.seq_len)
    if tc.data == "shakespeare":
        return CharText(shakespeare_path(), tc.model.seq_len)
    raise ValueError(tc.data)


def _metric_mean(x: float, device) -> float:
    """Average a scalar over ranks without touching the comm counters."""
    t = torch.tensor([x], device=device)
    dist.all_reduce(t)
    return t.item() / dist.get_world_size()


def train(env: DistEnv, tc: TrainConfig) -> dict:
    assert tc.mode in MODES, tc.mode
    rank, world, dev = env.rank, env.world_size, env.device
    if tc.mode == "single":
        assert world == 1, "single mode is the 1-process reference"
    if tc.mode in ("tp", "pp"):
        assert tc.grad_clip is None, "grad clipping across TP/PP shards is not in v0"
    data = make_data(tc)
    ref = build_reference(tc.model, tc.seed).to(dev)
    amp = autocast_ctx(tc.precision, dev)
    scaler = DynamicLossScaler.for_precision(tc.precision, init_scale=tc.scaler_init,
                                             growth_interval=tc.scaler_growth_interval)

    # ---- build the model/optimizer for this mode ------------------------------------
    ddp = zopt = opt = None
    if tc.mode in ("single", "dp_naive", "dp", "zero1"):
        model = ref
    elif tc.mode == "tp":
        model = build_tp_model(ref.state_dict(), tc.model).to(dev)
    else:  # pp
        model = build_stage(ref.state_dict(), tc.model, rank, world).to(dev)
    params = list(model.parameters())

    if tc.mode == "dp":
        cd = DTYPES[tc.comm_dtype] if tc.comm_dtype else None
        ddp = BucketedDDP(model, bucket_cap_mb=tc.bucket_cap_mb, overlap=tc.overlap, comm_dtype=cd)
    if tc.mode == "zero1":
        zopt = ZeroAdamW(params, lr=tc.lr, weight_decay=tc.weight_decay, grad_clip=tc.grad_clip)
    else:
        opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)

    # ---- resume -------------------------------------------------------------------------
    start = 0
    if tc.resume and tc.ckpt_dir:
        loaded = checkpoint.load(tc.ckpt_dir, tc.mode)
        if loaded is not None:
            shard, replicated, meta = loaded
            if tc.mode in ("single", "dp_naive", "dp"):
                model.load_state_dict(replicated["model"])
                opt.load_state_dict(replicated["opt"])
            elif tc.mode == "zero1":
                model.load_state_dict(replicated["model"])
                zopt.load_shard_state_dict(shard["zero"])
            else:
                model.load_state_dict(shard["model"])
                opt.load_state_dict(shard["opt"])
            scaler.load_state_dict(meta["scaler"])
            start = meta["step"]

    losses, step_times, wire, exposed, grad_norms = [], [], [], [], []
    for step in range(start, tc.steps):
        comm.STATS.reset()
        if ddp is not None:
            ddp.exposed_comm_s = 0.0
        t0 = time.perf_counter()
        x, y = data.batch(step, tc.global_batch)
        x, y = x.to(dev), y.to(dev)
        if zopt:
            zopt.zero_grad()
        else:
            opt.zero_grad(set_to_none=True)

        if tc.mode == "pp":
            loss_val = gpipe_step(model, x, y, tc.n_micro, amp, loss_scale=scaler.scale)
        else:
            if tc.mode in ("dp_naive", "dp", "zero1"):
                x, y = shard_batch(x, rank, world), shard_batch(y, rank, world)
            with amp:
                _, loss = model(x, y)
            scaler.scale_loss(loss).backward()
            loss_val = loss.item()

        if tc.inject_inf_at == step and rank == world - 1:
            params[-1].grad.view(-1)[0] = float("inf")

        # ---- gradient sync + optimizer --------------------------------------------------
        if tc.mode == "zero1":
            found_inf = zopt.step(loss_scale=scaler.scale)
            if zopt.last_grad_norm is not None:
                grad_norms.append(zopt.last_grad_norm)
        else:
            if tc.mode == "dp":
                ddp.finish_grad_sync()
            elif tc.mode == "dp_naive":
                allreduce_grads_naive(model)
            found_inf = scaler.unscale_and_check(params)
            if not found_inf:
                if tc.grad_clip is not None:
                    grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)))
                opt.step()
        scaler.update(found_inf)
        step_times.append(time.perf_counter() - t0)
        wire.append(comm.STATS.as_dict())
        if ddp is not None:
            exposed.append(ddp.exposed_comm_s)

        if tc.mode in ("dp_naive", "dp", "zero1"):
            loss_val = _metric_mean(loss_val, dev)
        losses.append(loss_val)

        if tc.ckpt_dir and tc.ckpt_every and (step + 1) % tc.ckpt_every == 0:
            _save(tc, step + 1, model, opt, zopt, scaler)

    return {
        "losses": losses,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "step_times": step_times,
        "comm": wire,
        "exposed_comm": exposed,
        "grad_norms": grad_norms,
        "scaler": scaler.state_dict(),
        "start_step": start,
        "n_params_local": sum(p.numel() for p in params),
        "optim_state_bytes": (zopt.state_bytes_per_rank() if zopt else
                              3 * 4 * sum(p.numel() for p in params)),
        "buckets": len(ddp.buckets) if ddp else None,
    }


def _save(tc, step, model, opt, zopt, scaler) -> None:
    meta = {"mode": tc.mode, "scaler": scaler.state_dict(), "model_cfg": tc.model.to_dict()}
    if tc.mode in ("single", "dp_naive", "dp"):
        checkpoint.save(tc.ckpt_dir, step, shard={},
                        replicated={"model": model.state_dict(), "opt": opt.state_dict()}, meta=meta)
    elif tc.mode == "zero1":
        checkpoint.save(tc.ckpt_dir, step, shard={"zero": zopt.shard_state_dict()},
                        replicated={"model": model.state_dict()}, meta=meta)
    else:
        checkpoint.save(tc.ckpt_dir, step, shard={"model": model.state_dict(), "opt": opt.state_dict()},
                        replicated=None, meta=meta)
