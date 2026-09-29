"""Train many SAEs at once from one streamed activation buffer.

GPT-2 is the expensive part of each step on a shared Mac, so every SAE in a
run consumes the same shuffled batches: one GPT-2 forward feeds all of them.
Each SAE has its own Adam optimizer and its own hook layer.

Schedule: linear LR warmup, then constant, then linear decay to zero over the
last 20% of progress. Progress is max(tokens / token_budget, time / time_budget),
so a run that is slowed down by other jobs on the machine still decays its
learning rate properly before the wall-clock limit. ReLU SAEs also warm up
their L1 coefficient over the first 5% of progress.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .buffer import ActivationBuffer
from .sae import SAE, SAEConfig, init_from_sample


@dataclass
class TrainConfig:
    token_budget: int = 20_000_000
    minutes: float = 40.0
    batch_size: int = 4096
    lr: float = 4e-4
    warmup_steps: int = 200
    decay_frac: float = 0.2
    l1_warmup_frac: float = 0.05
    buffer_tokens: int = 2 ** 18
    log_every: int = 50
    saes: list[SAEConfig] = field(default_factory=list)


def lr_factor(progress: float, step: int, cfg: TrainConfig) -> float:
    warm = min(1.0, (step + 1) / cfg.warmup_steps)
    decay_start = 1.0 - cfg.decay_frac
    if progress <= decay_start:
        return warm
    return warm * max(0.0, (1.0 - progress) / cfg.decay_frac)


def train(cfg: TrainConfig, buffer: ActivationBuffer, out_dir: Path, device: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = buffer.store  # the initial fill, already shuffled
    saes, opts = [], []
    for sc in cfg.saes:
        sae = SAE(sc).to(device)
        layer = int(sc.hook.split(".")[1])
        sample = first[layer][: 2 ** 15].float()
        init_from_sample(sae, sample)
        saes.append(sae)
        opts.append(torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999)))

    logs = {sc.name(): [] for sc in cfg.saes}
    t_start = time.time()
    step, tokens = 0, 0
    while True:
        progress = max(tokens / cfg.token_budget, (time.time() - t_start) / (60 * cfg.minutes))
        if progress >= 1.0:
            break
        batch = buffer.next_batch()
        if batch is None:
            break
        f_lr = lr_factor(progress, step, cfg)
        log_now = step % cfg.log_every == 0
        for sae, opt in zip(saes, opts):
            sc = sae.cfg
            name = sc.name()
            layer = int(sc.hook.split(".")[1])
            for g in opt.param_groups:
                g["lr"] = cfg.lr * f_lr
            l1_scale = min(1.0, progress / cfg.l1_warmup_frac) if cfg.l1_warmup_frac > 0 else 1.0
            out = sae.loss(batch[layer])
            loss = out["mse"] + (sc.l1_coeff * l1_scale * out["l1"] if sc.arch == "relu"
                                 else sc.aux_coeff * out["aux"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sae.remove_parallel_grad()
            opt.step()
            sae.normalize_decoder()
            if log_now:
                row = {"step": step, "tokens": tokens + cfg.batch_size,
                       "elapsed_s": round(time.time() - t_start, 1), "lr": cfg.lr * f_lr,
                       "mse": float(out["mse"]), "fvu": float(out["fvu"]), "l0": float(out["l0"]),
                       "aux_or_l1": float(out.get("aux", out.get("l1", 0.0))),
                       "dead_frac": float((sae.since_fired >= sc.dead_tokens).float().mean())}
                logs[name].append(row)
        step += 1
        tokens += cfg.batch_size
        if log_now:
            el = time.time() - t_start
            print(f"step {step} tokens {tokens/1e6:.2f}M elapsed {el/60:.1f}m "
                  f"tok/s {tokens/el:.0f} progress {progress:.3f} | " + " | ".join(
                      f"{s.cfg.name()} fvu {logs[s.cfg.name()][-1]['fvu']:.3f} "
                      f"l0 {logs[s.cfg.name()][-1]['l0']:.0f} dead {logs[s.cfg.name()][-1]['dead_frac']:.3f}"
                      for s in saes), flush=True)

    if device == "mps":
        torch.mps.synchronize()
    wall = time.time() - t_start
    for sae in saes:
        name = sae.cfg.name()
        sae.save(out_dir / name)
        with open(out_dir / name / "train_log.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(logs[name][0].keys()))
            w.writeheader()
            w.writerows(logs[name])
    summary = {
        "tokens": tokens, "steps": step, "wall_s": wall,
        "end_to_end_tokens_per_s": tokens / wall,
        "gpt2_tokens_harvested": buffer.tokens_seen,
        "n_saes": len(saes),
        "train_cfg": {k: v for k, v in asdict(cfg).items() if k != "saes"},
        "saes": [asdict(s) for s in cfg.saes],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
