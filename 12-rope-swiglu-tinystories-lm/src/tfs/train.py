"""Training loop: AdamW, linear warmup + cosine decay, grad clipping, optional bf16 autocast.

Usage: uv run python -m tfs.train --out-dir runs/v0 --max-minutes 30
Logs one CSV row per eval to <out-dir>/log.csv and saves <out-dir>/ckpt.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import TokenDataset
from .model import ModelConfig, Transformer


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def autocast_ctx(device: torch.device, amp: str):
    """amp: 'none' | 'bf16' | 'fp16'. bf16 needs no loss scaling (same exponent range as fp32)."""
    if amp == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    return torch.autocast(device_type=device.type, dtype=dtype)


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warmup to max_lr, then cosine decay to min_lr at `total`."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    # Decay matrices only. Norm gains are 1-D and decaying them toward 0 just
    # fights the normalization; the tied embedding is 2-D and is decayed.
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def evaluate(model, batches, device, amp) -> float:
    model.eval()
    losses = []
    for x, y in batches:
        with autocast_ctx(device, amp):
            losses.append(model.loss(x.to(device), y.to(device)).item())
    model.train()
    return float(np.mean(losses))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/v0"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--attn-impl", default="manual", choices=["manual", "sdpa"])
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--max-minutes", type=float, default=None, help="hard wall-clock stop")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    train_ds = TokenDataset(args.data_dir / "train.bin")
    val_ds = TokenDataset(args.data_dir / "valid.bin")
    from .bpe import BPETokenizer

    tok = BPETokenizer.load(args.data_dir / "tokenizer.json")

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        attn_impl=args.attn_impl,
    )
    model = Transformer(cfg).to(device)
    opt = make_optimizer(model, args.lr, args.weight_decay)
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(
        f"device={device} amp={args.amp} params={model.num_params():,} "
        f"tokens/step={tokens_per_step:,} train_tokens={len(train_ds):,}"
    )

    # Fixed validation batches so every eval sees the same tokens.
    val_rng = np.random.default_rng(0)
    val_batches = [val_ds.get_batch(args.batch_size, args.seq_len, val_rng) for _ in range(args.eval_batches)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps({"model": cfg.to_dict(), "train": {k: str(v) for k, v in vars(args).items()}}, indent=2)
    )
    log_f = (args.out_dir / "log.csv").open("w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "tokens", "elapsed_s", "lr", "train_loss", "val_loss", "grad_norm", "tok_per_s"])

    model.train()
    t_start = time.perf_counter()
    t_window, tokens_window = t_start, 0
    train_losses: list[float] = []
    step = 0
    stopped_early = False
    for step in range(args.steps + 1):
        # ---- eval / log
        if step % args.eval_every == 0 or step == args.steps:
            sync(device)
            now = time.perf_counter()
            tps = tokens_window / (now - t_window) if tokens_window else float("nan")
            val = evaluate(model, val_batches, device, args.amp)
            tr = float(np.mean(train_losses)) if train_losses else float("nan")
            elapsed = now - t_start
            lr_now = opt.param_groups[0]["lr"]
            gn = grad_norm if step else float("nan")  # noqa: F821 (set in loop body)
            log.writerow([step, step * tokens_per_step, f"{elapsed:.1f}", f"{lr_now:.3e}", f"{tr:.4f}", f"{val:.4f}", f"{gn:.3f}", f"{tps:.0f}"])
            log_f.flush()
            print(f"step {step:5d} | train {tr:.4f} | val {val:.4f} | lr {lr_now:.2e} | {tps:,.0f} tok/s | {elapsed / 60:.1f} min")
            train_losses.clear()
            sync(device)
            t_window, tokens_window = time.perf_counter(), 0  # exclude eval time from tok/s
            if args.max_minutes and elapsed > args.max_minutes * 60 and step < args.steps:
                stopped_early = True
                break
        if step == args.steps:
            break

        # ---- one optimizer step
        lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = train_ds.get_batch(args.batch_size, args.seq_len, rng)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx(device, args.amp):
                loss = model.loss(x, y) / args.grad_accum
            loss.backward()
            loss_acc += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        opt.step()
        train_losses.append(float(loss_acc))
        tokens_window += tokens_per_step

    total_s = time.perf_counter() - t_start
    log_f.close()
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step}, args.out_dir / "ckpt.pt")
    summary = {
        "steps": step,
        "stopped_early_by_time_budget": stopped_early,
        "tokens_seen": step * tokens_per_step,
        "wall_seconds": round(total_s, 1),
        "final_val_loss": val,
        "params": model.num_params(),
        "device": str(device),
        "amp": args.amp,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
