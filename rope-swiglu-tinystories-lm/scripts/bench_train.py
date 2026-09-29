"""Measure training-step throughput for precision x attention implementation.

Usage: uv run python scripts/bench_train.py
Writes results/train_throughput.csv. Uses random token ids (throughput does
not depend on the data) with the same shapes as the v0 training run.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from tfs.model import ModelConfig, Transformer
from tfs.train import autocast_ctx, make_optimizer, pick_device, sync


def bench(device, amp, attn_impl, args) -> tuple[float, float]:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=4096, d_model=args.d_model, n_layers=args.n_layers,
                      n_heads=args.n_heads, max_seq_len=args.seq_len, attn_impl=attn_impl)
    model = Transformer(cfg).to(device).train()
    opt = make_optimizer(model, 1e-3, 0.1)
    x = torch.randint(0, 4096, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, 4096, (args.batch_size, args.seq_len), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            loss = model.loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    for _ in range(args.warmup):
        step()
    sync(device)
    if device.type == "mps":
        torch.mps.reset_peak_memory_stats() if hasattr(torch.mps, "reset_peak_memory_stats") else None
    t0 = time.perf_counter()
    for _ in range(args.iters):
        loss = step()
    loss.item()
    sync(device)
    dt = time.perf_counter() - t0
    mem = torch.mps.driver_allocated_memory() / 2**20 if device.type == "mps" else float("nan")
    return args.iters * args.batch_size * args.seq_len / dt, mem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/train_throughput.csv"))
    args = ap.parse_args()
    device = pick_device(args.device)
    rows = []
    for attn in ("manual", "sdpa"):
        for amp in ("none", "bf16", "fp16"):
            try:
                tps, mem = bench(device, amp, attn, args)
                status = "ok"
            except Exception as e:  # record unsupported combos instead of crashing
                tps, mem, status = float("nan"), float("nan"), f"error: {type(e).__name__}: {e}"[:120]
            rows.append([str(device), attn, amp, args.batch_size, args.seq_len, f"{tps:.0f}", f"{mem:.0f}", status])
            print(rows[-1])
    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "attn_impl", "amp", "batch_size", "seq_len", "train_tok_per_s", "driver_mem_mb", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
