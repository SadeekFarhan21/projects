"""Decode throughput with and without the KV cache.

Two sweeps, batch size 1, greedy decoding, fp32:
  trained : the v0 checkpoint (context 256), 16-token prompt, 32..240 new tokens
  long    : same architecture, random weights, max_seq_len 1024, 128..768 new tokens, to show
            how the gap grows with length (throughput does not depend on weights)
Each point is the median of --repeats runs (--long-repeats for the long sweep). Every run also checks that cached
and uncached decoding produced the same tokens.

Usage: uv run python scripts/bench_generate.py --ckpt runs/v0/ckpt.pt
Writes results/generation_throughput.csv and results/generation_throughput.png.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tfs.generate import load_model
from tfs.model import ModelConfig, Transformer
from tfs.sampling import generate
from tfs.train import sync


def time_generate(model, prompt, n_new, use_cache, repeats):
    device = prompt.device
    generate(model, prompt, min(8, n_new), use_cache=use_cache)  # warm up kernels
    times, out = [], None
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = generate(model, prompt, n_new, use_cache=use_cache)
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/v0/ckpt.pt"))
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--long-repeats", type=int, default=1, help="the long sweep is slow without a cache")
    ap.add_argument("--out", type=Path, default=Path("results/generation_throughput.csv"))
    args = ap.parse_args()

    rows = []
    for dev_name in args.devices.split(","):
        device = torch.device(dev_name)
        sweeps = []
        if args.ckpt.exists():
            sweeps.append(("trained", load_model(args.ckpt, device), [32, 64, 128, 240], args.repeats))
        torch.manual_seed(0)
        long_model = Transformer(ModelConfig(max_seq_len=1024)).to(device).eval()
        sweeps.append(("long", long_model, [128, 384, 768], args.long_repeats))
        for sweep, model, lengths, repeats in sweeps:
            prompt = torch.randint(0, model.cfg.vocab_size, (1, 16), generator=torch.Generator().manual_seed(0)).to(device)
            for n in lengths:
                n = min(n, model.cfg.max_seq_len - 16)
                t_c, out_c = time_generate(model, prompt, n, True, repeats)
                t_u, out_u = time_generate(model, prompt, n, False, repeats)
                same = bool(torch.equal(out_c, out_u))
                row = [dev_name, sweep, 16, n, repeats, f"{n / t_c:.1f}", f"{n / t_u:.1f}", f"{t_u / t_c:.2f}", same]
                rows.append(row)
                print(row, flush=True)

    args.out.parent.mkdir(exist_ok=True)
    header = ["device", "sweep", "prompt_len", "new_tokens", "repeats", "tok_per_s_cache", "tok_per_s_nocache", "speedup", "same_tokens"]
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, sweep in zip(axes, ["trained", "long"]):
        for dev_name, color in zip(args.devices.split(","), ["#1f77b4", "#d62728"]):
            pts = [r for r in rows if r[0] == dev_name and r[1] == sweep]
            if not pts:
                continue
            xs = [r[3] for r in pts]
            ax.plot(xs, [float(r[5]) for r in pts], "-o", color=color, label=f"{dev_name} KV cache")
            ax.plot(xs, [float(r[6]) for r in pts], "--s", color=color, alpha=0.6, label=f"{dev_name} no cache")
        ax.set_title({"trained": "v0 checkpoint (ctx 256)", "long": "same arch, ctx 1024, random weights"}[sweep], fontsize=10)
        ax.set_xlabel("new tokens generated (batch 1, 16-token prompt)")
        ax.set_ylabel("tokens / second")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, f"Source: {args.out}. Measured on a heavily shared machine; see DEVLOG.", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out.with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
