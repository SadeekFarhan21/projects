"""Experiment 01. Train SAEs on streamed GPT-2 small residual activations.

Presets (all 16x expansion = 12,288 latents):
  main      L5 and L8, TopK k=32 and ReLU+L1, plus a second TopK seed at L8
  frontier  L8 only, TopK k=16 and k=64, ReLU+L1 at two more coefficients
  pilot     short L8 run for calibrating L1 coefficients

  uv run python experiments/01_train.py --preset main --minutes 40
  uv run python experiments/01_train.py --preset main --tokens 300000000 --minutes 720   # full run

Checkpoints go to checkpoints/<run>/<sae>/ (gitignored). Training curves and
the run summary are copied to results/01_train_<run>_*.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from saelab.buffer import ActivationBuffer
from saelab.data import iter_batches
from saelab.model import get_device, harvest, hook_name, load_model
from saelab.sae import SAEConfig
from saelab.train import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]

RELU_MAIN = 2.0  # chosen from the pilot run, see DEVLOG


def preset(name: str, relu_coeffs: list[float] | None) -> list[SAEConfig]:
    if name == "main":
        return [
            SAEConfig(arch="topk", k=32, hook=hook_name(5)),
            SAEConfig(arch="relu", l1_coeff=RELU_MAIN, hook=hook_name(5)),
            SAEConfig(arch="topk", k=32, hook=hook_name(8)),
            SAEConfig(arch="relu", l1_coeff=RELU_MAIN, hook=hook_name(8)),
            SAEConfig(arch="topk", k=32, hook=hook_name(8), seed=1),
        ]
    if name == "frontier":
        coeffs = relu_coeffs or [1.0, 4.0]
        return ([SAEConfig(arch="topk", k=k, hook=hook_name(8)) for k in (16, 64)]
                + [SAEConfig(arch="relu", l1_coeff=c, hook=hook_name(8)) for c in coeffs])
    if name == "pilot":
        coeffs = relu_coeffs or [0.5, 2.0, 8.0]
        return ([SAEConfig(arch="topk", k=32, hook=hook_name(8))]
                + [SAEConfig(arch="relu", l1_coeff=c, hook=hook_name(8)) for c in coeffs])
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="main")
    ap.add_argument("--tokens", type=int, default=300_000_000)
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--relu-coeffs", type=float, nargs="*")
    ap.add_argument("--ctx", type=int, default=128)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(0)
    model = load_model(device)
    saes = preset(args.preset, args.relu_coeffs)
    layers = tuple(sorted({int(s.hook.split(".")[1]) for s in saes}))
    cfg = TrainConfig(token_budget=args.tokens, minutes=args.minutes, saes=saes)
    if args.preset == "pilot":
        cfg.warmup_steps = 50
    toks = iter_batches("train", ctx=args.ctx, batch=64)
    buf = ActivationBuffer(toks, lambda t: harvest(model, t, layers), layers,
                           model.cfg.d_model, cfg.buffer_tokens, cfg.batch_size, device)
    out = ROOT / "checkpoints" / args.preset
    summary = train(cfg, buf, out, device)
    summary["layers"] = layers
    summary["ctx"] = args.ctx
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / f"01_train_{args.preset}_summary.json").write_text(json.dumps(summary, indent=1))
    for s in saes:
        shutil.copy(out / s.name() / "train_log.csv", res / f"01_train_{args.preset}_{s.name()}_log.csv")
    print(json.dumps({k: summary[k] for k in ("tokens", "steps", "wall_s", "end_to_end_tokens_per_s")}))


if __name__ == "__main__":
    main()
