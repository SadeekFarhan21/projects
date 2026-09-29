"""Class conditional CIFAR-10 training with a wall clock budget.

Example:
  uv run python -m diffusion.train --arch unet --objective rf --minutes 8 --out runs/unet_rf
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import DeviceLoader, load_cifar10
from .ema import EMA
from .models import build, n_params
from .objectives import diffusion_loss

ARCH_DEFAULTS = {
    # sized so both backbones have about 4M parameters and similar step cost on the M4 Pro
    "unet": dict(ch=48, ch_mult=(1, 2, 2), blocks=2, attn_res=(8,), dropout=0.1),
    "dit": dict(patch=2, dim=192, depth=6, heads=6),
}


def device_default() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH_DEFAULTS), default="unet")
    ap.add_argument("--objective", choices=["eps", "v", "rf"], default="rf")
    ap.add_argument("--minutes", type=float, default=5.0, help="wall clock budget for optimisation steps")
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p-uncond", type=float, default=0.1, help="label dropout for classifier free guidance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=device_default())
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = args.device
    net = build(args.arch, **ARCH_DEFAULTS[args.arch]).to(dev)
    ema = EMA(net, args.ema)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    x, y = load_cifar10("train")
    loader = DeviceLoader(x, y, args.batch, dev, seed=args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    null = 10

    print(f"{args.arch}/{args.objective}: {n_params(net)/1e6:.2f}M params on {dev}", flush=True)
    log = []
    step = 0
    budget = args.minutes * 60
    t_start = None  # set after the first (warm up / compile) step so it is excluded
    t_log, imgs_log = 0.0, 0
    while step < args.max_steps and (t_start is None or time.perf_counter() - t_start < budget):
        xb, yb = loader.next()
        drop = torch.rand(yb.shape, device=dev, generator=gen) < args.p_uncond
        yb = torch.where(drop, torch.full_like(yb, null), yb)
        loss = diffusion_loss(net, args.objective, xb, yb, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(net)
        step += 1
        if t_start is None:
            sync(dev)
            t_start = t_log = time.perf_counter()
            continue
        imgs_log += args.batch
        if step % args.log_every == 0:
            lv = loss.item()  # forces a device sync so the timing is honest
            now = time.perf_counter()
            log.append({"step": step, "loss": lv, "imgs_per_s": imgs_log / (now - t_log),
                        "train_seconds": now - t_start})
            t_log, imgs_log = now, 0
            print(f"step {step} loss {lv:.4f} {log[-1]['imgs_per_s']:.0f} img/s t={now - t_start:.0f}s", flush=True)
    sync(dev)
    t_total = time.perf_counter() - t_start
    summary = {
        "arch": args.arch, "objective": args.objective, "params": n_params(net), "steps": step,
        "batch": args.batch, "train_seconds": t_total, "imgs_per_s": (step - 1) * args.batch / t_total,
        "final_loss_avg": sum(r["loss"] for r in log[-10:]) / max(1, len(log[-10:])),
        "args": vars(args), "device": dev,
    }
    torch.save({"ema": ema.model.state_dict(), "net": net.state_dict(), "arch": args.arch,
                "arch_kwargs": ARCH_DEFAULTS[args.arch], "objective": args.objective, "summary": summary},
               out / "ckpt.pt")
    (out / "log.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1))
    print(json.dumps(summary), flush=True)
    return summary


def load_ckpt(path: str | Path, device: str, use_ema: bool = True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build(ck["arch"], **ck["arch_kwargs"])
    net.load_state_dict(ck["ema" if use_ema else "net"])
    return net.to(device).eval(), ck


if __name__ == "__main__":
    main()
