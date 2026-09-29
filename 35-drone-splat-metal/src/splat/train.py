"""Train a Gaussian splat from COLMAP poses (no densification, degree-0 colour).

Example:
  uv run python -m splat.train --scene data/garden --out runs/garden --minutes 35
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import load_scene
from .export import save_checkpoint, write_cameras_json, write_splat
from .metrics import psnr, ssim
from .model import Gaussians, render
from .rasterize import TILE


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def random_rect(W: int, H: int, crop: int, rng: random.Random):
    """Tile-aligned crop. crop <= 0 means the full image."""
    if crop <= 0 or (crop >= W and crop >= H):
        return (0, 0, W, H)
    ntx, nty = math.ceil(W / TILE), math.ceil(H / TILE)
    ct = crop // TILE
    tx = rng.randint(0, max(0, ntx - ct))
    ty = rng.randint(0, max(0, nty - ct))
    x0, y0 = tx * TILE, ty * TILE
    return (x0, y0, min(W, x0 + crop), min(H, y0 + crop))


@torch.no_grad()
def evaluate(g, views, device, bg, save_dir: Path | None = None, limit: int | None = None):
    rows = []
    for v in views[:limit]:
        t0 = time.time()
        img = render(g, v.cam, background=bg, chunk=256, tile_batch=1024).clamp(0, 1)
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        rows.append({"view": v.name, "psnr": psnr(img, v.image), "ssim": ssim(img, v.image).item(), "render_s": dt})
        if save_dir is not None:
            from PIL import Image
            save_dir.mkdir(parents=True, exist_ok=True)
            both = torch.cat([v.image, img], 1).cpu().numpy()
            Image.fromarray((both * 255).astype(np.uint8)).save(save_dir / f"{Path(v.name).stem}.jpg", quality=90)
    return rows


def summarize(rows):
    return {
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "render_s": float(np.mean([r["render_s"] for r in rows])),
        "views": len(rows),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--minutes", type=float, default=35.0, help="wall clock training budget")
    ap.add_argument("--crop", type=int, default=512, help="training crop side in pixels, 0 = full image")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=0, help="steps between quick evals on 4 test views")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiplies every learning rate")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    scene = load_scene(args.scene, args.downscale, device=dev, max_images=args.max_images)
    g = Gaussians.from_points(scene.xyz, scene.rgb).to(dev)
    print(f"loaded {len(scene.train)} train / {len(scene.test)} test views, {g.n} gaussians, "
          f"extent {scene.extent:.3f}, {time.time() - t_load:.1f}s", flush=True)
    bg = torch.zeros(3, device=dev)

    # learning rates from the 3DGS reference implementation
    s = args.lr_scale
    pos_lr0, pos_lr1 = 1.6e-4 * scene.extent * s, 1.6e-6 * scene.extent * s
    opt = torch.optim.Adam(
        [
            {"params": [g.means], "lr": pos_lr0, "name": "means"},
            {"params": [g.log_scales], "lr": 5e-3 * s, "name": "scales"},
            {"params": [g.quats], "lr": 1e-3 * s, "name": "quats"},
            {"params": [g.logit_opa], "lr": 5e-2 * s, "name": "opacity"},
            {"params": [g.sh0], "lr": 2.5e-3 * s, "name": "sh0"},
        ],
        eps=1e-15,
    )

    t0 = time.time()
    init = evaluate(g, scene.test, dev, bg)
    init_s = summarize(init)
    print(f"init eval psnr {init_s['psnr']:.3f} ssim {init_s['ssim']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    log = open(out / "train_log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["step", "elapsed_s", "loss", "l1", "psnr_crop", "pixels", "pairs"])
    quick_rows = []
    budget = args.minutes * 60
    order: list[int] = []
    t_train = time.time()
    step = 0
    pixels_seen = 0
    while step < args.max_steps and time.time() - t_train < budget:
        if not order:
            order = list(range(len(scene.train)))
            rng.shuffle(order)
        v = scene.train[order.pop()]
        # position lr: log-linear decay over the planned run (by time when time-bound)
        frac = min(1.0, max(step / args.max_steps, (time.time() - t_train) / budget))
        opt.param_groups[0]["lr"] = math.exp(math.log(pos_lr0) * (1 - frac) + math.log(pos_lr1) * frac)

        rect = random_rect(v.cam.width, v.cam.height, args.crop, rng)
        img, st = render(g, v.cam, rect=rect, background=bg, chunk=args.chunk, return_stats=True)
        gt = v.image[rect[1]:rect[3], rect[0]:rect[2]]
        l1 = (img - gt).abs().mean()
        loss = (1 - args.ssim_weight) * l1 + args.ssim_weight * (1 - ssim(img, gt))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        pixels_seen += gt.shape[0] * gt.shape[1]
        if step % 10 == 0 or step == 1:
            wr.writerow([step, round(time.time() - t_train, 2), round(loss.item(), 5), round(l1.item(), 5),
                         round(psnr(img.detach(), gt), 3), gt.shape[0] * gt.shape[1], st["pairs"]])
            log.flush()
        if step % 100 == 0:
            print(f"step {step} {time.time() - t_train:.0f}s loss {loss.item():.4f}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            q = summarize(evaluate(g, scene.test, dev, bg, limit=4))
            q.update(step=step, elapsed_s=time.time() - t_train)
            quick_rows.append(q)
            print(f"  quick eval step {step}: psnr {q['psnr']:.3f} ssim {q['ssim']:.4f}", flush=True)
    train_s = time.time() - t_train
    log.close()

    final = evaluate(g, scene.test, dev, bg, save_dir=out / "test_renders")
    fin_s = summarize(final)
    print(f"final eval psnr {fin_s['psnr']:.3f} ssim {fin_s['ssim']:.4f} after {step} steps, {train_s / 60:.1f} min", flush=True)

    save_checkpoint(g, out / "gaussians.pt")
    n_splat = write_splat(g, out / "scene.splat")
    write_cameras_json(scene, out / "cameras.json")
    with open(out / "per_view.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "view", "psnr", "ssim", "render_s"])
        w.writeheader()
        for stage, rows in (("init", init), ("final", final)):
            for r in rows:
                w.writerow({"stage": stage, **r})
    summary = {
        "scene": str(args.scene),
        "device": dev,
        "downscale": args.downscale,
        "image_size": [scene.test[0].cam.width, scene.test[0].cam.height],
        "num_gaussians": g.n,
        "train_views": len(scene.train),
        "test_views": len(scene.test),
        "steps": step,
        "crop": args.crop,
        "train_minutes": train_s / 60,
        "pixels_seen": pixels_seen,
        "full_image_equivalents": pixels_seen / (scene.test[0].cam.width * scene.test[0].cam.height),
        "init": init_s,
        "final": fin_s,
        "quick_evals": quick_rows,
        "splat_file_gaussians": n_splat,
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
