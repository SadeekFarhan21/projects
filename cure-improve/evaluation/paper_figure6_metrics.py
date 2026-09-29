#!/usr/bin/env python3
"""
Figure-6-style scalability evaluation for concept erasure.

This script computes paper-aligned proxy curves for sequential artist erasure:
  - LPIPSe (erased artists):   higher = stronger visual divergence from base SD
  - LPIPSu (unerased artists): lower = better preservation of untargeted styles
  - CLIP_u  (unerased prompts): higher = better prompt-image alignment

It supports CURE and CURE-Sequential and logs metrics at configurable erase checkpoints.

Notes:
  - Metrics here are proxy metrics for rapid analysis, not a full benchmark suite.
  - Requires torch, diffusers, transformers, lpips, Pillow, matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_STYLE_TEMPLATES = [
    "a portrait in the style of {artist}",
    "an oil painting in the style of {artist}",
    "a landscape painting in the style of {artist}",
    "a still life in the style of {artist}",
    "a city scene in the style of {artist}",
    "a nighttime scene in the style of {artist}",
    "a fantasy artwork in the style of {artist}",
    "a surreal composition in the style of {artist}",
    "a watercolor in the style of {artist}",
    "a charcoal drawing in the style of {artist}",
    "an impressionist painting by {artist}",
    "an abstract painting by {artist}",
    "a detailed illustration by {artist}",
    "a vivid color palette artwork by {artist}",
    "a moody atmospheric artwork by {artist}",
    "a dramatic lighting painting by {artist}",
    "a modern reinterpretation by {artist}",
    "a classical composition by {artist}",
    "a museum-quality painting by {artist}",
    "an expressive brushstroke painting by {artist}",
]

DEFAULT_ERASED_ARTISTS = [
    "Van Gogh",
    "Kelly McKernan",
    "Pablo Picasso",
    "Rembrandt",
    "Andy Warhol",
    "Caravaggio",
    "Thomas Kinkade",
    "Tyler Edlin",
    "Kilian Eng",
    "Ajin: DemiHuman",
]

DEFAULT_UNERASED_ARTISTS = [
    "Claude Monet",
    "Paul Cezanne",
    "Edgar Degas",
    "Henri Matisse",
    "Salvador Dali",
]

DEFAULT_CHECKPOINTS = [1, 5, 10, 50, 100]

SAFE_CONCEPT_MARKERS = (
    "nsfw",
    "nudity",
    "sexual",
    "violence",
    "blood",
)


@dataclass
class PromptRecord:
    idx: int
    group: str  # "erased" or "unerased"
    artist: str
    template_id: int
    prompt: str
    seed: int


@dataclass
class CheckpointMetrics:
    n_erased: int
    lpips_e_mean: float
    lpips_e_std: float
    lpips_u_mean: float
    lpips_u_std: float
    clip_u_mean: float
    clip_u_std: float
    n_erased_pairs: int
    n_unerased_pairs: int


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_csv(raw: str) -> List[int]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def parse_methods(raw: str) -> List[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    allowed = {"cure", "cure_seq"}
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unsupported method(s): {bad}. Allowed: {sorted(allowed)}")
    if not methods:
        raise ValueError("No methods provided.")
    return methods


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_unique_run_dir(output_root: Path) -> Tuple[str, Path]:
    """
    Create a unique timestamped run directory under output_root.

    Uses microsecond-resolution UTC timestamp and falls back to numeric suffix
    if a collision still occurs.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = output_root / run_id
    suffix = 1
    while out_dir.exists():
        out_dir = output_root / f"{run_id}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return run_id, out_dir


def detect_device(requested: Optional[str]):
    try:
        import torch
    except ImportError:
        return requested or "cpu"

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_templates(path: Optional[Path]) -> List[str]:
    if path is None:
        return list(DEFAULT_STYLE_TEMPLATES)
    templates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{artist}" not in line:
            raise ValueError(f"Template missing '{{artist}}': {line}")
        templates.append(line)
    if not templates:
        raise ValueError(f"No templates parsed from {path}")
    return templates


def load_name_list(path: Path) -> List[str]:
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    if not names:
        raise ValueError(f"No names parsed from {path}")
    return names


def build_prompt_records(
    erased_artists: Sequence[str],
    unerased_artists: Sequence[str],
    templates: Sequence[str],
    seeds: Sequence[int],
    max_prompts_per_group: Optional[int] = None,
) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    idx = 0
    group_counts = {"erased": 0, "unerased": 0}

    def add_group(group: str, artists: Sequence[str]) -> None:
        nonlocal idx
        for artist in artists:
            for t_idx, template in enumerate(templates):
                prompt = template.format(artist=artist)
                for seed in seeds:
                    if max_prompts_per_group is not None and group_counts[group] >= max_prompts_per_group:
                        return
                    records.append(
                        PromptRecord(
                            idx=idx,
                            group=group,
                            artist=artist,
                            template_id=t_idx,
                            prompt=prompt,
                            seed=seed,
                        )
                    )
                    idx += 1
                    group_counts[group] += 1

    add_group("erased", erased_artists)
    add_group("unerased", unerased_artists)
    return records


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def ensure_no_unsafe_concepts(concepts: Sequence[str]) -> None:
    lowered = [c.lower() for c in concepts]
    bad = [c for c in lowered if any(m in c for m in SAFE_CONCEPT_MARKERS)]
    if bad:
        raise ValueError(
            "Unsafe concept markers detected in erase list. "
            f"Refusing to run with concepts: {bad}"
        )


def make_sd14_pipe(device: str, cache_dir: Path, dtype):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(cache_dir),
    )
    return pipe.to(device)


def build_eraser(method: str, device: str, cache_dir: Path, embedding_mode: str, spectral_mode: str):
    import torch

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = make_sd14_pipe(device=device, cache_dir=cache_dir, dtype=dtype)
    if method == "cure":
        from cure import CURE

        return CURE(pipe, device=device, embedding_mode=embedding_mode, spectral_mode=spectral_mode)
    if method == "cure_seq":
        from cure_seq import SequentialCURE

        return SequentialCURE(pipe, device=device, embedding_mode=embedding_mode, spectral_mode=spectral_mode)
    raise ValueError(f"Unknown method '{method}'")


def generate_image(
    generator_model,
    prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
):
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    if hasattr(generator_model, "generate"):
        images = generator_model.generate(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            height=height,
            width=width,
        )
        return images[0]

    # Raw diffusers pipeline path
    output = generator_model(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
        height=height,
        width=width,
    )
    return output.images[0]


def load_lpips_model(device: str, lpips_net: str):
    import lpips

    model = lpips.LPIPS(net=lpips_net).to(device)
    model.eval()
    return model


def pil_to_lpips_tensor(image, device):
    import numpy as np
    import torch

    arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def compute_lpips_score(lpips_model, img_a, img_b, device: str) -> float:
    import torch

    ta = pil_to_lpips_tensor(img_a, device=device)
    tb = pil_to_lpips_tensor(img_b, device=device)
    with torch.no_grad():
        score = lpips_model(ta, tb)
    return float(score.item())


def load_clip_for_score(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def compute_clip_scores(images, prompts: Sequence[str], clip_model, clip_processor, device: str, batch_size: int = 8):
    import torch

    if len(images) != len(prompts):
        raise ValueError("images and prompts lengths differ")

    out_scores: List[float] = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        inputs = clip_processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosine = (image_embeds * text_embeds).sum(dim=-1)
            # CLIPScore-style scaling
            scores = torch.clamp(100.0 * cosine, min=0.0)
            out_scores.extend([float(x) for x in scores.cpu().tolist()])
    return out_scores


def erase_once(eraser, method: str, concept: str, alpha: float):
    from cure.utils import get_default_forget_prompts

    prompts = get_default_forget_prompts(concept)
    if method == "cure":
        eraser.erase_concept(
            forget_prompts=prompts,
            retain_prompts=None,
            alpha=alpha,
            save_original=False,
        )
        return
    if method == "cure_seq":
        eraser.erase_concept(
            forget_prompts=prompts,
            retain_prompts=None,
            alpha=alpha,
            concept_name=concept,
            save_original=False,
        )
        return
    raise ValueError(f"Unknown method '{method}'")


def maybe_load_or_generate_baselines(
    baseline_dir: Path,
    records: Sequence[PromptRecord],
    baseline_generator,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    write_images: bool = True,
):
    from PIL import Image

    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_images = []
    for rec in records:
        out_path = baseline_dir / f"{rec.idx:06d}.png"
        if out_path.exists():
            image = Image.open(out_path).convert("RGB")
            baseline_images.append(image)
            continue

        image = generate_image(
            baseline_generator,
            prompt=rec.prompt,
            seed=rec.seed,
            steps=steps,
            guidance=guidance,
            height=height,
            width=width,
        )
        if write_images:
            image.save(out_path)
        baseline_images.append(image.convert("RGB"))
    return baseline_images


def evaluate_checkpoint(
    eraser,
    method: str,
    n_erased: int,
    records: Sequence[PromptRecord],
    baseline_images,
    lpips_model,
    clip_model,
    clip_processor,
    device: str,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    out_images_dir: Optional[Path],
) -> CheckpointMetrics:
    lpips_e: List[float] = []
    lpips_u: List[float] = []
    clip_u_images = []
    clip_u_prompts = []

    if out_images_dir is not None:
        out_images_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        image = generate_image(
            eraser,
            prompt=rec.prompt,
            seed=rec.seed,
            steps=steps,
            guidance=guidance,
            height=height,
            width=width,
        ).convert("RGB")

        if out_images_dir is not None:
            image.save(out_images_dir / f"{rec.idx:06d}.png")

        lp = compute_lpips_score(lpips_model, baseline_images[rec.idx], image, device=device)
        if rec.group == "erased":
            lpips_e.append(lp)
        else:
            lpips_u.append(lp)
            clip_u_images.append(image)
            clip_u_prompts.append(rec.prompt)

    clip_u_scores = compute_clip_scores(
        clip_u_images,
        clip_u_prompts,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
    )

    lpips_e_mean, lpips_e_std = _mean_std(lpips_e)
    lpips_u_mean, lpips_u_std = _mean_std(lpips_u)
    clip_u_mean, clip_u_std = _mean_std(clip_u_scores)

    print(
        f"[{method}] n_erased={n_erased:4d} "
        f"LPIPSe={lpips_e_mean:.4f} LPIPSu={lpips_u_mean:.4f} CLIP_u={clip_u_mean:.4f}"
    )

    return CheckpointMetrics(
        n_erased=n_erased,
        lpips_e_mean=lpips_e_mean,
        lpips_e_std=lpips_e_std,
        lpips_u_mean=lpips_u_mean,
        lpips_u_std=lpips_u_std,
        clip_u_mean=clip_u_mean,
        clip_u_std=clip_u_std,
        n_erased_pairs=len(lpips_e),
        n_unerased_pairs=len(lpips_u),
    )


def write_summary_md(out_path: Path, payload: Dict) -> None:
    lines = []
    cfg = payload["config"]
    lines.append("# Figure-6 Style Metrics Summary")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Methods: {cfg['methods']}")
    lines.append(f"- Erased concepts ({len(cfg['erased_concepts'])}): {cfg['erased_concepts']}")
    lines.append(f"- Unerased artists ({len(cfg['unerased_artists'])}): {cfg['unerased_artists']}")
    lines.append(f"- Checkpoints: {cfg['checkpoints']}")
    lines.append(f"- Seeds: {cfg['seeds']}")
    lines.append(f"- Steps: {cfg['steps']}, Guidance: {cfg['guidance']}, Size: {cfg['height']}x{cfg['width']}")
    lines.append("")

    for method, stats in payload["results"].items():
        lines.append(f"## Method: {method}")
        lines.append("")
        lines.append("| # Erased | LPIPSe (↑) | LPIPSu (↓) | CLIP_u (↑) |")
        lines.append("|---:|---:|---:|---:|")
        for cp in stats["checkpoints"]:
            lines.append(
                f"| {cp['n_erased']} "
                f"| {cp['lpips_e_mean']:.4f} +- {cp['lpips_e_std']:.4f} "
                f"| {cp['lpips_u_mean']:.4f} +- {cp['lpips_u_std']:.4f} "
                f"| {cp['clip_u_mean']:.4f} +- {cp['clip_u_std']:.4f} |"
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- LPIPSe higher means stronger divergence from baseline on erased artists (stronger removal proxy).")
    lines.append("- LPIPSu lower means less collateral change on unerased artists (better preservation).")
    lines.append("- CLIP_u higher means better text-image alignment on unerased prompts.")
    lines.append("- This reproduces Figure-6-style trend tracking, not a full benchmark.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def create_plots(results_by_method: Dict[str, List[CheckpointMetrics]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-method Fig-6 style dual-axis
    for method, cps in results_by_method.items():
        xs = [cp.n_erased for cp in cps]
        clip_u = [cp.clip_u_mean for cp in cps]
        lpips_u = [cp.lpips_u_mean for cp in cps]

        fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax1.set_xscale("log")
        ax1.plot(xs, clip_u, marker="o", color="#1f77b4", label="CLIP_u")
        ax1.set_xlabel("# Artists Erased (log scale)")
        ax1.set_ylabel("CLIP Score (unerased) ↑", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(xs, lpips_u, marker="s", color="#d62728", label="LPIPSu")
        ax2.set_ylabel("LPIPS (unerased) ↓", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        plt.title(f"Figure-6 Style Curve ({method})")
        fig.savefig(out_dir / f"figure6_like_{method}.png", dpi=220)
        plt.close(fig)

    # Cross-method overlay for LPIPSu
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.set_xscale("log")
    for method, cps in results_by_method.items():
        ax.plot(
            [cp.n_erased for cp in cps],
            [cp.lpips_u_mean for cp in cps],
            marker="o",
            label=f"{method} LPIPSu",
        )
    ax.set_xlabel("# Artists Erased (log scale)")
    ax.set_ylabel("LPIPSu ↓")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title("Unerased-Style Perceptual Drift")
    fig.savefig(out_dir / "lpips_u_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure-6 style LPIPS/CLIP evaluation for artist erasure.")
    parser.add_argument("--methods", type=str, default="cure,cure_seq")
    parser.add_argument("--erased-concepts", type=str, default=None)
    parser.add_argument("--erased-concepts-file", type=str, default=None)
    parser.add_argument("--unerased-artists", type=str, default=None)
    parser.add_argument("--unerased-artists-file", type=str, default=None)
    parser.add_argument("--templates-file", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default="1,5,10")
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--embedding-mode", type=str, default="mean_masked")
    parser.add_argument("--spectral-mode", type=str, default="tikhonov")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./models")
    parser.add_argument("--output-dir", type=str, default="outputs/figure6_eval")
    parser.add_argument("--max-prompts-per-group", type=int, default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    if args.erased_concepts_file:
        erased_concepts = load_name_list(Path(args.erased_concepts_file))
    elif args.erased_concepts:
        erased_concepts = parse_csv_list(args.erased_concepts)
    else:
        erased_concepts = list(DEFAULT_ERASED_ARTISTS)

    if args.unerased_artists_file:
        unerased_artists = load_name_list(Path(args.unerased_artists_file))
    elif args.unerased_artists:
        unerased_artists = parse_csv_list(args.unerased_artists)
    else:
        unerased_artists = list(DEFAULT_UNERASED_ARTISTS)

    checkpoints = sorted(set(parse_int_csv(args.checkpoints)))
    seeds = parse_int_csv(args.seeds)
    templates = load_templates(Path(args.templates_file) if args.templates_file else None)

    ensure_no_unsafe_concepts(erased_concepts)
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")
    if not erased_concepts:
        raise ValueError("No erased concepts parsed from --erased-concepts")
    if not checkpoints:
        raise ValueError("No checkpoints parsed from --checkpoints")
    if max(checkpoints) > len(erased_concepts):
        raise ValueError(
            f"Checkpoint {max(checkpoints)} exceeds number of erased concepts ({len(erased_concepts)})."
        )

    records = build_prompt_records(
        erased_artists=erased_concepts,
        unerased_artists=unerased_artists,
        templates=templates,
        seeds=seeds,
        max_prompts_per_group=args.max_prompts_per_group,
    )
    n_erased_records = sum(1 for r in records if r.group == "erased")
    n_unerased_records = sum(1 for r in records if r.group == "unerased")

    device = detect_device(args.device)
    run_id, out_dir = make_unique_run_dir(Path(args.output_dir))

    print(f"Device: {device}")
    print(f"Methods: {methods}")
    print(f"Erased concepts: {len(erased_concepts)}")
    print(f"Unerased artists: {len(unerased_artists)}")
    print(f"Templates: {len(templates)}")
    print(f"Seeds: {seeds}")
    print(f"Records: erased={n_erased_records}, unerased={n_unerased_records}, total={len(records)}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Output: {out_dir}")

    if args.dry_run:
        payload = {
            "created_at_utc": now_utc(),
            "config": {
                "methods": methods,
                "erased_concepts": erased_concepts,
                "unerased_artists": unerased_artists,
                "checkpoints": checkpoints,
                "seeds": seeds,
                "steps": args.steps,
                "guidance": args.guidance,
                "height": args.height,
                "width": args.width,
                "embedding_mode": args.embedding_mode,
                "spectral_mode": args.spectral_mode,
                "clip_model": args.clip_model,
                "lpips_net": args.lpips_net,
                "device": device,
                "erased_concepts_file": args.erased_concepts_file,
                "unerased_artists_file": args.unerased_artists_file,
                "run_id": run_id,
            },
            "record_counts": {
                "erased": n_erased_records,
                "unerased": n_unerased_records,
                "total": len(records),
            },
        }
        (out_dir / "dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved dry-run config: {out_dir / 'dry_run.json'}")
        return

    import torch

    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_dir = Path(args.cache_dir)

    # Baseline images from clean model for paired LPIPS.
    clean_pipe = make_sd14_pipe(device=device, cache_dir=cache_dir, dtype=dtype)
    baseline_generator = clean_pipe
    baseline_dir = out_dir / "baseline"
    baseline_images = maybe_load_or_generate_baselines(
        baseline_dir=baseline_dir,
        records=records,
        baseline_generator=baseline_generator,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        write_images=args.save_images,
    )
    del clean_pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    lpips_model = load_lpips_model(device=device, lpips_net=args.lpips_net)
    clip_model, clip_processor = load_clip_for_score(args.clip_model, device=device)

    results: Dict[str, Dict] = {}
    results_for_plot: Dict[str, List[CheckpointMetrics]] = {}

    for method in methods:
        print(f"\n=== Method: {method} ===")
        eraser = build_eraser(
            method=method,
            device=device,
            cache_dir=cache_dir,
            embedding_mode=args.embedding_mode,
            spectral_mode=args.spectral_mode,
        )

        method_checkpoints: List[CheckpointMetrics] = []
        for step, concept in enumerate(erased_concepts, start=1):
            erase_once(eraser, method=method, concept=concept, alpha=args.alpha)
            if step in checkpoints:
                eval_dir = out_dir / method / f"k_{step}"
                cp = evaluate_checkpoint(
                    eraser=eraser,
                    method=method,
                    n_erased=step,
                    records=records,
                    baseline_images=baseline_images,
                    lpips_model=lpips_model,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=device,
                    steps=args.steps,
                    guidance=args.guidance,
                    height=args.height,
                    width=args.width,
                    out_images_dir=eval_dir if args.save_images else None,
                )
                method_checkpoints.append(cp)

        results[method] = {"checkpoints": [asdict(cp) for cp in method_checkpoints]}
        results_for_plot[method] = method_checkpoints

        del eraser
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "created_at_utc": now_utc(),
        "config": {
            "methods": methods,
            "erased_concepts": erased_concepts,
            "unerased_artists": unerased_artists,
            "checkpoints": checkpoints,
            "seeds": seeds,
            "steps": args.steps,
            "guidance": args.guidance,
            "height": args.height,
            "width": args.width,
            "alpha": args.alpha,
            "embedding_mode": args.embedding_mode,
            "spectral_mode": args.spectral_mode,
            "clip_model": args.clip_model,
            "lpips_net": args.lpips_net,
            "device": device,
            "cache_dir": str(cache_dir),
            "erased_concepts_file": args.erased_concepts_file,
            "unerased_artists_file": args.unerased_artists_file,
            "templates_file": args.templates_file,
            "n_templates": len(templates),
            "max_prompts_per_group": args.max_prompts_per_group,
            "run_id": run_id,
        },
        "record_counts": {
            "erased": n_erased_records,
            "unerased": n_unerased_records,
            "total": len(records),
        },
        "results": results,
    }

    results_json = out_dir / "results.json"
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", payload)
    create_plots(results_for_plot, out_dir / "plots")

    print(f"\nSaved: {results_json}")
    print(f"Saved: {out_dir / 'summary.md'}")
    print(f"Saved plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
