#!/usr/bin/env python3
"""
Unified evaluation runner for cure / cure_seq / cure_dit.

This script standardizes:
1) concept sets
2) seeds
3) generation settings
4) run metadata and JSON outputs

It is designed for re-baselining after core method changes (for example,
embedding aggregation updates).
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure workspace root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cure.utils import EMBEDDING_MODES
from evaluation.protocol import (
    SCHEMA_VERSION,
    EvalConfig,
    build_concept_specs,
    build_run_id,
    config_to_dict,
    resolve_concepts,
    utc_now_iso,
    write_json,
)


DEFAULT_MODELS = {
    "cure": "CompVis/stable-diffusion-v1-4",
    "cure_seq": "CompVis/stable-diffusion-v1-4",
    "cure_dit": "stabilityai/stable-diffusion-3.5-medium",
}


def safe_pkg_version(pkg_name: str) -> str:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def detect_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float16 if device == "cuda" else torch.float32

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_arg]


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.strip().lower()).strip("_")


def git_commit(root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def create_eraser(
    branch: str,
    model_id: str,
    device: str,
    dtype: torch.dtype,
    embedding_mode: str,
    cache_dir: Path,
):
    if branch in {"cure", "cure_seq"}:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            cache_dir=str(cache_dir),
        )
        if branch == "cure":
            from cure import CURE

            return CURE(pipe, device=device, embedding_mode=embedding_mode)

        from cure_seq import SequentialCURE

        return SequentialCURE(pipe, device=device, embedding_mode=embedding_mode)

    if branch == "cure_dit":
        from diffusers import StableDiffusion3Pipeline
        from cure_dit import SD3CURE

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=str(cache_dir),
        )
        return SD3CURE(pipe, device=device, embedding_mode=embedding_mode)

    raise ValueError(f"Unsupported branch '{branch}'")


def generate_and_save(
    eraser,
    prompt: str,
    samples_per_concept: int,
    base_seed: int,
    steps: int,
    guidance_scale: float,
    out_dir: Path,
    prefix: str,
) -> Tuple[List[str], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[str] = []
    total_time = 0.0

    for i in range(samples_per_concept):
        seed = base_seed + i
        generator = torch.Generator(device="cpu").manual_seed(seed)

        t0 = time.time()
        images = eraser.generate(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        total_time += time.time() - t0

        if not images:
            continue

        path = out_dir / f"{prefix}_{i:03d}.png"
        images[0].save(path)
        image_paths.append(str(path))

    return image_paths, total_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument("--branch", choices=["cure", "cure_seq", "cure_dit"], required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu/mps")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        default="mean_masked",
        choices=EMBEDDING_MODES,
    )
    parser.add_argument("--concept-set", type=str, default="objects10")
    parser.add_argument("--concepts", type=str, default=None, help="Comma-separated concept list")
    parser.add_argument("--max-concepts", type=int, default=None)
    parser.add_argument(
        "--erasure-mode",
        choices=["auto", "isolated", "sequential"],
        default="auto",
        help="isolated: reset model per concept; sequential: accumulate erasures",
    )
    parser.add_argument("--samples-per-concept", type=int, default=1)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-object", type=float, default=2.0)
    parser.add_argument("--alpha-nsfw", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="outputs/shared_eval")
    parser.add_argument("--cache-dir", type=str, default="./models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    model_id = args.model_id or DEFAULT_MODELS[args.branch]
    concepts = resolve_concepts(
        concept_set=args.concept_set,
        concepts_csv=args.concepts,
        max_concepts=args.max_concepts,
    )
    specs = build_concept_specs(
        concepts=concepts,
        alpha_object=args.alpha_object,
        alpha_nsfw=args.alpha_nsfw,
    )

    erasure_mode = args.erasure_mode
    if erasure_mode == "auto":
        erasure_mode = "sequential" if args.branch == "cure_seq" else "isolated"
    if args.branch == "cure_seq" and erasure_mode == "isolated":
        raise ValueError("cure_seq does not support isolated mode; use sequential.")

    run_id = build_run_id(args.branch)
    run_dir = Path(args.output_dir) / run_id
    images_dir = run_dir / "images"

    config = EvalConfig(
        branch=args.branch,
        model_id=model_id,
        device=device,
        embedding_mode=args.embedding_mode,
        concept_set=args.concept_set,
        concepts=concepts,
        erasure_mode=erasure_mode,
        samples_per_concept=args.samples_per_concept,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        alpha_object=args.alpha_object,
        alpha_nsfw=args.alpha_nsfw,
        output_dir=str(Path(args.output_dir)),
        run_id=run_id,
    )

    if args.dry_run:
        print("Dry run config:")
        print(config_to_dict(config))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", config_to_dict(config))

    print(f"Run ID: {run_id}")
    print(f"Branch: {args.branch}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Embedding mode: {args.embedding_mode}")
    print(f"Erasure mode: {erasure_mode}")
    print(f"Concepts ({len(concepts)}): {concepts}")

    eraser = create_eraser(
        branch=args.branch,
        model_id=model_id,
        device=device,
        dtype=dtype,
        embedding_mode=args.embedding_mode,
        cache_dir=Path(args.cache_dir),
    )

    # Save clean checkpoint for isolated mode if available
    if erasure_mode == "isolated" and hasattr(eraser, "save_original_weights"):
        eraser.save_original_weights()

    concept_results: List[Dict[str, Any]] = []
    total_erase_time = 0.0
    total_before_time = 0.0
    total_after_time = 0.0

    for idx, spec in enumerate(specs, start=1):
        slug = slugify(spec.concept)
        concept_dir = images_dir / f"{idx:03d}_{slug}"
        prompt = spec.forget_prompts[0] if spec.forget_prompts else spec.concept

        print(f"[{idx:03d}/{len(specs):03d}] Concept='{spec.concept}' alpha={spec.alpha}")

        before_paths, before_time = generate_and_save(
            eraser=eraser,
            prompt=prompt,
            samples_per_concept=args.samples_per_concept,
            base_seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            out_dir=concept_dir / "before",
            prefix="before",
        )
        total_before_time += before_time

        t0 = time.time()
        if args.branch == "cure":
            erase_stats = eraser.erase_concept(
                forget_prompts=spec.forget_prompts,
                retain_prompts=spec.retain_prompts,
                alpha=spec.alpha,
                save_original=False,
            )
        elif args.branch == "cure_seq":
            erase_stats = eraser.erase_concept(
                forget_prompts=spec.forget_prompts,
                retain_prompts=spec.retain_prompts,
                alpha=spec.alpha,
                concept_name=spec.concept,
                save_original=False,
            )
        else:
            erase_stats = eraser.erase_concept(
                forget_prompts=spec.forget_prompts,
                retain_prompts=spec.retain_prompts,
                alpha=spec.alpha,
                concept_name=spec.concept,
                save_original=False,
            )
        erase_time = time.time() - t0
        total_erase_time += erase_time

        after_paths, after_time = generate_and_save(
            eraser=eraser,
            prompt=prompt,
            samples_per_concept=args.samples_per_concept,
            base_seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            out_dir=concept_dir / "after",
            prefix="after",
        )
        total_after_time += after_time

        # Reset model after each concept in isolated mode
        if erasure_mode == "isolated" and hasattr(eraser, "restore_original_weights"):
            eraser.restore_original_weights()

        concept_results.append(
            {
                "index": idx,
                "concept": spec.concept,
                "alpha": spec.alpha,
                "forget_prompt_count": len(spec.forget_prompts),
                "retain_prompt_count": len(spec.retain_prompts),
                "prompt_used": prompt,
                "before": {
                    "image_paths": before_paths,
                    "generation_time_s": before_time,
                },
                "erase": {
                    "erase_time_s": erase_time,
                    "eraser_stats": erase_stats,
                },
                "after": {
                    "image_paths": after_paths,
                    "generation_time_s": after_time,
                },
            }
        )

        # Persist incrementally for crash-safe long runs
        partial_payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "config": config_to_dict(config),
            "concept_results": concept_results,
        }
        write_json(run_dir / "results.partial.json", partial_payload)

    summary = {
        "n_concepts": len(concept_results),
        "total_erase_time_s": total_erase_time,
        "total_before_generation_time_s": total_before_time,
        "total_after_generation_time_s": total_after_time,
        "total_generation_time_s": total_before_time + total_after_time,
    }

    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "run_id": run_id,
        "git_commit": git_commit(ROOT),
        "config": config_to_dict(config),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "diffusers": safe_pkg_version("diffusers"),
            "transformers": safe_pkg_version("transformers"),
        },
        "concept_results": concept_results,
        "summary": summary,
    }

    write_json(run_dir / "results.json", results_payload)
    print(f"Saved protocol results: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
