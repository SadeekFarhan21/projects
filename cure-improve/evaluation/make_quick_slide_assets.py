#!/usr/bin/env python3
"""
Build slide-ready qualitative assets from before/after folders.

Given matching image names in:
  before/<name>.png
  after/<name>.png

This script creates:
  - Per-concept side-by-side panels with labels
  - A single contact-sheet summary grid
  - A JSON + Markdown summary with quick pixel-difference stats

It is intentionally lightweight and does not require torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int = 24) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def mae_score(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    return float(np.mean(np.abs(arr_a - arr_b)))


def make_panel(
    before: Image.Image,
    after: Image.Image,
    label: str,
    score: float,
    pad: int = 24,
    title_h: int = 44,
) -> Image.Image:
    before = before.convert("RGB")
    after = after.convert("RGB")
    w, h = before.size
    if after.size != (w, h):
        after = after.resize((w, h), Image.Resampling.BICUBIC)

    out_w = (2 * w) + (3 * pad)
    out_h = h + (2 * pad) + title_h
    canvas = Image.new("RGB", (out_w, out_h), "white")

    canvas.paste(before, (pad, pad + title_h))
    canvas.paste(after, (2 * pad + w, pad + title_h))

    draw = ImageDraw.Draw(canvas)
    f_title = load_font(24)
    f_label = load_font(18)

    draw.text((pad, 8), f"{label}  |  pixel MAE={score:.4f}", fill="black", font=f_title)
    draw.text((pad, pad + 8), "Before", fill="black", font=f_label)
    draw.text((2 * pad + w, pad + 8), "After", fill="black", font=f_label)

    return canvas


def make_contact_sheet(images: List[Image.Image], cols: int = 2, pad: int = 24) -> Image.Image:
    if not images:
        raise ValueError("No images provided for contact sheet.")

    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out_w = cols * w + (cols + 1) * pad
    out_h = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), "white")

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(img, (x, y))
    return canvas


def collect_pairs(before_dir: Path, after_dir: Path) -> List[Tuple[Path, Path, str]]:
    before_files = sorted([p for p in before_dir.glob("*.png") if p.is_file()])
    pairs: List[Tuple[Path, Path, str]] = []

    for b in before_files:
        a = after_dir / b.name.replace("before", "after")
        if not a.exists():
            a = after_dir / b.name
        if not a.exists():
            # fallback: exact stem match
            alt = list(after_dir.glob(f"{b.stem}.*"))
            if alt:
                a = alt[0]
            else:
                continue

        label = b.stem
        # normalize common suffix pattern like "car_0"
        if label.endswith("_0"):
            label = label[:-2]
        pairs.append((b, a, label))

    return pairs


def write_markdown_summary(out_md: Path, rows: List[Dict[str, object]], overview_img: Path) -> None:
    lines = [
        "# Quick Slide Pack",
        "",
        f"Overview image: `{overview_img}`",
        "",
        "| Concept | Pixel MAE | Panel |",
        "|---|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['concept']} | {r['pixel_mae']:.4f} | `{r['panel_path']}` |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create slide-ready before/after assets")
    parser.add_argument("--before-dir", type=Path, default=Path("outputs/demo_sequential/before"))
    parser.add_argument("--after-dir", type=Path, default=Path("outputs/demo_sequential/after"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/slide_pack"))
    parser.add_argument("--cols", type=int, default=2, help="Columns in summary contact sheet")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = args.out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.before_dir, args.after_dir)
    if not pairs:
        raise SystemExit(
            f"No before/after pairs found. Checked:\n  {args.before_dir}\n  {args.after_dir}"
        )

    rows: List[Dict[str, object]] = []
    panel_images: List[Image.Image] = []

    for i, (before_path, after_path, concept) in enumerate(pairs, start=1):
        before = Image.open(before_path)
        after = Image.open(after_path)
        score = mae_score(before, after)
        panel = make_panel(before, after, concept, score)

        panel_path = panels_dir / f"{i:02d}_{concept}.png"
        panel.save(panel_path)
        panel_images.append(panel)

        rows.append(
            {
                "index": i,
                "concept": concept,
                "before_path": str(before_path),
                "after_path": str(after_path),
                "panel_path": str(panel_path),
                "pixel_mae": score,
            }
        )

    overview = make_contact_sheet(panel_images, cols=max(1, args.cols))
    overview_path = args.out_dir / "overview.png"
    overview.save(overview_path)

    summary_json = args.out_dir / "summary.json"
    summary_md = args.out_dir / "summary.md"
    summary_payload = {
        "n_pairs": len(rows),
        "before_dir": str(args.before_dir),
        "after_dir": str(args.after_dir),
        "overview_path": str(overview_path),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_markdown_summary(summary_md, rows, overview_path)

    print(f"Wrote: {overview_path}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    print(f"Panels: {panels_dir}")


if __name__ == "__main__":
    main()
