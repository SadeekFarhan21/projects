#!/usr/bin/env python3
"""
Create a consolidated quick-proof summary across multiple results.json files.

Outputs:
- summary markdown table with key findings
- machine-readable CSV
- lightweight plots (if matplotlib is installed)

Example:
  python3 evaluation/summarize_quick_proof_results.py \
    --results-glob "<OUTPUT_DIR>/outputs/quick_proof/**/results.json" \
    --output-md evaluation/summary.md \
    --output-csv evaluation/summary.csv \
    --plots-dir evaluation/plots
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_RESULTS_GLOBS = [
    "<OUTPUT_DIR>/**/results.json",
    "<OUTPUT_DIR>/outputs/quick_proof/**/results.json",
]


def _pick_number(data: Dict, *keys: str) -> float:
    for key in keys:
        if key in data:
            return float(data[key])
    raise KeyError(f"Missing any of keys={keys}")


def _aggregate_block(method_block: Dict) -> Dict:
    if isinstance(method_block, dict) and "aggregate" in method_block:
        return method_block["aggregate"]
    return method_block


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_canonical_path(path: Path) -> bool:
    return "/outputs/quick_proof/" in str(path)


def _short_spectral_name(name: str) -> str:
    if name == "gavish_donoho":
        return "gd"
    if name == "tikhonov":
        return "tik"
    return name


@dataclass
class RunRow:
    path: str
    run_id: str
    experiment: str
    spectral_mode: str
    concept_order: str
    alpha: float
    n_concepts: int
    n_seeds: int
    cure_target: float
    seq_target: float
    seq_minus_cure_target: float
    cure_drop: float
    seq_drop: float
    seq_minus_cure_drop: float
    cure_single_delta: float
    seq_single_delta: float
    malformed_concepts: bool

    @property
    def run_label(self) -> str:
        order = "fwd" if self.concept_order == "forward" else "rev"
        return f"{_short_spectral_name(self.spectral_mode)}-{order}-{self.n_concepts}c-a{self.alpha:g}"

    @property
    def verdict(self) -> str:
        better_target = self.seq_minus_cure_target < 0
        better_drop = self.seq_minus_cure_drop < 0
        if better_target and better_drop:
            return "Seq wins both"
        if better_target and not better_drop:
            return "Seq better suppression"
        if not better_target and better_drop:
            return "Seq better retention"
        return "CURE wins both"


def _parse_row(path: Path) -> RunRow:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    results = payload["results"]

    cure = _aggregate_block(results["cure"])
    seq = _aggregate_block(results["cure_seq"])

    cure_last = cure["checkpoints"][-1]
    seq_last = seq["checkpoints"][-1]

    concepts_ordered = config.get("concepts_ordered", config.get("concepts", []))
    seeds = config.get("seeds")
    if seeds is None:
        seed = config.get("seed")
        seeds = [seed] if seed is not None else []

    concepts_clean = [str(c).strip().lower() for c in concepts_ordered]
    malformed_tokens = {"chain", "saw", "springer"}
    malformed_concepts = any(token in malformed_tokens for token in concepts_clean)

    cure_target = _pick_number(cure_last, "target_clip_mean", "target_clip")
    seq_target = _pick_number(seq_last, "target_clip_mean", "target_clip")
    cure_drop = _pick_number(cure_last, "retention_drop_vs_base_mean", "retention_drop_vs_base")
    seq_drop = _pick_number(seq_last, "retention_drop_vs_base_mean", "retention_drop_vs_base")

    cure_single_delta = _pick_number(cure, "single_concept_delta_mean", "single_concept_delta")
    seq_single_delta = _pick_number(seq, "single_concept_delta_mean", "single_concept_delta")

    return RunRow(
        path=str(path),
        run_id=path.parent.name,
        experiment=path.parent.parent.name,
        spectral_mode=str(config.get("spectral_mode", "unknown")),
        concept_order=str(config.get("concept_order", "unknown")),
        alpha=float(config.get("alpha", 0.0)),
        n_concepts=len(concepts_ordered),
        n_seeds=len([s for s in seeds if s is not None]),
        cure_target=cure_target,
        seq_target=seq_target,
        seq_minus_cure_target=(seq_target - cure_target),
        cure_drop=cure_drop,
        seq_drop=seq_drop,
        seq_minus_cure_drop=(seq_drop - cure_drop),
        cure_single_delta=cure_single_delta,
        seq_single_delta=seq_single_delta,
        malformed_concepts=malformed_concepts,
    )


def discover_unique_results(globs_in: Sequence[str]) -> Tuple[List[Path], Dict[str, int]]:
    all_paths: List[Path] = []
    for pattern in globs_in:
        all_paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))

    # Keep one canonical path per file content hash.
    chosen_by_hash: Dict[str, Path] = {}
    duplicate_count_by_hash: Dict[str, int] = {}

    for path in sorted(set(all_paths)):
        if not path.exists():
            continue
        file_hash = _hash_file(path)
        duplicate_count_by_hash[file_hash] = duplicate_count_by_hash.get(file_hash, 0) + 1
        current = chosen_by_hash.get(file_hash)
        if current is None:
            chosen_by_hash[file_hash] = path
            continue

        # Prefer canonical outputs/quick_proof copies.
        if _is_canonical_path(path) and not _is_canonical_path(current):
            chosen_by_hash[file_hash] = path

    dedupe_stats = {
        "all_paths": len(all_paths),
        "unique_hashes": len(chosen_by_hash),
        "duplicate_paths": max(0, len(all_paths) - len(chosen_by_hash)),
    }
    return sorted(chosen_by_hash.values()), dedupe_stats


def write_csv(rows: Sequence[RunRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_label",
                "experiment",
                "run_id",
                "spectral_mode",
                "concept_order",
                "alpha",
                "n_concepts",
                "n_seeds",
                "cure_target",
                "seq_target",
                "seq_minus_cure_target",
                "cure_drop",
                "seq_drop",
                "seq_minus_cure_drop",
                "cure_single_delta",
                "seq_single_delta",
                "verdict",
                "malformed_concepts",
                "results_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.run_label,
                    row.experiment,
                    row.run_id,
                    row.spectral_mode,
                    row.concept_order,
                    row.alpha,
                    row.n_concepts,
                    row.n_seeds,
                    row.cure_target,
                    row.seq_target,
                    row.seq_minus_cure_target,
                    row.cure_drop,
                    row.seq_drop,
                    row.seq_minus_cure_drop,
                    row.cure_single_delta,
                    row.seq_single_delta,
                    row.verdict,
                    row.malformed_concepts,
                    row.path,
                ]
            )


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _find_identical_gd_groups(rows: Sequence[RunRow]) -> List[str]:
    groups: Dict[Tuple[str, str, int, int], List[RunRow]] = {}
    for row in rows:
        if row.spectral_mode != "gavish_donoho":
            continue
        key = (row.spectral_mode, row.concept_order, row.n_concepts, row.n_seeds)
        groups.setdefault(key, []).append(row)

    findings: List[str] = []
    for key, group_rows in groups.items():
        if len(group_rows) < 2:
            continue
        by_metric = {
            (
                _fmt(r.cure_target),
                _fmt(r.seq_target),
                _fmt(r.cure_drop),
                _fmt(r.seq_drop),
                _fmt(r.cure_single_delta),
                _fmt(r.seq_single_delta),
            )
            for r in group_rows
        }
        if len(by_metric) == 1:
            alphas = sorted({r.alpha for r in group_rows})
            findings.append(
                f"GD {key[1]} ({key[2]} concepts, {key[3]} seeds) is alpha-invariant across {alphas}."
            )
    return findings


def write_markdown(rows: Sequence[RunRow], dedupe_stats: Dict[str, int], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Consolidated Quick-Proof Summary")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Unique runs summarized: {len(rows)}")
    lines.append(
        f"- Dedupe: {dedupe_stats['all_paths']} discovered paths -> {dedupe_stats['unique_hashes']} unique result files"
    )
    lines.append("")

    lines.append("## Final Checkpoint Comparison")
    lines.append("")
    lines.append(
        "| Run | Spectral | Order | Alpha | Concepts | Seeds | CURE target | Seq target | seq-cure target | CURE drop | Seq drop | seq-cure drop | Verdict |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        verdict = row.verdict
        if row.malformed_concepts:
            verdict += " (malformed concepts)"
        lines.append(
            f"| {row.experiment}/{row.run_id} | {row.spectral_mode} | {row.concept_order} | {row.alpha:g} "
            f"| {row.n_concepts} | {row.n_seeds} | {_fmt(row.cure_target)} | {_fmt(row.seq_target)} "
            f"| {_fmt(row.seq_minus_cure_target)} | {_fmt(row.cure_drop)} | {_fmt(row.seq_drop)} "
            f"| {_fmt(row.seq_minus_cure_drop)} | {verdict} |"
        )
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    valid_rows = [r for r in rows if not r.malformed_concepts]
    if valid_rows:
        wins_both = [r for r in valid_rows if r.verdict == "Seq wins both"]
        if wins_both:
            # More negative sum means larger combined win.
            best = min(wins_both, key=lambda r: (r.seq_minus_cure_target + r.seq_minus_cure_drop))
            lines.append(
                f"1. Best Seq-on-both run: `{best.experiment}/{best.run_id}` "
                f"(target delta {_fmt(best.seq_minus_cure_target)}, drop delta {_fmt(best.seq_minus_cure_drop)})."
            )
        else:
            lines.append("1. No valid run showed Sequential winning both suppression and retention simultaneously.")

        reverse_rows = [r for r in valid_rows if r.concept_order == "reverse"]
        if reverse_rows:
            best_ret_reverse = min(reverse_rows, key=lambda r: r.seq_minus_cure_drop)
            lines.append(
                f"2. Strongest reverse-order retention improvement: `{best_ret_reverse.experiment}/{best_ret_reverse.run_id}` "
                f"(seq-cure drop {_fmt(best_ret_reverse.seq_minus_cure_drop)})."
            )
    else:
        lines.append("1. No valid rows found (all detected as malformed).")

    gd_findings = _find_identical_gd_groups(valid_rows)
    if gd_findings:
        for idx, finding in enumerate(gd_findings, start=3):
            lines.append(f"{idx}. {finding}")
    else:
        lines.append("3. No alpha-invariance pattern detected for GD groups.")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `target` lower is better suppression.")
    lines.append("- `drop` lower is better retention preservation.")
    lines.append("- `seq-cure` values below 0 favor Sequential.")
    lines.append("- CLIP scores here are fast proxy metrics, not final benchmark metrics.")
    lines.append("")

    lines.append("## Canonical Result Paths")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row.path}`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def create_plots(rows: Sequence[RunRow], out_dir: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return "Skipped plots: matplotlib not installed."

    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [r.run_label for r in rows]
    x = np.arange(len(rows))
    width = 0.36

    # Panel 1: target (lower better)
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x - width / 2, [r.cure_target for r in rows], width, label="CURE", color="#4C78A8")
    ax.bar(x + width / 2, [r.seq_target for r in rows], width, label="CURE-Sequential", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Final target CLIP")
    ax.set_title("Final Suppression Proxy (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_dir / "final_target_clip.png", dpi=200)
    plt.close(fig)

    # Panel 2: retention drop (lower better)
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x - width / 2, [r.cure_drop for r in rows], width, label="CURE", color="#4C78A8")
    ax.bar(x + width / 2, [r.seq_drop for r in rows], width, label="CURE-Sequential", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Final retention drop vs base")
    ax.set_title("Final Retention Damage Proxy (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_dir / "final_retention_drop.png", dpi=200)
    plt.close(fig)

    # Panel 3: tradeoff scatter
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for row in rows:
        ax.scatter(row.cure_target, row.cure_drop, marker="o", color="#4C78A8", s=65)
        ax.scatter(row.seq_target, row.seq_drop, marker="s", color="#F58518", s=65)
        ax.annotate(row.run_label, (row.seq_target, row.seq_drop), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Final target CLIP (lower better)")
    ax.set_ylabel("Final retention drop (lower better)")
    ax.set_title("Suppression-Retention Tradeoff")
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "tradeoff_scatter.png", dpi=200)
    plt.close(fig)

    return "Plots written."


def _sort_rows(rows: Iterable[RunRow]) -> List[RunRow]:
    return sorted(
        rows,
        key=lambda r: (
            r.spectral_mode,
            r.concept_order,
            r.n_concepts,
            r.alpha,
            r.experiment,
            r.run_id,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize quick_proof results into one report.")
    parser.add_argument(
        "--results-glob",
        nargs="+",
        default=DEFAULT_RESULTS_GLOBS,
        help="Glob patterns for results.json files.",
    )
    parser.add_argument("--output-md", type=Path, default=Path("evaluation/summary.md"))
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation/summary.csv"))
    parser.add_argument("--plots-dir", type=Path, default=Path("evaluation/plots"))
    args = parser.parse_args()

    chosen_paths, dedupe_stats = discover_unique_results(args.results_glob)
    if not chosen_paths:
        raise SystemExit("No results.json files found. Pass --results-glob with valid paths.")

    rows = _sort_rows(_parse_row(path) for path in chosen_paths)
    write_csv(rows, args.output_csv)
    write_markdown(rows, dedupe_stats, args.output_md)
    plot_msg = create_plots(rows, args.plots_dir)

    print(f"Wrote: {args.output_md}")
    print(f"Wrote: {args.output_csv}")
    print(f"Plots: {plot_msg} ({args.plots_dir})")


if __name__ == "__main__":
    main()
