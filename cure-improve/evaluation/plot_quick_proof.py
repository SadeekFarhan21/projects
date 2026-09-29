#!/usr/bin/env python3
"""
Plot quick_proof benchmark outputs for poster-ready comparison.

Usage:
  python3 evaluation/plot_quick_proof.py \
    --results /path/to/run1/results.json /path/to/run2/results.json \
    --out-dir outputs/quick_proof_plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _extract_method_aggregate(result_block: Dict) -> Dict:
    """
    Support both new schema:
      results[method] = {"aggregate": ..., "per_seed": ...}
    and older schema:
      results[method] = {...}
    """
    if "aggregate" in result_block:
        return result_block["aggregate"]
    return result_block


def _metric_from_method(method_agg: Dict) -> Dict[str, float]:
    checkpoints = method_agg.get("checkpoints", [])
    if not checkpoints:
        raise ValueError("No checkpoints found in results payload.")
    final_cp = checkpoints[-1]

    # Backward compatible field extraction.
    target_mean = final_cp.get("target_clip_mean", final_cp.get("target_clip"))
    target_std = final_cp.get("target_clip_std", 0.0)
    drop_mean = final_cp.get(
        "retention_drop_vs_base_mean",
        final_cp.get("retention_drop_vs_base"),
    )
    drop_std = final_cp.get("retention_drop_vs_base_std", 0.0)
    delta_mean = method_agg.get(
        "single_concept_delta_mean",
        method_agg.get("single_concept_delta"),
    )
    delta_std = method_agg.get("single_concept_delta_std", 0.0)

    return {
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "drop_mean": float(drop_mean),
        "drop_std": float(drop_std),
        "single_delta_mean": float(delta_mean),
        "single_delta_std": float(delta_std),
    }


def _run_names(payload: Dict, path: Path) -> Tuple[str, str]:
    config = payload.get("config", {})
    spectral = config.get("spectral_mode", "unknown")
    order = config.get("concept_order", "unknown")
    run_id = path.parent.name
    return f"{spectral}/{order}", run_id


def load_runs(results_paths: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for path in results_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload["results"]
        label, run_id = _run_names(payload, path)

        cure_agg = _extract_method_aggregate(results["cure"])
        seq_agg = _extract_method_aggregate(results["cure_seq"])

        rows.append(
            {
                "path": str(path),
                "label": label,
                "run_id": run_id,
                "cure": _metric_from_method(cure_agg),
                "seq": _metric_from_method(seq_agg),
            }
        )
    return rows


def write_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_label",
                "run_id",
                "results_json",
                "method",
                "final_target_clip_mean",
                "final_target_clip_std",
                "final_retention_drop_mean",
                "final_retention_drop_std",
                "single_concept_delta_mean",
                "single_concept_delta_std",
            ]
        )
        for row in rows:
            for method in ("cure", "seq"):
                m = row[method]
                writer.writerow(
                    [
                        row["label"],
                        row["run_id"],
                        row["path"],
                        "CURE" if method == "cure" else "CURE-Sequential",
                        m["target_mean"],
                        m["target_std"],
                        m["drop_mean"],
                        m["drop_std"],
                        m["single_delta_mean"],
                        m["single_delta_std"],
                    ]
                )


def _grouped_values(rows: List[Dict], key_mean: str, key_std: str) -> Tuple[List, List, List, List, List]:
    labels = [r["label"] for r in rows]
    cure_vals = [r["cure"][key_mean] for r in rows]
    seq_vals = [r["seq"][key_mean] for r in rows]
    cure_err = [r["cure"][key_std] for r in rows]
    seq_err = [r["seq"][key_std] for r in rows]
    return labels, cure_vals, seq_vals, cure_err, seq_err


def create_plots(rows: List[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    width = 0.36
    x = np.arange(len(rows))

    panels = [
        ("target_mean", "target_std", "Final Target CLIP (lower better)", "final_target_clip.png"),
        ("drop_mean", "drop_std", "Final Retention Drop vs Base (lower better)", "final_retention_drop.png"),
        ("single_delta_mean", "single_delta_std", "Single-Concept Delta (after-before, lower better)", "single_delta.png"),
    ]

    for ax, (mean_key, std_key, title, _) in zip(axes, panels):
        labels, cure_vals, seq_vals, cure_err, seq_err = _grouped_values(rows, mean_key, std_key)
        ax.bar(x - width / 2, cure_vals, width, yerr=cure_err, label="CURE", color="#4C78A8", capsize=4)
        ax.bar(x + width / 2, seq_vals, width, yerr=seq_err, label="CURE-Sequential", color="#F58518", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="best")
    summary_path = out_dir / "quick_proof_summary_bars.png"
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    # Tradeoff scatter: lower target and lower retention drop are both better.
    fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for row in rows:
        for method, color, marker in (("cure", "#4C78A8", "o"), ("seq", "#F58518", "s")):
            m = row[method]
            name = "CURE" if method == "cure" else "Seq"
            ax2.scatter(m["target_mean"], m["drop_mean"], c=color, marker=marker, s=90)
            ax2.annotate(
                f"{name}: {row['label'].splitlines()[0]}",
                (m["target_mean"], m["drop_mean"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax2.set_title("Final Tradeoff: Suppression vs Retention Damage")
    ax2.set_xlabel("Final Target CLIP (lower better)")
    ax2.set_ylabel("Final Retention Drop vs Base (lower better)")
    ax2.grid(alpha=0.25)
    scatter_path = out_dir / "quick_proof_tradeoff_scatter.png"
    fig2.savefig(scatter_path, dpi=200)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot quick_proof result JSON files.")
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to quick_proof results.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/quick_proof_plots"),
        help="Output directory for charts and CSV summary.",
    )
    args = parser.parse_args()

    for path in args.results:
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")

    rows = load_runs(args.results)
    write_summary_csv(rows, args.out_dir / "quick_proof_summary.csv")
    create_plots(rows, args.out_dir)

    print(f"Wrote: {args.out_dir / 'quick_proof_summary.csv'}")
    print(f"Wrote: {args.out_dir / 'quick_proof_summary_bars.png'}")
    print(f"Wrote: {args.out_dir / 'quick_proof_tradeoff_scatter.png'}")


if __name__ == "__main__":
    main()
