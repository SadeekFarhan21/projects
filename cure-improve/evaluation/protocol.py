"""
Shared evaluation protocol and result schema helpers.

This module defines:
1) Standard concept sets for cross-branch comparisons
2) Alpha assignment rules
3) Run/result schema helpers for reproducible reporting
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cure.utils import get_default_forget_prompts


SCHEMA_VERSION = "cure-shared-eval-v1"

DEFAULT_CONCEPT_SETS: Dict[str, List[str]] = {
    # Matches CURE object benchmark classes
    "objects10": [
        "cassette player",
        "chain saw",
        "church",
        "english springer",
        "french horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
        "tench",
    ],
    "artists2": ["van gogh", "kelly mckernan"],
    "identities4": ["emma stone", "taylor swift", "elon musk", "jennifer lawrence"],
    "nsfw1": ["nudity"],
}

NSFW_CONCEPTS = {"nudity", "nude", "naked", "explicit", "nsfw"}


@dataclass
class ConceptSpec:
    concept: str
    forget_prompts: List[str]
    retain_prompts: List[str]
    alpha: float


@dataclass
class EvalConfig:
    branch: str
    model_id: str
    device: str
    embedding_mode: str
    concept_set: str
    concepts: List[str]
    erasure_mode: str
    samples_per_concept: int
    steps: int
    guidance_scale: float
    seed: int
    alpha_object: float
    alpha_nsfw: float
    output_dir: str
    run_id: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id(branch: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{branch}"


def resolve_concepts(
    concept_set: str,
    concepts_csv: Optional[str] = None,
    max_concepts: Optional[int] = None,
) -> List[str]:
    if concepts_csv:
        concepts = [c.strip() for c in concepts_csv.split(",") if c.strip()]
    else:
        if concept_set not in DEFAULT_CONCEPT_SETS:
            raise ValueError(
                f"Unknown concept_set '{concept_set}'. "
                f"Choose from {list(DEFAULT_CONCEPT_SETS.keys())} or pass --concepts."
            )
        concepts = list(DEFAULT_CONCEPT_SETS[concept_set])

    if max_concepts is not None:
        concepts = concepts[:max_concepts]

    if not concepts:
        raise ValueError("No concepts selected.")

    return concepts


def alpha_for_concept(concept: str, alpha_object: float, alpha_nsfw: float) -> float:
    normalized = concept.strip().lower()
    if normalized in NSFW_CONCEPTS:
        return alpha_nsfw
    return alpha_object


def build_concept_specs(
    concepts: List[str],
    alpha_object: float,
    alpha_nsfw: float,
) -> List[ConceptSpec]:
    specs: List[ConceptSpec] = []
    for concept in concepts:
        specs.append(
            ConceptSpec(
                concept=concept,
                forget_prompts=get_default_forget_prompts(concept),
                retain_prompts=[],
                alpha=alpha_for_concept(concept, alpha_object=alpha_object, alpha_nsfw=alpha_nsfw),
            )
        )
    return specs


def to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        # torch/numpy scalar
        return value.item()
    return str(value)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, ensure_ascii=True)


def config_to_dict(config: EvalConfig) -> Dict[str, Any]:
    return asdict(config)
