"""Locate and load model weights from safetensors."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_dir(model: str | None = None) -> Path:
    """Find a local directory holding config.json, tokenizer.json and *.safetensors.

    Search order: explicit path, $MINIVLLM_MODEL_DIR, ./models/<name>, then the
    Hugging Face cache (offline only; use scripts/download_model.py to fetch).
    """
    candidates: list[Path] = []
    if model and Path(model).is_dir():
        candidates.append(Path(model))
    if os.environ.get("MINIVLLM_MODEL_DIR"):
        candidates.append(Path(os.environ["MINIVLLM_MODEL_DIR"]))
    model_id = model if model and not Path(model).is_dir() else DEFAULT_MODEL_ID
    candidates.append(PROJECT_ROOT / "models" / model_id.split("/")[-1])
    for c in candidates:
        if (c / "config.json").exists():
            return c
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception as e:  # pragma: no cover - depends on local state
        raise FileNotFoundError(
            f"Could not find weights for {model_id}. Run `uv run python scripts/download_model.py`."
        ) from e


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        state.update(load_file(str(f)))
    if not state:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return state
