"""Model and engine configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """The subset of a Qwen2 `config.json` that the model code needs."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: int
    max_position_embeddings: int = 32768

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            eos_token_id=raw["eos_token_id"],
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
        )


@dataclass
class EngineConfig:
    """Knobs for the serving engine.

    block_size:            tokens per KV cache block (page).
    num_blocks:            total KV blocks in the pool; this is the memory budget.
    max_num_seqs:          max sequences in the running batch. 1 gives naive serving.
    max_num_batched_tokens: prefill token budget per step.
    max_model_len:         max prompt + output tokens for any single sequence.
    """

    block_size: int = 16
    num_blocks: int = 2048
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    max_model_len: int = 2048
