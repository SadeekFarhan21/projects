"""Sample stories from a trained checkpoint.

Usage: uv run python -m tfs.generate --ckpt runs/v0/ckpt.pt --prompt "Once upon a time" \
           --temperature 0.8 --top-k 50 --top-p 0.95 --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .bpe import BPETokenizer
from .model import ModelConfig, Transformer
from .sampling import SamplingConfig, generate
from .train import pick_device, sync


def load_model(ckpt_path: str | Path, device: torch.device, **overrides) -> Transformer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**{**ckpt["config"], **overrides})
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/v0/ckpt.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)
    prompt = torch.tensor([tok.encode(args.prompt)], device=device)
    budget = min(args.max_new_tokens, model.cfg.max_seq_len - prompt.shape[1])
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    gen = torch.Generator().manual_seed(args.seed)
    for i in range(args.num_samples):
        t0 = time.perf_counter()
        out = generate(model, prompt, budget, sampling, use_cache=not args.no_cache,
                       generator=gen, stop_id=tok.eot_id)
        sync(device)
        dt = time.perf_counter() - t0
        n_new = out.shape[1] - prompt.shape[1]
        text = tok.decode(out[0].tolist()).replace("<|endoftext|>", "")
        print(f"--- sample {i} ({n_new} tokens, {n_new / dt:.0f} tok/s)\n{text}\n")


if __name__ == "__main__":
    main()
