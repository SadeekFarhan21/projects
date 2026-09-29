"""Download everything this project needs into data/ (gitignored).

  uv run python scripts/download.py            # text shard + public SAEs + explainer
  uv run python scripts/download.py --no-mlx   # skip the explainer model

Sizes: one OpenWebText parquet shard (~300 MB), two SAELens-format
"gpt2-small-res-jb" SAEs (~150 MB each), two OpenAI v5 32k TopK SAEs
reformatted for SAELens (~200 MB each), and a 4-bit MLX Qwen2.5 1.5B
Instruct explainer (~900 MB, stored in the Hugging Face cache).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

OWT_REPO = "Skylion007/openwebtext"
OWT_REV = "refs/convert/parquet"
OWT_SHARD = "plain_text/train/0000.parquet"

JB_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
OAI_REPO = "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs"
EXPLAINER = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Our layers are resid_pre 5 and resid_pre 8, which equal resid_post 4 and 7.
JB_HOOKS = ["blocks.5.hook_resid_pre", "blocks.8.hook_resid_pre"]
OAI_LAYERS = [4, 7]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mlx", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    p = hf_hub_download(OWT_REPO, OWT_SHARD, repo_type="dataset", revision=OWT_REV,
                        local_dir=DATA / "owt")
    print("text shard", p)
    for hook in JB_HOOKS:
        for f in ["cfg.json", "sae_weights.safetensors"]:
            print(hf_hub_download(JB_REPO, f"{hook}/{f}", local_dir=DATA / "public" / "jb"))
    for layer in OAI_LAYERS:
        for f in ["cfg.json", "sae_weights.safetensors"]:
            print(hf_hub_download(OAI_REPO, f"v5_32k_layer_{layer}.pt/{f}",
                                  local_dir=DATA / "public" / "oai"))
    if not args.no_mlx:
        print(snapshot_download(EXPLAINER))


if __name__ == "__main__":
    main()
