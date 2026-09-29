"""GPT-2 small loading and residual-stream harvesting.

We load with fold_ln=True but center_writing_weights=False, so the residual
stream holds the same values as the original Hugging Face model. That matters
for comparing against public SAEs: the OpenAI TopK SAEs were trained on raw
GPT-2 activations. (center_writing_weights would subtract each token's mean
over d_model from the residual stream; see public.py for how we test which
convention each public SAE expects.)

Head indices follow ../tiny-circuits, where the default TransformerLens
processing was used. Weight processing does not change which head is which.
"""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

SEED = 1337
LAYERS = (5, 8)  # residual stream *before* block 5 and before block 8


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str | None = None) -> HookedTransformer:
    device = device or get_device()
    torch.manual_seed(SEED)
    model = HookedTransformer.from_pretrained(
        "gpt2", device=device, fold_ln=True, center_writing_weights=False,
        center_unembed=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def harvest(model: HookedTransformer, tokens: torch.Tensor,
            layers: tuple[int, ...] = LAYERS) -> dict[int, torch.Tensor]:
    """Residual activations [batch, pos, d_model] at resid_pre of each layer.

    Runs only blocks 0..max(layers)-1: resid_pre of layer L equals the
    residual after block L-1, so the forward stops early and the deepest
    layer is taken from the returned residual.
    """
    top = max(layers)
    acts: dict[int, torch.Tensor] = {}

    def save(layer):
        def fn(x, hook):
            acts[layer] = x.detach()
        return fn

    hooks = [(f"blocks.{L - 1}.hook_resid_post", save(L)) for L in layers if L != top]
    resid = model.run_with_hooks(tokens.to(model.cfg.device), stop_at_layer=top,
                                 fwd_hooks=hooks)
    acts[top] = resid.detach()
    return acts
