"""Loaders for public GPT-2 small residual SAEs, wrapped to our interface.

  jb   "gpt2-small-res-jb" (Joseph Bloom, SAELens): ReLU + L1, 24,576 latents,
       resid_pre. Trained through TransformerLens with its default weight
       processing, i.e. on a residual stream with centered writing weights.
  oai  OpenAI "v5 32k" TopK (k=32), 32,768 latents, resid_post of layer L-1
       (the same tensor as our resid_pre L). Inputs are layer-normalized
       ((x - mean) / std per token) before encoding and un-normalized after.

Both expose .encode(raw_x) -> latents and .decode(latents) -> raw-scale x_hat
(the per-token normalization statistics are cached by encode for the
matching decode call, which is how SAELens handles it too).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "data" / "public"


class PublicSAE(nn.Module):
    def __init__(self, W_enc, b_enc, W_dec, b_dec, arch: str, k: int | None,
                 layernorm: bool, center: bool, name: str):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.arch, self.k, self.layernorm, self.center, self.name = arch, k, layernorm, center, name
        self._stats = None

    def _pre(self, x):
        mu = x.mean(-1, keepdim=True)
        if self.layernorm:
            # LN already removes the mean, so centering is implied here
            std = (x - mu).std(-1, keepdim=True) + 1e-5
            self._stats = ("ln", mu, std)
            return (x - mu) / std
        if self.center:
            # centered-writing-weights residual = raw residual minus its mean over d_model
            self._stats = ("center", mu)
            return x - mu
        self._stats = None
        return x

    def pre_acts(self, x):
        return (self._pre(x) - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        pre = self.pre_acts(x)
        if self.arch == "relu":
            return torch.relu(pre)
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))

    def decode(self, f):
        y = f @ self.W_dec + self.b_dec
        st = self._stats
        if st is not None and st[0] == "ln":
            y = y * st[2] + st[1]
        elif st is not None and st[0] == "center":
            y = y + st[1]
        return y

    def forward(self, x):
        return self.decode(self.encode(x))


def load_public(kind: str, layer: int, device: str = "cpu", center: bool | None = None) -> PublicSAE:
    """Load a public SAE for resid_pre `layer`. `center` overrides the default convention."""
    if kind == "jb":
        d = PUBLIC / "jb" / f"blocks.{layer}.hook_resid_pre"
        w = load_file(str(d / "sae_weights.safetensors"))
        sae = PublicSAE(w["W_enc"], w["b_enc"], w["W_dec"], w["b_dec"], "relu", None,
                        layernorm=False, center=True if center is None else center,
                        name=f"jb_L{layer}")
    elif kind == "oai":
        d = PUBLIC / "oai" / f"v5_32k_layer_{layer - 1}.pt"
        cfg = json.loads((d / "cfg.json").read_text())
        w = load_file(str(d / "sae_weights.safetensors"))
        sae = PublicSAE(w["W_enc"], w["b_enc"], w["W_dec"], w["b_dec"], "topk",
                        cfg["activation_fn_kwargs"]["k"], layernorm=True,
                        center=False if center is None else center, name=f"oai_L{layer}")
    else:
        raise ValueError(kind)
    return sae.to(device)
