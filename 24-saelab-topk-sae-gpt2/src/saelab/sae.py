"""Sparse autoencoders written from scratch.

Two architectures share one module:

  relu  f = ReLU(W_enc (x_n - b_dec) + b_enc), loss = ||x_n - x_hat_n||^2 + lam * sum(f)
  topk  f = TopK_k(ReLU(W_enc (x_n - b_dec) + b_enc)), loss = ||x_n - x_hat_n||^2 + alpha * aux

x_n = x / input_scale is the residual activation divided by one scalar per
layer, chosen so that E||x_n||^2 = d_model. The scale lives in the module, so
callers always pass and receive raw residual activations.

The decoder rows W_dec[i] are kept at unit norm (the "decoder norm
constraint"). With unit rows the L1 penalty on f is the same as the
Anthropic-style penalty sum_i f_i ||W_dec[i]||, so the SAE cannot shrink the
penalty by shrinking f and growing the decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass
class SAEConfig:
    d_in: int = 768
    d_sae: int = 768 * 16
    arch: str = "topk"          # "topk" or "relu"
    k: int = 32                 # TopK only
    l1_coeff: float = 1.0       # ReLU only, on the normalized scale
    k_aux: int = 256            # TopK aux loss: how many dead latents to use
    aux_coeff: float = 1 / 32   # TopK aux loss weight (Gao et al. 2024)
    dead_tokens: int = 2_000_000  # a latent is "dead" if silent this many tokens
    hook: str = "blocks.8.hook_resid_pre"
    seed: int = 0

    def name(self) -> str:
        layer = self.hook.split(".")[1]
        if self.arch == "topk":
            return f"L{layer}_topk_k{self.k}_s{self.seed}"
        return f"L{layer}_relu_l1{self.l1_coeff:g}_s{self.seed}"


class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        w = torch.randn(cfg.d_sae, cfg.d_in, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w.clone())                 # [d_sae, d_in], unit rows
        # tied init. ReLU SAEs get a 0.1 encoder scale: with 12,288 unit rows and
        # a full-scale encoder about half the latents fire at step 0 and the
        # reconstruction is ~50x too large. TopK only keeps k latents, so it
        # starts fine at full scale.
        enc_scale = 1.0 if cfg.arch == "topk" else 0.1
        self.W_enc = nn.Parameter(w.t().clone() * enc_scale)  # [d_in, d_sae]
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("input_scale", torch.tensor(1.0))
        # tokens since each latent last fired, for the dead-latent aux loss
        self.register_buffer("since_fired", torch.zeros(cfg.d_sae, dtype=torch.long))

    # ------------------------------------------------------------------ core
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        xn = x / self.input_scale
        return (xn - self.b_dec) @ self.W_enc + self.b_enc

    def activate(self, pre: torch.Tensor) -> torch.Tensor:
        if self.cfg.arch == "relu":
            return F.relu(pre)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        out = torch.zeros_like(pre)
        return out.scatter_(-1, idx, F.relu(vals))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw residual [..., d_in] -> latent activations [..., d_sae]."""
        return self.activate(self.pre_acts(x))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Latents [..., d_sae] -> raw-scale reconstruction [..., d_in]."""
        return (f @ self.W_dec + self.b_dec) * self.input_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # -------------------------------------------------------------- training
    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training loss on a batch of raw activations [B, d_in]."""
        xn = x / self.input_scale
        pre = (xn - self.b_dec) @ self.W_enc + self.b_enc
        if self.cfg.arch == "topk":
            # sparse decode: only the k active rows of W_dec are touched
            vals, idx = pre.topk(self.cfg.k, dim=-1)
            vals = F.relu(vals)
            xhat = F.embedding_bag(idx, self.W_dec, per_sample_weights=vals, mode="sum") + self.b_dec
            f = torch.zeros_like(pre).scatter_(-1, idx, vals)
        else:
            f = F.relu(pre)
            xhat = f @ self.W_dec + self.b_dec
        err = xn - xhat
        mse = err.pow(2).sum(-1).mean()
        out = {"mse": mse}
        fired = (f > 0).any(0)
        with torch.no_grad():
            self.since_fired += x.shape[0]
            self.since_fired[fired] = 0
        if self.cfg.arch == "relu":
            l1 = f.sum(-1).mean()
            out["l1"] = l1
            out["loss"] = mse + self.cfg.l1_coeff * l1
        else:
            aux = self.aux_loss(pre, err)
            out["aux"] = aux
            out["loss"] = mse + self.cfg.aux_coeff * aux
        with torch.no_grad():
            out["l0"] = (f > 0).float().sum(-1).mean()
            out["fvu"] = err.pow(2).sum() / (xn - xn.mean(0)).pow(2).sum()
        return out

    def aux_loss(self, pre: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """AuxK loss: let the top k_aux dead latents model the residual error.

        This gives dead latents a gradient so they can come back to life.
        Returned as a normalized MSE (1.0 means the aux reconstruction is no
        better than predicting zero error).
        """
        dead = self.since_fired >= self.cfg.dead_tokens
        n_dead = int(dead.sum())
        if n_dead == 0:
            return pre.new_zeros(())
        k_aux = min(self.cfg.k_aux, n_dead)
        masked = pre.masked_fill(~dead, float("-inf"))
        vals, idx = masked.topk(k_aux, dim=-1)
        f_aux = torch.zeros_like(pre).scatter_(-1, idx, F.relu(vals))
        e_hat = f_aux @ self.W_dec
        target = err.detach()
        return (target - e_hat).pow(2).sum() / target.pow(2).sum().clamp_min(1e-8)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Project out the gradient component that would change row norms."""
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data
        g = self.W_dec.grad
        g -= (g * w).sum(1, keepdim=True) * w

    # ----------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(tensors, str(path / "sae.safetensors"))
        (path / "cfg.json").write_text(json.dumps(asdict(self.cfg), indent=1))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "SAE":
        cfg = SAEConfig(**json.loads((path / "cfg.json").read_text()))
        sae = cls(cfg)
        sae.load_state_dict(load_file(str(path / "sae.safetensors")))
        return sae.to(device)


def geometric_median(x: torch.Tensor, iters: int = 100, eps: float = 1e-6) -> torch.Tensor:
    """Weiszfeld's algorithm: the point minimizing the sum of L2 distances."""
    m = x.mean(0)
    for _ in range(iters):
        d = (x - m).norm(dim=1).clamp_min(eps)
        w = 1.0 / d
        m_new = (w[:, None] * x).sum(0) / w.sum()
        if (m_new - m).norm() < eps * (1 + m.norm()):
            m = m_new
            break
        m = m_new
    return m


@torch.no_grad()
def init_from_sample(sae: SAE, sample: torch.Tensor) -> None:
    """Set input_scale and b_dec (geometric median) from raw activations."""
    scale = (sample.pow(2).sum(-1).mean() / sample.shape[-1]).sqrt()
    sae.input_scale.fill_(float(scale))
    sae.b_dec.data.copy_(geometric_median(sample / scale).to(sae.b_dec))
