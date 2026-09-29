"""From scratch denoiser backbones: toy MLP, small U-Net, small DiT (patch 2).

Every network has the signature net(x_t, t, y) with t in (0, 1) as a float
tensor of shape [B] and y an int64 label tensor of shape [B] (or None for the
unconditional toy MLP). Class conditional nets reserve label `num_classes` as
the null label for classifier free guidance. Output layers are zero initialised
so a fresh network predicts 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * scale)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class CondEmbed(nn.Module):
    """Sinusoidal time embedding plus optional learned class embedding."""

    def __init__(self, dim: int, num_classes: int | None):
        super().__init__()
        self.freq_dim = min(dim, 256)
        self.mlp = nn.Sequential(nn.Linear(self.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.label = nn.Embedding(num_classes + 1, dim) if num_classes is not None else None

    def forward(self, t, y):
        e = self.mlp(timestep_embedding(t, self.freq_dim))
        if self.label is not None and y is not None:
            e = e + self.label(y)
        return e


# ---------------------------------------------------------------- toy MLP

class ToyMLP(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.embed = CondEmbed(hidden, None)
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, y=None):
        e = self.embed(t, None)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(F.silu(h + e))
        return self.out(F.silu(h))


# ---------------------------------------------------------------- U-Net

def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float):
        super().__init__()
        self.n1 = _gn(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, 2 * cout)  # scale and shift
        self.n2 = _gn(cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm = _gn(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    """DDPM style U-Net: ch_mult levels, `blocks` res blocks each, attention at `attn_res`."""

    def __init__(self, in_ch=3, ch=64, ch_mult=(1, 2, 2), blocks=2, attn_res=(8,), num_classes=10,
                 dropout=0.1, img=32):
        super().__init__()
        emb = ch * 4
        self.embed = CondEmbed(emb, num_classes)
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans = [ch]
        cur, res = ch, img
        for li, m in enumerate(ch_mult):
            for _ in range(blocks):
                self.down.append(nn.ModuleList([ResBlock(cur, ch * m, emb, dropout),
                                                AttnBlock(ch * m) if res in attn_res else nn.Identity()]))
                cur = ch * m
                chans.append(cur)
            if li != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(cur, cur, 3, stride=2, padding=1)]))
                chans.append(cur)
                res //= 2
        self.mid1 = ResBlock(cur, cur, emb, dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, emb, dropout)
        self.up = nn.ModuleList()
        for li, m in reversed(list(enumerate(ch_mult))):
            for _ in range(blocks + 1):
                self.up.append(nn.ModuleList([ResBlock(cur + chans.pop(), ch * m, emb, dropout),
                                              AttnBlock(ch * m) if res in attn_res else nn.Identity()]))
                cur = ch * m
            if li != 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(cur, cur, 3, padding=1)]))
                res *= 2
        self.out_norm = _gn(cur)
        self.out = nn.Conv2d(cur, in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, y=None):
        e = self.embed(t, y)
        h = self.inp(x)
        hs = [h]
        for mod in self.down:
            if len(mod) == 2:
                h = mod[1](mod[0](h, e))
            else:
                h = mod[0](h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for mod in self.up:
            if isinstance(mod[0], ResBlock):
                h = mod[1](mod[0](torch.cat([h, hs.pop()], dim=1), e))
            else:
                h = mod[1](mod[0](h))
        return self.out(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------- DiT

def _sincos_2d(dim: int, n: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding for an n x n grid, shape [n*n, dim]."""
    def one_d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d // 2))
        out = pos[:, None] * omega[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    gy, gx = torch.meshgrid(torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij")
    return torch.cat([one_d(dim // 2, gy.reshape(-1)), one_d(dim // 2, gx.reshape(-1))], dim=1).float()


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Linear(hid, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)  # adaLN-Zero: each block starts as identity
        nn.init.zeros_(self.ada.bias)
        self.heads = heads

    def forward(self, x, c):
        b, n, d = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(c)).chunk(6, dim=1)
        q, k, v = self.qkv(_modulate(self.n1(x), sh1, sc1)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + g1[:, None] * self.proj(a)
        x = x + g2[:, None] * self.mlp(_modulate(self.n2(x), sh2, sc2))
        return x


class DiT(nn.Module):
    def __init__(self, in_ch=3, img=32, patch=2, dim=256, depth=6, heads=4, num_classes=10):
        super().__init__()
        self.patch, self.in_ch, self.grid = patch, in_ch, img // patch
        self.embed_x = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.register_buffer("pos", _sincos_2d(dim, self.grid)[None], persistent=False)
        self.embed = CondEmbed(dim, num_classes)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, patch * patch * in_ch)
        for m in (self.ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, y=None):
        b = x.shape[0]
        h = self.embed_x(x).flatten(2).transpose(1, 2) + self.pos
        c = self.embed(t, y)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada(F.silu(c)).chunk(2, dim=1)
        h = self.out(_modulate(self.norm(h), shift, scale))  # b, n, p*p*c
        p, g = self.patch, self.grid
        h = h.reshape(b, g, g, p, p, self.in_ch).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(b, self.in_ch, g * p, g * p)


def build(arch: str, **kw) -> nn.Module:
    if arch == "unet":
        return UNet(**kw)
    if arch == "dit":
        return DiT(**kw)
    if arch == "mlp":
        return ToyMLP(**kw)
    raise ValueError(arch)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
