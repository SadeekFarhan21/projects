"""Small residual policy/value network for Connect Four.

Input: (B, 3, 6, 7) planes from c4core (current player stones, opponent
stones, first-player-to-move flag). Output: policy logits (B, 7) and a value
in [-1, 1] from the side to move's view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 5
    value_hidden: int = 64


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class C4Net(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetConfig()
        ch = self.cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(3, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.blocks = nn.ModuleList(ResBlock(ch) for _ in range(self.cfg.blocks))
        self.p_conv = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.Linear(2 * 42, 7)
        self.v_conv = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc1 = nn.Linear(42, self.cfg.value_hidden)
        self.v_fc2 = nn.Linear(self.cfg.value_hidden, 1)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = self.stem(x)
        hidden = [h]
        for b in self.blocks:
            h = b(h)
            hidden.append(h)
        p = self.p_fc(self.p_conv(h).flatten(1))
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(self.v_conv(h).flatten(1))))).squeeze(1)
        if return_hidden:
            return p, v, hidden
        return p, v


def save_checkpoint(path, net: C4Net, extra: dict | None = None) -> None:
    torch.save({"config": asdict(net.cfg), "state_dict": net.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device: str = "cpu") -> C4Net:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = C4Net(NetConfig(**ck["config"]))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    return "mps" if torch.backends.mps.is_available() else "cpu"
