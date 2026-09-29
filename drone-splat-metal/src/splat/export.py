"""Export trained Gaussians.

``.splat`` layout (the de facto format of small web viewers), 32 bytes per Gaussian,
little endian, sorted by opacity * volume, largest first:

  offset  type        field
  0       float32[3]  position (world)
  12      float32[3]  scale (std dev, already exp'd)
  24      uint8[4]    r, g, b, a   (a = opacity)
  28      uint8[4]    quaternion w, x, y, z mapped from [-1, 1] to [0, 255]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import Gaussians

RECORD = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
assert RECORD.itemsize == 32


@torch.no_grad()
def to_records(g: Gaussians, min_opacity: float = 1 / 255) -> np.ndarray:
    pos = g.means.detach().float().cpu().numpy()
    scale = torch.exp(g.log_scales).detach().float().cpu().numpy()
    rgb = g.colors().clamp(0, 1).detach().float().cpu().numpy()
    a = g.opacities().detach().float().cpu().numpy()
    q = torch.nn.functional.normalize(g.quats.detach().float(), dim=-1).cpu().numpy()
    keep = a >= min_opacity
    order = np.argsort(-(a * scale.prod(1))[keep])
    rec = np.zeros(int(keep.sum()), RECORD)
    rec["pos"] = pos[keep][order]
    rec["scale"] = scale[keep][order]
    rec["rgba"][:, :3] = np.round(rgb[keep][order] * 255)
    rec["rgba"][:, 3] = np.round(a[keep][order] * 255)
    rec["rot"] = np.clip(np.round(q[keep][order] * 128 + 128), 0, 255)
    return rec


def write_splat(g: Gaussians, path: Path) -> int:
    rec = to_records(g)
    Path(path).write_bytes(rec.tobytes())
    return len(rec)


def read_splat(path: Path) -> np.ndarray:
    return np.frombuffer(Path(path).read_bytes(), RECORD)


def save_checkpoint(g: Gaussians, path: Path) -> None:
    torch.save({k: v.detach().cpu() for k, v in g.state_dict().items()}, path)


def load_checkpoint(path: Path, device="cpu") -> Gaussians:
    sd = torch.load(path, map_location=device)
    return Gaussians(sd["means"], sd["log_scales"], sd["quats"], sd["logit_opa"], sd["sh0"])


def write_cameras_json(scene, path: Path) -> None:
    """Test cameras for the viewer (world-to-camera matrices, pinhole intrinsics)."""
    cams = []
    test_ids = {id(v) for v in scene.test}
    for v in scene.test + scene.train:
        c = v.cam
        cams.append({
            "name": v.name,
            "split": "test" if id(v) in test_ids else "train",
            "viewmat": c.viewmat.cpu().double().numpy().tolist(),
            "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
            "width": c.width, "height": c.height,
        })
    Path(path).write_text(json.dumps({"center": scene.center.tolist(), "extent": scene.extent, "cameras": cams}))
