"""Scene loading: COLMAP model + images at a chosen downscale, train/test split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from .colmap import read_model
from .model import Cam


@dataclass
class View:
    name: str
    cam: Cam
    image: torch.Tensor  # [H, W, 3] float in [0, 1]


@dataclass
class Scene:
    train: list[View]
    test: list[View]
    xyz: np.ndarray
    rgb: np.ndarray
    extent: float  # radius of the camera centres (3DGS "cameras_extent")
    center: np.ndarray


def _image_dir(root: Path, downscale: int) -> Path:
    """Use images_<d> if present; otherwise build it from images/ once and cache."""
    d = root / (f"images_{downscale}" if downscale > 1 else "images")
    if d.exists():
        return d
    src = root / "images"
    if not src.exists():
        raise FileNotFoundError(f"neither {d} nor {src} exists")
    d.mkdir()
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        im = PILImage.open(p)
        w, h = im.size
        im.resize((round(w / downscale), round(h / downscale)), PILImage.LANCZOS).save(d / p.name, quality=95)
    return d


def load_scene(root: str | Path, downscale: int = 4, holdout_every: int = 8, device="cpu", max_images: int | None = None) -> Scene:
    root = Path(root)
    cams, imgs, (xyz, rgb) = read_model(root / "sparse" / "0")
    img_dir = _image_dir(root, downscale)
    views = []
    for im in sorted(imgs.values(), key=lambda i: i.name):
        c = cams[im.camera_id]
        path = img_dir / im.name
        if not path.exists():
            continue
        pil = PILImage.open(path).convert("RGB")
        W, H = pil.size
        fx, fy, cx, cy = c.pinhole()
        sx, sy = W / c.width, H / c.height
        cam = Cam(
            viewmat=torch.tensor(im.world_to_cam(), dtype=torch.float32, device=device),
            fx=fx * sx, fy=fy * sy, cx=cx * sx, cy=cy * sy, width=W, height=H,
        )
        t = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).to(device)
        views.append(View(im.name, cam, t))
    if max_images:
        views = views[:max_images]
    test = [v for i, v in enumerate(views) if i % holdout_every == 0]
    train = [v for i, v in enumerate(views) if i % holdout_every != 0]
    centers = np.stack([imgs_c for imgs_c in (v_center(v) for v in views)])
    center = centers.mean(0)
    extent = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    return Scene(train, test, xyz, rgb, extent, center)


def v_center(v: View) -> np.ndarray:
    M = v.cam.viewmat.cpu().double().numpy()
    return -M[:3, :3].T @ M[:3, 3]
