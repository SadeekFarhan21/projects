"""Gaussian scene parameters and the full differentiable render call."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import projection as P
from .rasterize import rasterize


@dataclass
class Cam:
    viewmat: torch.Tensor  # [4, 4] world -> camera
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class Gaussians(torch.nn.Module):
    """N anisotropic Gaussians with degree-0 (view independent) colour.

    Stored in unconstrained form:
      means      [N, 3]  world position
      log_scales [N, 3]  exp -> per-axis std dev
      quats      [N, 4]  (w, x, y, z), normalised on use
      logit_opa  [N]     sigmoid -> opacity
      sh0        [N, 3]  degree-0 SH coefficient, rgb = 0.5 + C0 * sh0
    """

    def __init__(self, means, log_scales, quats, logit_opa, sh0):
        super().__init__()
        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(log_scales)
        self.quats = torch.nn.Parameter(quats)
        self.logit_opa = torch.nn.Parameter(logit_opa)
        self.sh0 = torch.nn.Parameter(sh0)

    @property
    def n(self) -> int:
        return self.means.shape[0]

    @classmethod
    def from_points(cls, xyz: np.ndarray, rgb: np.ndarray, init_opacity: float = 0.1, dtype=torch.float32):
        """3DGS initialisation: isotropic scale = mean distance to 3 nearest neighbours."""
        tree = cKDTree(xyz)
        dist, _ = tree.query(xyz, k=4)
        d = np.sqrt(np.mean(dist[:, 1:] ** 2, axis=1)).clip(1e-7)
        n = len(xyz)
        means = torch.tensor(xyz, dtype=dtype)
        log_scales = torch.log(torch.tensor(d, dtype=dtype))[:, None].repeat(1, 3)
        quats = torch.zeros(n, 4, dtype=dtype)
        quats[:, 0] = 1
        logit = torch.full((n,), float(np.log(init_opacity / (1 - init_opacity))), dtype=dtype)
        sh0 = P.rgb_to_sh0(torch.tensor(rgb, dtype=dtype) / 255.0)
        return cls(means, log_scales, quats, logit, sh0)

    def opacities(self):
        return torch.sigmoid(self.logit_opa)

    def colors(self):
        return P.sh0_to_rgb(self.sh0)


def render(g: Gaussians, cam: Cam, rect=None, background=None, chunk=64, tile_batch=None, return_stats=False):
    cov3d = P.covariance_3d(g.log_scales, g.quats)
    means2d, cov2d, depths, in_front = P.project_gaussians(
        g.means, cov3d, cam.viewmat, cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height
    )
    opa = g.opacities()
    conics = P.conic_from_cov2d(cov2d)
    ext = P.screen_extents(cov2d, opa)
    return rasterize(
        means2d, conics, g.colors(), opa, depths, ext, in_front,
        cam.width, cam.height, rect=rect, background=background,
        chunk=chunk, tile_batch=tile_batch, return_stats=return_stats,
    )
