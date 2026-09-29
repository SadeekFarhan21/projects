"""3D Gaussian -> 2D screen-space Gaussian (EWA splatting, as in Kerbl et al. 2023).

Everything is plain differentiable PyTorch; autograd supplies the backward pass.

Conventions
-----------
* COLMAP / OpenCV camera: +x right, +y down, +z forward.
* ``viewmat`` is the 4x4 world-to-camera matrix.
* Pixel (i, j) has its centre at (i + 0.5, j + 0.5), so a camera-space point
  (x, y, z) lands at u = fx * x / z + cx, v = fy * y / z + cy.
"""
from __future__ import annotations

import math

import torch

SH_C0 = 0.28209479177387814
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
COV2D_DILATION = 0.3  # low-pass filter added to the 2D covariance diagonal (pixels^2)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: [N, 4] as (w, x, y, z), not necessarily normalised -> [N, 3, 3]."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def covariance_3d(log_scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """Sigma = R S S^T R^T with S = diag(exp(log_scales)). Returns [N, 3, 3]."""
    R = quat_to_rotmat(quats)
    M = R * torch.exp(log_scales)[..., None, :]  # R @ diag(s)
    return M @ M.transpose(-1, -2)


def project_gaussians(
    means: torch.Tensor,
    cov3d: torch.Tensor,
    viewmat: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float = 0.2,
    fov_clamp: float = 1.3,
    dilation: float = COV2D_DILATION,
):
    """Project N Gaussians.

    Returns
    -------
    means2d : [N, 2] pixel coordinates
    cov2d   : [N, 2, 2] screen-space covariance (with low-pass dilation)
    depths  : [N] camera-space z
    in_front: [N] bool, z > near
    """
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    p = means @ R.T + t  # camera space
    x, y, z = p.unbind(-1)
    in_front = z > near
    zs = torch.where(in_front, z, torch.full_like(z, near))  # keep culled rows finite

    u = fx * x / zs + cx
    v = fy * y / zs + cy
    means2d = torch.stack([u, v], -1)

    # Jacobian of the perspective map evaluated at a clamped x/z, y/z so that
    # Gaussians far outside the frustum do not produce huge footprints.
    lim_x = fov_clamp * (0.5 * width / fx)
    lim_y = fov_clamp * (0.5 * height / fy)
    tx = zs * torch.clamp(x / zs, -lim_x, lim_x)
    ty = zs * torch.clamp(y / zs, -lim_y, lim_y)
    zero = torch.zeros_like(zs)
    J = torch.stack(
        [
            fx / zs, zero, -fx * tx / (zs * zs),
            zero, fy / zs, -fy * ty / (zs * zs),
        ],
        -1,
    ).reshape(-1, 2, 3)
    T = J @ R  # [N, 2, 3]
    cov2d = T @ cov3d @ T.transpose(-1, -2)
    eye = torch.eye(2, dtype=cov2d.dtype, device=cov2d.device)
    cov2d = cov2d + dilation * eye
    return means2d, cov2d, z, in_front


def conic_from_cov2d(cov2d: torch.Tensor) -> torch.Tensor:
    """Inverse of a symmetric 2x2 as (a, b, c) meaning [[a, b], [b, c]]. [N, 3]."""
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-12)
    return torch.stack([c / det, -b / det, a / det], -1)


@torch.no_grad()
def screen_extents(cov2d: torch.Tensor, opacities: torch.Tensor) -> torch.Tensor:
    """Exact per-axis half extent (pixels) of the region where alpha >= 1/255.

    alpha = o * exp(-q / 2) >= 1/255  <=>  q <= 2 ln(255 o) =: k.
    The ellipse {d : d^T Sigma^-1 d <= k} has x half-width sqrt(k Sigma_xx) and
    y half-width sqrt(k Sigma_yy). Gaussians with o < 1/255 never contribute
    and get a negative extent (culled by the binner).
    """
    k = 2.0 * torch.log((opacities.clamp_min(1e-12) * 255.0))
    ex = torch.sqrt(k.clamp_min(0) * cov2d[:, 0, 0])
    ey = torch.sqrt(k.clamp_min(0) * cov2d[:, 1, 1])
    ext = torch.stack([ex, ey], -1)
    return torch.where((k > 0)[:, None], ext, torch.full_like(ext, -1.0))


def sh0_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    return (0.5 + SH_C0 * dc).clamp_min(0.0)


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def fov_from_focal(f: float, size: int) -> float:
    return 2 * math.atan(size / (2 * f))
