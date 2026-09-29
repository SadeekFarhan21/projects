"""Projection math checked against finite differences."""
import math

import numpy as np
import torch

from splat import projection as P

torch.set_default_dtype(torch.float64)


def random_view(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 4, generator=g)
    R = P.quat_to_rotmat(q)[0]
    V = torch.eye(4)
    V[:3, :3] = R
    V[:3, 3] = torch.tensor([0.1, -0.2, 4.0])
    return V


def scene(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    V = random_view(seed)
    # put means in front of the camera: sample in camera space, map to world
    pc = torch.randn(n, 3, generator=g) * 0.3
    pc[:, 2] = pc[:, 2].abs() + 2.0
    R, t = V[:3, :3], V[:3, 3]
    means = (pc - t) @ R  # R^T (pc - t)
    log_s = torch.randn(n, 3, generator=g) * 0.3 - 2.0
    quats = torch.randn(n, 4, generator=g)
    return means, log_s, quats, V


INTR = dict(fx=80.0, fy=75.0, cx=33.0, cy=29.0, width=64, height=60)


def test_quat_rotmat_orthonormal():
    q = torch.randn(50, 4)
    R = P.quat_to_rotmat(q)
    I = R @ R.transpose(-1, -2)
    assert torch.allclose(I, torch.eye(3).expand_as(I), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(50), atol=1e-12)


def test_covariance_is_spd_and_matches_scales():
    means, log_s, quats, _ = scene()
    S = P.covariance_3d(log_s, quats)
    ev = torch.linalg.eigvalsh(S)
    expect = torch.sort(torch.exp(2 * log_s), dim=-1).values
    assert torch.allclose(ev, expect, rtol=1e-10)


def test_projection_jacobian_matches_finite_difference_of_pinhole():
    """cov2d must equal J_fd Sigma J_fd^T + dilation, J_fd the numerical Jacobian
    of the world->pixel map at the mean."""
    means, log_s, quats, V = scene()
    cov3d = P.covariance_3d(log_s, quats)
    m2d, cov2d, _, _ = P.project_gaussians(means, cov3d, V, **INTR, dilation=0.0)

    def pix(p):
        pc = p @ V[:3, :3].T + V[:3, 3]
        return torch.stack([INTR["fx"] * pc[..., 0] / pc[..., 2] + INTR["cx"],
                            INTR["fy"] * pc[..., 1] / pc[..., 2] + INTR["cy"]], -1)

    h = 1e-6
    for i in range(means.shape[0]):
        J = torch.zeros(2, 3)
        for k in range(3):
            e = torch.zeros(3)
            e[k] = h
            J[:, k] = (pix(means[i] + e) - pix(means[i] - e)) / (2 * h)
        expect = J @ cov3d[i] @ J.T
        assert torch.allclose(cov2d[i], expect, rtol=1e-6, atol=1e-9), (cov2d[i], expect)
        assert torch.allclose(m2d[i], pix(means[i]), atol=1e-10)


def test_projection_gradcheck():
    """Autograd backward of the full projection + conic vs central differences."""
    means, log_s, quats, V = scene(n=4)
    means.requires_grad_(True)
    log_s.requires_grad_(True)
    quats.requires_grad_(True)

    def f(m, s, q):
        cov3d = P.covariance_3d(s, q)
        m2d, cov2d, depth, _ = P.project_gaussians(m, cov3d, V, **INTR)
        return m2d, P.conic_from_cov2d(cov2d), depth

    assert torch.autograd.gradcheck(f, (means, log_s, quats), eps=1e-6, atol=1e-5)


def test_projected_covariance_monte_carlo():
    """Sample a far-away, small 3D Gaussian, push samples through the pinhole
    model, and compare the empirical 2D covariance with the EWA prediction."""
    means, log_s, quats, V = scene(n=1, seed=3)
    cov3d = P.covariance_3d(log_s, quats)
    _, cov2d, _, _ = P.project_gaussians(means, cov3d, V, **INTR, dilation=0.0)
    L = torch.linalg.cholesky(cov3d[0])
    g = torch.Generator().manual_seed(0)
    x = means[0] + torch.randn(200_000, 3, generator=g) @ L.T
    pc = x @ V[:3, :3].T + V[:3, 3]
    uv = torch.stack([INTR["fx"] * pc[:, 0] / pc[:, 2], INTR["fy"] * pc[:, 1] / pc[:, 2]], -1)
    emp = torch.cov(uv.T)
    assert torch.allclose(emp, cov2d[0], rtol=0.05, atol=1e-3 * cov2d[0].abs().max().item())


def test_screen_extent_is_exact_alpha_threshold():
    cov2d = torch.tensor([[[9.0, 2.0], [2.0, 4.0]]])
    o = torch.tensor([0.8])
    ext = P.screen_extents(cov2d, o)[0]
    conic = P.conic_from_cov2d(cov2d)[0]
    # walk along x at the optimal y for that x, alpha at the boundary is 1/255
    # the extreme point of the ellipse in x is d = k * Sigma e_x / sqrt(k Sigma_xx)
    k = 2 * math.log(255 * 0.8)
    d = k * cov2d[0][:, 0] / math.sqrt(k * cov2d[0, 0, 0])
    q = conic[0] * d[0] ** 2 + 2 * conic[1] * d[0] * d[1] + conic[2] * d[1] ** 2
    alpha = 0.8 * math.exp(-0.5 * q)
    assert abs(d[0] - ext[0]) < 1e-9
    assert abs(alpha - 1 / 255) < 1e-9
    assert (P.screen_extents(cov2d, torch.tensor([0.003])) < 0).all()


def test_sh0_roundtrip():
    rgb = torch.rand(10, 3)
    assert torch.allclose(P.sh0_to_rgb(P.rgb_to_sh0(rgb)), rgb)
