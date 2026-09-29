import math

import torch

from diffusion.process import SIGMA_MAX, SIGMA_MIN, Linear, VPCosine


def test_sigma_roundtrip():
    for p in (VPCosine(), Linear()):
        t = torch.linspace(0.01, 0.98, 50, dtype=torch.float64)
        assert torch.allclose(p.t_of_sigma(p.sigma(t)), t, atol=1e-10)
        lo, hi = p.t_range()
        assert math.isclose(p.sigma(torch.tensor(hi, dtype=torch.float64)).item(), SIGMA_MAX, rel_tol=1e-9)
        assert math.isclose(p.sigma(torch.tensor(lo, dtype=torch.float64)).item(), SIGMA_MIN, rel_tol=1e-9)


def test_closed_form_matches_iterative_coefficients():
    """prod(1 - beta) from the discrete chain equals alpha(t)^2 from the closed form."""
    p = VPCosine()
    n = 1000
    betas = p.discrete_betas(n)
    abar = torch.cumprod(1 - betas, 0)
    t = torch.arange(1, n + 1, dtype=torch.float64) / n
    # the last step is clamped to beta = 0.999 (alpha(1) = 0), skip it
    assert torch.allclose(abar[:-1], p.alpha(t[:-1]) ** 2, rtol=1e-9, atol=1e-12)


def test_closed_form_matches_iterative_noising_in_distribution():
    """Run x_i = sqrt(1-b_i) x_{i-1} + sqrt(b_i) z_i for k steps and compare with the closed form."""
    p = VPCosine()
    n, k, m = 1000, 600, 200_000
    betas = p.discrete_betas(n)
    g = torch.Generator().manual_seed(0)
    x0 = torch.full((m,), 1.5, dtype=torch.float64)
    x = x0.clone()
    for i in range(k):
        x = torch.sqrt(1 - betas[i]) * x + torch.sqrt(betas[i]) * torch.randn(m, generator=g, dtype=torch.float64)
    t = torch.tensor(k / n, dtype=torch.float64)
    xc = p.q_sample(x0, t.expand(m), torch.randn(m, generator=g, dtype=torch.float64))
    a, s = p.alpha(t).item(), p.s(t).item()
    for xs in (x, xc):
        assert abs(xs.mean().item() - 1.5 * a) < 0.01
        assert abs(xs.std().item() - s) < 0.01
    assert abs(x.mean().item() - xc.mean().item()) < 0.01
    assert abs(x.var().item() - xc.var().item()) < 0.01


def test_iterative_noising_exact_with_shared_noise():
    """Composing the per step Gaussians gives exactly alpha, s of the closed form (deterministic check)."""
    p = VPCosine()
    n = 200
    betas = p.discrete_betas(n)
    mean_coef, var = torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    for i in range(n - 1):
        mean_coef = mean_coef * torch.sqrt(1 - betas[i])
        var = (1 - betas[i]) * var + betas[i]
        t = torch.tensor((i + 1) / n, dtype=torch.float64)
        assert torch.isclose(mean_coef, p.alpha(t), atol=1e-10)
        assert torch.isclose(var, p.s(t) ** 2, atol=1e-10)


def test_linear_process_q_sample():
    p = Linear()
    x0 = torch.ones(4, 3)
    eps = torch.zeros(4, 3)
    t = torch.full((4,), 0.25)
    assert torch.allclose(p.q_sample(x0, t, eps), 0.75 * x0)
