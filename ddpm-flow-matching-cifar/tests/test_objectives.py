import pytest
import torch

from diffusion.models import DiT, ToyMLP, UNet
from diffusion.objectives import Denoiser, diffusion_loss, target, x0_from_output
from diffusion.process import process_for


@pytest.mark.parametrize("objective", ["eps", "v", "rf"])
def test_x0_inversion(objective):
    """Feeding the exact target through x0_from_output recovers x0."""
    p = process_for(objective)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(64, 5, generator=g, dtype=torch.float64)
    eps = torch.randn(64, 5, generator=g, dtype=torch.float64)
    t = torch.rand(64, generator=g, dtype=torch.float64) * 0.9 + 0.05
    a, s = p.alpha(t)[:, None], p.s(t)[:, None]
    xt = a * x0 + s * eps
    out = target(objective, x0, eps, a, s)
    assert torch.allclose(x0_from_output(objective, out, xt, a, s, t[:, None]), x0, atol=1e-9)


@pytest.mark.parametrize("arch", ["unet", "dit"])
def test_initial_eps_loss_near_one(arch):
    """Zero initialised output means eps prediction starts at loss E[eps^2] = 1 on unit variance data."""
    torch.manual_seed(0)
    net = UNet(ch=32, ch_mult=(1, 2), blocks=1) if arch == "unet" else DiT(dim=64, depth=2)
    x0 = torch.randn(16, 3, 32, 32)  # unit variance data
    y = torch.randint(0, 10, (16,))
    loss = diffusion_loss(net, "eps", x0, y, generator=torch.Generator().manual_seed(1)).item()
    assert 0.95 < loss < 1.05


def test_initial_eps_loss_near_one_after_one_step_perturbation():
    """With a randomly (not zero) initialised last layer the loss is still close to 1."""
    torch.manual_seed(0)
    net = ToyMLP()
    torch.nn.init.normal_(net.out.weight, std=0.01)
    x0 = torch.randn(4096, 2)
    loss = diffusion_loss(net, "eps", x0, None, generator=torch.Generator().manual_seed(1)).item()
    assert 0.9 < loss < 1.1


def test_denoiser_cfg_scale_one_equals_conditional():
    torch.manual_seed(0)
    net = DiT(dim=64, depth=2)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(4, 3, 32, 32) * 3
    y = torch.tensor([0, 1, 2, 3])
    sig = torch.full((4,), 3.0)
    d1 = Denoiser(net, "v", null_label=10, cfg_scale=1.0)(x, sig, y)
    d0 = Denoiser(net, "v", null_label=10, cfg_scale=0.0)(x, sig, y)
    d_null = Denoiser(net, "v")(x, sig, torch.full_like(y, 10))
    assert torch.allclose(d0, d_null, atol=1e-5)
    assert not torch.allclose(d1, d0, atol=1e-5)
