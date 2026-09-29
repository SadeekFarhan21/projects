import torch

from diffusion.analytic import GaussianData
from diffusion.models import DiT, UNet
from diffusion.objectives import Denoiser, diffusion_loss
from diffusion.process import SIGMA_MAX, VPCosine
from diffusion.samplers import sample


def test_seeded_sampling_is_deterministic():
    data = GaussianData.random(3, seed=0)
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(123)
        x = data.marginal_sample(500, SIGMA_MAX, g)
        outs.append(sample("ddpm", data.denoise, x, 20, VPCosine(), generator=g))
    assert torch.equal(outs[0], outs[1])
    g = torch.Generator().manual_seed(124)
    x = data.marginal_sample(500, SIGMA_MAX, g)
    assert not torch.equal(outs[0], sample("ddpm", data.denoise, x, 20, VPCosine(), generator=g))


def _train_steps(seed):
    torch.manual_seed(seed)
    net = UNet(ch=16, ch_mult=(1, 2), blocks=1, dropout=0.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(8, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (8,), generator=g)
    losses = []
    for _ in range(3):
        loss = diffusion_loss(net, "v", x0, y, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, net


def test_seeded_training_is_deterministic():
    l1, n1 = _train_steps(7)
    l2, n2 = _train_steps(7)
    assert l1 == l2
    for a, b in zip(n1.parameters(), n2.parameters()):
        assert torch.equal(a, b)


def test_seeded_model_sampling_is_deterministic():
    torch.manual_seed(0)
    net = DiT(dim=32, depth=1)
    d = Denoiser(net, "rf", null_label=10, cfg_scale=2.0)
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(5)
        x = torch.randn(2, 3, 32, 32, generator=g) * SIGMA_MAX
        outs.append(sample("heun", d, x, 7, d.process, y=torch.tensor([1, 2]), generator=g))
    assert torch.equal(outs[0], outs[1])
