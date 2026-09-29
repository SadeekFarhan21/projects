import pytest
import torch

from diffusion.analytic import GaussianData, gaussian_w2
from diffusion.process import SIGMA_MAX, VPCosine
from diffusion.samplers import SAMPLERS, ddim, euler, karras_sigmas, native_sigmas, nfe_of, sample, steps_for_nfe

DIM = 4
N = 40_000


def _run(sampler, nfe, seed=0):
    data = GaussianData.random(DIM, seed=3)
    g = torch.Generator().manual_seed(seed)
    x = data.marginal_sample(N, SIGMA_MAX, g)
    out = sample(sampler, data.denoise, x, nfe, VPCosine(), generator=g)
    return data, x, out


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_sampler_reproduces_gaussian(sampler):
    nfe = {"ddpm": 512, "ddim": 128, "euler": 256, "heun": 255}[sampler]
    data, _, out = _run(sampler, nfe)
    # compare against a fresh exact sample of the same size (sampling noise floor)
    ref = data.data_sample(N, torch.Generator().manual_seed(99))
    floor = gaussian_w2(ref, data.mu, data.cov)
    w2 = gaussian_w2(out, data.mu, data.cov)
    assert torch.allclose(out.mean(0), data.mu, atol=0.03)
    assert torch.allclose(torch.cov(out.T), data.cov, atol=0.06)
    assert w2 < 3 * floor + 0.02, (w2, floor)


@pytest.mark.parametrize("sampler", ["ddim", "euler", "heun"])
def test_deterministic_samplers_follow_exact_flow(sampler):
    data, x, out = _run(sampler, 255 if sampler == "heun" else 512)
    exact = data.flow_to_zero(x, SIGMA_MAX)
    err = (out - exact).pow(2).sum(1).mean().sqrt().item()
    assert err < 0.02, err


def test_heun_converges_faster_than_euler():
    data, x, _ = _run("euler", 4)
    exact = data.flow_to_zero(x, SIGMA_MAX)
    errs = {}
    for s in ("euler", "heun"):
        out = sample(s, data.denoise, x, 127, VPCosine())
        errs[s] = (out - exact).pow(2).sum(1).mean().sqrt().item()
    assert errs["heun"] < 0.25 * errs["euler"]


def test_ddim_eta0_equals_euler_on_same_grid():
    data = GaussianData.random(DIM, seed=1)
    x = data.marginal_sample(100, SIGMA_MAX, torch.Generator().manual_seed(0))
    sig = karras_sigmas(12)
    assert torch.allclose(ddim(data.denoise, x, sig), euler(data.denoise, x, sig), atol=1e-10)


def test_grids_and_nfe_accounting():
    for n in (1, 4, 17):
        k = karras_sigmas(n)
        v = native_sigmas(n, VPCosine())
        for s in (k, v):
            assert len(s) == n + 1 and s[-1] == 0 and abs(s[0].item() - SIGMA_MAX) < 1e-6
            assert (s[:-1] > s[1:]).all()
    for nfe in (4, 8, 16, 32, 64):
        assert nfe_of("heun", steps_for_nfe("heun", nfe)) <= nfe
        assert nfe_of("euler", steps_for_nfe("euler", nfe)) == nfe


def test_nfe_counter_matches_accounting():
    from diffusion.objectives import Denoiser
    from diffusion.models import ToyMLP

    d = Denoiser(ToyMLP(), "rf")
    x = torch.randn(8, 2) * SIGMA_MAX
    for s, nfe in (("heun", 16), ("euler", 8), ("ddpm", 8)):
        d.nfe = 0
        sample(s, d, x, nfe, d.process, generator=torch.Generator().manual_seed(0))
        assert d.nfe == nfe_of(s, steps_for_nfe(s, nfe))
