import numpy as np
import torch

from diffusion.analytic import GaussianData, gaussian_w2
from diffusion.metrics import energy_distance, fid_from_feats, kid_from_feats, mmd_rbf


def test_mmd_and_energy_distinguish():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(1000, 2, generator=g), torch.randn(1000, 2, generator=g)
    c = torch.randn(1000, 2, generator=g) + 1.0
    assert abs(mmd_rbf(a, b)) < 0.01 < mmd_rbf(a, c)
    assert energy_distance(a, b) < 0.05 < energy_distance(a, c)


def test_fid_kid_basic():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2000, 16)), rng.normal(size=(2000, 16))
    c = rng.normal(size=(2000, 16)) + 0.5
    assert fid_from_feats(a, b) < 0.1 < fid_from_feats(a, c)
    # FID between Gaussians with equal covariance is the squared mean gap: 16 * 0.25 = 4
    assert abs(fid_from_feats(a, c) - 4.0) < 0.3
    assert abs(kid_from_feats(a, b, subsets=10)[0]) < 0.01 < kid_from_feats(a, c, subsets=10)[0]


def test_gaussian_w2_zero_for_exact():
    d = GaussianData.random(3)
    x = d.data_sample(200_000, torch.Generator().manual_seed(0))
    assert gaussian_w2(x, d.mu, d.cov) < 0.02
