"""Tile rasterizer vs a brute-force per-pixel renderer on tiny scenes."""
import pytest
import torch

from splat import projection as P
from splat.rasterize import bin_gaussians, rasterize, render_bruteforce


def make_scene(n=25, W=45, H=37, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    means2d = torch.rand(n, 2, generator=g, dtype=dtype) * torch.tensor([W + 20, H + 20], dtype=dtype) - 10
    # random SPD 2x2 covariances of a few pixels
    A = torch.randn(n, 2, 2, generator=g, dtype=dtype) * 2.5
    cov2d = A @ A.transpose(-1, -2) + 0.5 * torch.eye(2, dtype=dtype)
    opa = torch.rand(n, generator=g, dtype=dtype) * 0.95 + 0.02
    colors = torch.rand(n, 3, generator=g, dtype=dtype)
    depths = torch.rand(n, generator=g, dtype=dtype) * 5 + 1
    valid = torch.rand(n, generator=g) > 0.1
    return means2d, cov2d, opa, colors, depths, valid, W, H


def fast(means2d, cov2d, opa, colors, depths, valid, W, H, **kw):
    ext = P.screen_extents(cov2d, opa)
    return rasterize(means2d, P.conic_from_cov2d(cov2d), colors, opa, depths, ext, valid, W, H, **kw)


def slow(means2d, cov2d, opa, colors, depths, valid, W, H, **kw):
    return render_bruteforce(means2d, P.conic_from_cov2d(cov2d), colors, opa, depths, valid, W, H, **kw)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_forward_matches_bruteforce(seed, chunk):
    s = make_scene(seed=seed)
    bg = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    a = fast(*s, background=bg, chunk=chunk, early_stop=False)
    b = slow(*s, background=bg)
    assert a.shape == b.shape == (s[-1], s[-2], 3)
    assert torch.allclose(a, b, atol=1e-12)


def test_dense_overlap_and_early_stop():
    """Many opaque Gaussians stacked on one spot: early termination must stay
    within T_EPS of the exact answer."""
    s = list(make_scene(n=200, W=32, H=32, seed=7))
    s[0] = torch.full_like(s[0], 16.0) + torch.randn_like(s[0])
    s[2] = torch.full_like(s[2], 0.95)
    a = fast(*s, chunk=8, early_stop=True)
    b = slow(*s)
    assert (a - b).abs().max() < 1e-4


def test_crop_equals_slice_of_full_render():
    s = make_scene(n=40, W=70, H=50, seed=3)
    full = fast(*s, early_stop=False)
    crop = fast(*s, rect=(16, 32, 60, 50), early_stop=False)
    assert torch.allclose(crop, full[32:50, 16:60])


def test_tile_batching_is_invisible():
    s = make_scene(n=40, W=70, H=50, seed=4)
    assert torch.allclose(fast(*s, tile_batch=2, chunk=4), fast(*s, chunk=4))


def test_binning_never_misses_a_contribution():
    """Every (pixel, Gaussian) pair with alpha >= 1/255 lies in a tile the Gaussian was binned to."""
    means2d, cov2d, opa, colors, depths, valid, W, H = make_scene(n=60, seed=5)
    ext = P.screen_extents(cov2d, opa)
    bins = bin_gaussians(means2d, ext, depths, valid, (0, 0, W, H))
    conic = P.conic_from_cov2d(cov2d)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2).double() + 0.5
    for gi in torch.nonzero(valid).squeeze(1).tolist():
        d = pix - means2d[gi]
        a, b, c = conic[gi]
        alpha = opa[gi] * torch.exp(-0.5 * (a * d[:, 0] ** 2 + c * d[:, 1] ** 2) - b * d[:, 0] * d[:, 1])
        hit = pix[alpha >= 1 / 255]
        tiles = set(((hit[:, 1] // 16) * bins.ntx + hit[:, 0] // 16).long().tolist())
        for t in tiles:
            st, n = bins.tile_start[t].item(), bins.tile_count[t].item()
            assert gi in bins.sorted_gid[st:st + n].tolist()


def test_per_tile_depth_order():
    means2d, cov2d, opa, colors, depths, valid, W, H = make_scene(n=60, seed=6)
    bins = bin_gaussians(means2d, P.screen_extents(cov2d, opa), depths, valid, (0, 0, W, H))
    for t in range(bins.ntx * bins.nty):
        st, n = bins.tile_start[t].item(), bins.tile_count[t].item()
        d = depths[bins.sorted_gid[st:st + n]]
        assert torch.all(d[1:] >= d[:-1])


@pytest.mark.parametrize("ckpt", [True, False])
def test_gradients_match_bruteforce(ckpt):
    means2d, cov2d, opa, colors, depths, valid, W, H = make_scene(n=15, W=30, H=20, seed=2)
    leaves = [t.clone().requires_grad_(True) for t in (means2d, cov2d, opa, colors)]
    w = torch.randn(H, W, 3, dtype=torch.float64)

    def loss(fn, ls, **kw):
        m, c, o, col = ls
        c = 0.5 * (c + c.transpose(-1, -2))
        return (fn(m, c, o, col, depths, valid, W, H, **kw) * w).sum()

    ga = torch.autograd.grad(loss(fast, leaves, chunk=4, early_stop=False, use_checkpoint=ckpt), leaves)
    gb = torch.autograd.grad(loss(slow, leaves), leaves)
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, atol=1e-10)


def test_rasterizer_gradcheck():
    """Finite-difference check of the tiled path itself (tiny scene, float64).
    Extents are frozen so the test probes compositing, not tile membership."""
    means2d, cov2d, opa, colors, depths, valid, W, H = make_scene(n=4, W=20, H=18, seed=11)
    valid = torch.ones_like(valid)
    conic = P.conic_from_cov2d(cov2d)
    ext = P.screen_extents(cov2d, opa) + 3.0  # generous, so FD nudges never cross a bin edge
    m = means2d.clone().requires_grad_(True)
    con = conic.clone().requires_grad_(True)
    col = colors.clone().requires_grad_(True)
    o = (opa * 0.5).clone().requires_grad_(True)

    def f(m, con, col, o):
        return rasterize(m, con, col, o, depths, ext, valid, W, H, chunk=2, early_stop=False)

    assert torch.autograd.gradcheck(f, (m, con, col, o), eps=1e-7, atol=1e-5, nondet_tol=0.0)
