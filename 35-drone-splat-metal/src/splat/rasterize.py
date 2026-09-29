"""Pure-tensor tile rasterizer for 2D Gaussians.

Pipeline (all PyTorch ops, runs on CPU or MPS):

1. bin      : every Gaussian is duplicated once per 16x16 tile its alpha>=1/255
              footprint touches  -> (tile_id, gaussian_id) pairs
2. sort     : global depth argsort, then a *stable* sort by tile id, so each
              tile's pairs come out front-to-back
3. composite: tiles are processed in lock-step chunks of C Gaussians. Chunk k
              handles the k-th group of C Gaussians of every tile that still has
              that many. Inside a chunk, transmittance is an exclusive cumprod;
              between chunks a per-pixel transmittance T is carried.

Backward: autograd, with each chunk wrapped in activation checkpointing so
peak memory is O(pixels * C) instead of O(pixels * Gaussians-per-tile).

Only the pairs step and the tile bookkeeping are non-differentiable (integer
work); every quantity that touches a parameter stays in the autograd graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .projection import ALPHA_MAX, ALPHA_MIN

TILE = 16
T_EPS = 1e-4  # early termination threshold on transmittance (same as 3DGS)


@dataclass
class TileBins:
    ntx: int
    nty: int
    sorted_gid: torch.Tensor  # [P] gaussian id, grouped by tile, front to back
    tile_start: torch.Tensor  # [T] offset into sorted_gid
    tile_count: torch.Tensor  # [T]


@torch.no_grad()
def bin_gaussians(means2d, extents, depths, valid, rect, tile=TILE) -> TileBins:
    """rect = (x0, y0, x1, y1) in pixels, half-open. x0, y0 multiples of tile."""
    x0, y0, x1, y1 = rect
    ntx = (x1 - x0 + tile - 1) // tile
    nty = (y1 - y0 + tile - 1) // tile
    dev = means2d.device
    m = means2d.detach()
    ex, ey = extents[:, 0], extents[:, 1]
    # a pixel centre p is touched iff |p - m| <= e on both axes
    vis = (
        valid
        & (ex >= 0)
        & (m[:, 0] + ex >= x0 + 0.5)
        & (m[:, 0] - ex <= x1 - 0.5)
        & (m[:, 1] + ey >= y0 + 0.5)
        & (m[:, 1] - ey <= y1 - 0.5)
    )
    ids = torch.nonzero(vis).squeeze(1)
    # front to back
    ids = ids[torch.argsort(depths.detach()[ids])]
    mx, my = m[ids, 0], m[ids, 1]
    lx = torch.floor((mx - ex[ids] - 0.5 - x0) / tile).clamp(0, ntx - 1).long()
    hx = torch.floor((mx + ex[ids] - 0.5 - x0) / tile).clamp(0, ntx - 1).long()
    ly = torch.floor((my - ey[ids] - 0.5 - y0) / tile).clamp(0, nty - 1).long()
    hy = torch.floor((my + ey[ids] - 0.5 - y0) / tile).clamp(0, nty - 1).long()
    wx = hx - lx + 1
    cnt = wx * (hy - ly + 1)
    total = int(cnt.sum())
    ntiles = ntx * nty
    if total == 0:
        z = torch.zeros(ntiles, dtype=torch.long, device=dev)
        return TileBins(ntx, nty, torch.zeros(0, dtype=torch.long, device=dev), z, z.clone())
    owner = torch.repeat_interleave(torch.arange(len(ids), device=dev), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    k = torch.arange(total, device=dev) - first[owner]
    tx = lx[owner] + k % wx[owner]
    ty = ly[owner] + k // wx[owner]
    tile_id = ty * ntx + tx
    order = torch.sort(tile_id, stable=True).indices  # keeps depth order inside a tile
    sorted_gid = ids[owner[order]]
    tile_count = torch.bincount(tile_id, minlength=ntiles)
    tile_start = torch.cumsum(tile_count, 0) - tile_count
    return TileBins(ntx, nty, sorted_gid, tile_start, tile_count)


def _composite_chunk(means2d, conics, colors, opacities, gidx, gvalid, pix, T_in):
    """One chunk of front-to-back compositing.

    gidx, gvalid: [A, C]; pix: [A, P, 2]; T_in: [A, P]
    returns rgb contribution [A, P, 3] and transmittance after the chunk [A, P]
    """
    mu = means2d[gidx]  # [A, C, 2]
    con = conics[gidx]  # [A, C, 3]
    d = pix[:, :, None, :] - mu[:, None, :, :]  # [A, P, C, 2]
    dx, dy = d[..., 0], d[..., 1]
    a, b, c = con[..., 0][:, None], con[..., 1][:, None], con[..., 2][:, None]
    power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
    alpha = (opacities[gidx][:, None, :] * torch.exp(power)).clamp(max=ALPHA_MAX)
    keep = gvalid[:, None, :] & (alpha >= ALPHA_MIN) & (power <= 0)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    one_m = 1.0 - alpha
    T_incl = torch.cumprod(one_m, dim=-1)
    T_excl = torch.cat([torch.ones_like(T_incl[..., :1]), T_incl[..., :-1]], -1) * T_in[..., None]
    w = alpha * T_excl  # [A, P, C]
    rgb = torch.einsum("apc,acd->apd", w, colors[gidx])
    return rgb, T_in * T_incl[..., -1]


def rasterize(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    extents: torch.Tensor,
    valid: torch.Tensor,
    width: int,
    height: int,
    rect: tuple[int, int, int, int] | None = None,
    background: torch.Tensor | None = None,
    chunk: int = 64,
    tile_batch: int | None = None,
    early_stop: bool = True,
    use_checkpoint: bool | None = None,
    return_stats: bool = False,
):
    """Render the pixels of ``rect`` (default whole image). Returns [h, w, 3]."""
    if rect is None:
        rect = (0, 0, width, height)
    x0, y0, x1, y1 = rect
    assert x0 % TILE == 0 and y0 % TILE == 0
    dev, dt = means2d.device, means2d.dtype
    bins = bin_gaussians(means2d, extents, depths, valid, rect)
    ntiles = bins.ntx * bins.nty
    P = TILE * TILE
    if use_checkpoint is None:
        use_checkpoint = torch.is_grad_enabled()

    # pixel centres of every tile: [T, P, 2]
    ty, tx = torch.div(torch.arange(ntiles, device=dev), bins.ntx, rounding_mode="floor"), torch.arange(ntiles, device=dev) % bins.ntx
    py, px = torch.meshgrid(torch.arange(TILE, device=dev), torch.arange(TILE, device=dev), indexing="ij")
    pix_x = x0 + tx[:, None] * TILE + px.reshape(1, -1) + 0.5
    pix_y = y0 + ty[:, None] * TILE + py.reshape(1, -1) + 0.5
    pix = torch.stack([pix_x, pix_y], -1).to(dt)

    rgb_acc = torch.zeros(ntiles, P, 3, device=dev, dtype=dt)
    T = torch.ones(ntiles, P, device=dev, dtype=dt)
    count = bins.tile_count
    kmax = int(count.max()) if ntiles > 0 else 0
    ar = torch.arange(chunk, device=dev)
    npairs_processed = 0
    alive = torch.ones(ntiles, dtype=torch.bool, device=dev)
    for c0 in range(0, kmax, chunk):
        act = torch.nonzero((count > c0) & alive).squeeze(1)
        if act.numel() == 0:
            break
        batches = [act] if tile_batch is None else list(torch.split(act, tile_batch))
        for A in batches:
            slot = bins.tile_start[A, None] + c0 + ar[None, :]
            gvalid = (c0 + ar[None, :]) < count[A, None]
            slot = torch.where(gvalid, slot, torch.zeros_like(slot))
            gidx = bins.sorted_gid[slot]
            npairs_processed += int(gvalid.sum())
            args = (means2d, conics, colors, opacities, gidx, gvalid, pix[A], T[A])
            if use_checkpoint:
                rgb, T_out = checkpoint(_composite_chunk, *args, use_reentrant=False)
            else:
                rgb, T_out = _composite_chunk(*args)
            if torch.is_grad_enabled():
                rgb_acc = rgb_acc.index_add(0, A, rgb)
                T = T.index_copy(0, A, T_out)
            else:
                rgb_acc.index_add_(0, A, rgb)
                T.index_copy_(0, A, T_out)
        if early_stop:
            alive = alive & (T.detach().amax(dim=1) > T_EPS)

    if background is not None:
        rgb_acc = rgb_acc + T[..., None] * background.to(dt)
    img = (
        rgb_acc.reshape(bins.nty, bins.ntx, TILE, TILE, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(bins.nty * TILE, bins.ntx * TILE, 3)[: y1 - y0, : x1 - x0]
    )
    if return_stats:
        stats = {
            "pairs": int(bins.sorted_gid.numel()),
            "pairs_processed": npairs_processed,
            "max_per_tile": kmax,
            "tiles": ntiles,
        }
        return img, stats
    return img


def render_bruteforce(means2d, conics, colors, opacities, depths, valid, width, height, background=None):
    """Reference renderer: every pixel walks every Gaussian in global depth order.

    No tiles, no extents, no chunking, no early termination. Only the per-Gaussian
    alpha rule (clamp to 0.99, skip alpha < 1/255) is shared with the fast path.
    Differentiable; meant for tiny scenes in tests.
    """
    dev, dt = means2d.device, means2d.dtype
    ids = torch.nonzero(valid).squeeze(1)
    ids = ids[torch.argsort(depths.detach()[ids])]
    ys, xs = torch.meshgrid(torch.arange(height, device=dev), torch.arange(width, device=dev), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2).to(dt) + 0.5
    out = torch.zeros(pix.shape[0], 3, device=dev, dtype=dt)
    T = torch.ones(pix.shape[0], device=dev, dtype=dt)
    for g in ids.tolist():
        d = pix - means2d[g]
        a, b, c = conics[g]
        power = -0.5 * (a * d[:, 0] ** 2 + c * d[:, 1] ** 2) - b * d[:, 0] * d[:, 1]
        alpha = (opacities[g] * torch.exp(power)).clamp(max=ALPHA_MAX)
        alpha = torch.where((alpha >= ALPHA_MIN) & (power <= 0), alpha, torch.zeros_like(alpha))
        out = out + (alpha * T)[:, None] * colors[g]
        T = T * (1 - alpha)
    if background is not None:
        out = out + T[:, None] * background.to(dt)
    return out.reshape(height, width, 3)
