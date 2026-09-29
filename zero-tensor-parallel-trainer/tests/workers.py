"""Small distributed worker functions used by the unit-level tests."""

import torch
import torch.distributed as dist

from dtrain import comm
from dtrain.data_parallel import BucketedDDP
from dtrain.model import GPTConfig, build_reference, MLP, Attention
from dtrain.tensor_parallel import TPMLP, TPAttention, shard_state_dict
from dtrain.data import SyntheticMarkov, shard_batch


def ring_vs_native(env, numel: int):
    torch.manual_seed(100 + env.rank)
    x = torch.randn(numel, dtype=torch.float64)
    a, b = x.clone(), x.clone()
    comm.ring_all_reduce_(a)
    dist.all_reduce(b)
    return (a - b).abs().max().item()


def stats_formula(env):
    comm.STATS.reset()
    t = torch.zeros(1000)
    comm.all_reduce(t)
    out = torch.zeros(1000 // env.world_size)
    comm.reduce_scatter(out, torch.zeros(1000))
    comm.all_gather(torch.zeros(1000), out)
    return comm.STATS.as_dict()


def ddp_no_sync(env, cfg: GPTConfig, accum: int):
    """Gradient accumulation with no_sync must equal one big batch."""
    model = build_reference(cfg, 0)
    ddp = BucketedDDP(model, bucket_cap_mb=0.01)
    data = SyntheticMarkov(cfg.vocab_size, cfg.seq_len)
    x, y = data.batch(0, 16)
    x, y = shard_batch(x, env.rank, env.world_size), shard_batch(y, env.rank, env.world_size)
    xs, ys = x.chunk(accum), y.chunk(accum)
    for i in range(accum):
        if i < accum - 1:
            with ddp.no_sync():
                (model(xs[i], ys[i])[1] / accum).backward()
        else:
            (model(xs[i], ys[i])[1] / accum).backward()
    ddp.finish_grad_sync()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def tp_block_forward_backward(env, cfg: GPTConfig):
    """TP MLP and attention outputs and input grads equal the dense modules."""
    torch.manual_seed(0)
    mlp, attn = MLP(cfg), Attention(cfg)
    for m in (mlp, attn):
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.1)
    tmlp, tattn = TPMLP(cfg, env.world_size), TPAttention(cfg, env.world_size)
    tmlp.load_state_dict(shard_state_dict({"b.mlp." + k: v for k, v in mlp.state_dict().items()},
                                          env.rank, env.world_size) and
                         {k[len("b.mlp."):]: v for k, v in shard_state_dict(
                             {"b.mlp." + k: v for k, v in mlp.state_dict().items()},
                             env.rank, env.world_size).items()})
    tattn.load_state_dict({k[len("b.attn."):]: v for k, v in shard_state_dict(
        {"b.attn." + k: v for k, v in attn.state_dict().items()}, env.rank, env.world_size).items()})
    x = torch.randn(2, cfg.seq_len, cfg.d_model, generator=torch.Generator().manual_seed(1))
    out = {}
    for name, dense, tp in (("mlp", mlp, tmlp), ("attn", attn, tattn)):
        xd, xt = x.clone().requires_grad_(), x.clone().requires_grad_()
        yd, yt = dense(xd), tp(xt)
        yd.pow(2).sum().backward()
        yt.pow(2).sum().backward()
        out[name] = ((yd - yt).abs().max().item(), (xd.grad - xt.grad).abs().max().item())
    return out
