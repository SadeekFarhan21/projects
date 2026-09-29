"""AlphaZero self-play training loop.

Each iteration: play `games_per_iter` self-play games with the current network
(C++ batched PUCT, Dirichlet root noise, visit-count sampling for the first
`temp_moves` plies), push the samples into a sliding-window replay buffer,
then take `train_steps` Adam steps on minibatches drawn uniformly from the
buffer with random left-right mirroring. Every `eval_every` iterations the
network plays a short match against pure MCTS.

Example:
  uv run python -m az.train --minutes 35 --out runs/main
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from az import c4core
from az.evaluate import play_vs_mcts
from az.net import C4Net, NetConfig, pick_device, save_checkpoint
from az.search import NetEvaluator, drive


class ReplayBuffer:
    """Fixed-capacity ring buffer of (obs, pi, z)."""

    def __init__(self, capacity: int) -> None:
        self.obs = np.zeros((capacity, 3, 6, 7), np.float32)
        self.pi = np.zeros((capacity, 7), np.float32)
        self.z = np.zeros(capacity, np.float32)
        self.capacity = capacity
        self.size = 0
        self.head = 0

    def add(self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray) -> None:
        n = len(z)
        if n >= self.capacity:
            obs, pi, z = obs[-self.capacity :], pi[-self.capacity :], z[-self.capacity :]
            n = self.capacity
        idx = (self.head + np.arange(n)) % self.capacity
        self.obs[idx], self.pi[idx], self.z[idx] = obs, pi, z
        self.head = (self.head + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch: int, rng: np.random.Generator, augment: bool = True):
        idx = rng.integers(0, self.size, batch)
        obs, pi, z = self.obs[idx].copy(), self.pi[idx].copy(), self.z[idx].copy()
        if augment:
            flip = rng.random(batch) < 0.5
            obs[flip] = obs[flip][:, :, :, ::-1]
            pi[flip] = pi[flip][:, ::-1]
        return obs, pi, z


def mirror_batch(obs: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-right mirror: the only symmetry of Connect Four."""
    return obs[:, :, :, ::-1].copy(), pi[:, ::-1].copy()


def train_step(net, opt, obs, pi, z, device):
    net.train()
    x = torch.from_numpy(obs).to(device)
    tp = torch.from_numpy(pi).to(device)
    tz = torch.from_numpy(z).to(device)
    logits, v = net(x)
    p_loss = -(tp * F.log_softmax(logits, dim=1)).sum(1).mean()
    v_loss = F.mse_loss(v, tz)
    loss = p_loss + v_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return p_loss.item(), v_loss.item()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--minutes", type=float, default=35.0, help="wall clock budget")
    ap.add_argument("--max-iters", type=int, default=10_000)
    ap.add_argument("--slots", type=int, default=512, help="parallel self-play games")
    ap.add_argument("--games-per-iter", type=int, default=1024)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dir-alpha", type=float, default=1.0)
    ap.add_argument("--dir-eps", type=float, default=0.25)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--buffer", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-rollouts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    net = C4Net(NetConfig(channels=args.channels, blocks=args.blocks)).to(device)
    save_checkpoint(out / "ckpt" / "iter_000.pt", net, {"iter": 0, "games": 0})
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    buf = ReplayBuffer(args.buffer)
    cfg = c4core.PuctConfig(
        num_sims=args.sims,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dir_alpha,
        dirichlet_eps=args.dir_eps,
        temp_moves=args.temp_moves,
    )

    log_path = out / "train_log.csv"
    fields = [
        "iter", "wall_s", "games_total", "samples_total", "selfplay_s", "games_per_hour",
        "evals_per_s", "mean_game_len", "first_player_win", "draw_rate", "train_s",
        "policy_loss", "value_loss", "eval_win_rate", "eval_score", "eval_games",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    t_start = time.time()
    games_total = samples_total = 0
    for it in range(1, args.max_iters + 1):
        # ---- self-play
        net.eval()
        ev = NetEvaluator(net, device)
        t0 = time.time()
        eng = c4core.BatchedGames(args.slots, args.games_per_iter, cfg, int(rng.integers(1 << 62)))
        drive(eng, ev)
        sp_s = time.time() - t0
        obs, pi, z = eng.drain_samples()
        recs = eng.drain_games()
        buf.add(obs, pi, z)
        games_total += len(recs)
        samples_total += len(z)
        lens = [len(r.moves) for r in recs]

        # ---- training
        t1 = time.time()
        pl, vl = [], []
        for _ in range(args.train_steps):
            b = buf.sample(args.batch, rng)
            a, c = train_step(net, opt, *b, device)
            pl.append(a)
            vl.append(c)
        tr_s = time.time() - t1

        row = {
            "iter": it,
            "wall_s": round(time.time() - t_start, 1),
            "games_total": games_total,
            "samples_total": samples_total,
            "selfplay_s": round(sp_s, 2),
            "games_per_hour": round(len(recs) / sp_s * 3600, 0),
            "evals_per_s": round(eng.total_evals() / sp_s, 0),
            "mean_game_len": round(float(np.mean(lens)), 2),
            "first_player_win": round(float(np.mean([r.winner == 0 for r in recs])), 3),
            "draw_rate": round(float(np.mean([r.winner == -1 for r in recs])), 3),
            "train_s": round(tr_s, 2),
            "policy_loss": round(float(np.mean(pl)), 4),
            "value_loss": round(float(np.mean(vl)), 4),
            "eval_win_rate": "",
            "eval_score": "",
            "eval_games": "",
        }
        save_checkpoint(out / "ckpt" / f"iter_{it:03d}.pt", net, {"iter": it, "games": games_total})

        elapsed_min = (time.time() - t_start) / 60
        last = elapsed_min >= args.minutes or it == args.max_iters
        if args.eval_every and (it % args.eval_every == 0 or last):
            net.eval()
            res = play_vs_mcts(
                NetEvaluator(net, device), args.eval_rollouts, args.eval_games, args.sims,
                seed=10_000 + it,
            )
            row["eval_win_rate"] = round(res["win_rate"], 3)
            row["eval_score"] = round(res["score"], 3)
            row["eval_games"] = res["games"]
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(json.dumps(row), flush=True)
        if last:
            break
    save_checkpoint(out / "final.pt", net, {"iter": it, "games": games_total})
    print(f"done in {(time.time() - t_start) / 60:.1f} min, {games_total} games", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
