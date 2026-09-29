"""Functional coverage bins for the systolic array testbench.

Every bin is a named counter. The testbench samples shapes, tile edges, command
kinds, value extremes and handshake stall patterns; `merge` combines the JSON
files written by separate simulator runs and `closure` reports empty bins.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _dim_bin(x: int, n: int) -> str:
    if x < n:
        return "lt_n"
    if x == n:
        return "eq_n"
    return "mult_n" if x % n == 0 else "gt_n_ragged"


def _run_bin(r: int) -> str:
    if r == 0:
        return "0"
    if r == 1:
        return "1"
    if r <= 4:
        return "2_4"
    if r <= 15:
        return "5_15"
    return "16_plus"


BINS: dict[str, list[str]] = {
    "m_shape": ["one", "lt_n", "eq_n", "n_to_buf", "eq_buf", "multi_chunk_even", "multi_chunk_ragged"],
    "k_shape": ["lt_n", "eq_n", "mult_n", "gt_n_ragged"],
    "n_shape": ["lt_n", "eq_n", "mult_n", "gt_n_ragged"],
    "tile_edge": ["partial_k_tile", "partial_n_tile", "partial_m_chunk", "all_full"],
    "cmd_kind": ["lw1_acc0_dr1", "lw1_acc0_dr0", "lw1_acc1_dr0", "lw1_acc1_dr1", "lw0_acc0_dr1"],
    "values": ["a_min", "a_max", "w_min", "w_max", "out_abs_gt_2p20", "out_neg", "out_zero"],
    "o_stall_run": ["0", "1", "2_4", "5_15", "16_plus"],
    "o_stall_pos": ["first_beat", "middle_beat", "last_beat"],
    "in_gap_run": ["0", "1", "2_4", "5_15", "16_plus"],
    "in_gap_stream": ["cmd", "w", "a"],
    "array_n": ["4", "8", "16"],
}


class Coverage:
    def __init__(self) -> None:
        self.hits: dict[str, Counter] = {k: Counter() for k in BINS}

    def hit(self, group: str, name: str, count: int = 1) -> None:
        if name not in BINS[group]:
            raise KeyError(f"unknown bin {group}.{name}")
        self.hits[group][name] += count

    def sample_gemm(self, M: int, K: int, Nout: int, n: int, buf_rows: int, A, B, C) -> None:
        if M == 1:
            self.hit("m_shape", "one")
        elif M < n:
            self.hit("m_shape", "lt_n")
        elif M == n:
            self.hit("m_shape", "eq_n")
        elif M < buf_rows:
            self.hit("m_shape", "n_to_buf")
        elif M == buf_rows:
            self.hit("m_shape", "eq_buf")
        else:
            self.hit("m_shape", "multi_chunk_even" if M % buf_rows == 0 else "multi_chunk_ragged")
        self.hit("k_shape", _dim_bin(K, n))
        self.hit("n_shape", _dim_bin(Nout, n))
        full = True
        if K % n:
            self.hit("tile_edge", "partial_k_tile")
            full = False
        if Nout % n:
            self.hit("tile_edge", "partial_n_tile")
            full = False
        if M % buf_rows:
            self.hit("tile_edge", "partial_m_chunk")
            full = False
        if full:
            self.hit("tile_edge", "all_full")
        if (A == -128).any():
            self.hit("values", "a_min")
        if (A == 127).any():
            self.hit("values", "a_max")
        if (B == -128).any():
            self.hit("values", "w_min")
        if (B == 127).any():
            self.hit("values", "w_max")
        if (abs(C.astype("int64")) > (1 << 20)).any():
            self.hit("values", "out_abs_gt_2p20")
        if (C < 0).any():
            self.hit("values", "out_neg")
        if (C == 0).any():
            self.hit("values", "out_zero")
        if str(n) in BINS["array_n"]:
            self.hit("array_n", str(n))

    def sample_cmd(self, load_w: bool, acc: bool, drain: bool) -> None:
        self.hit("cmd_kind", f"lw{int(load_w)}_acc{int(acc)}_dr{int(drain)}")

    def sample_o_stall(self, run: int, beat: int, beats: int) -> None:
        self.hit("o_stall_run", _run_bin(run))
        if run:
            if beat == 0:
                self.hit("o_stall_pos", "first_beat")
            elif beat == beats - 1:
                self.hit("o_stall_pos", "last_beat")
            else:
                self.hit("o_stall_pos", "middle_beat")

    def sample_in_gap(self, stream: str, run: int) -> None:
        self.hit("in_gap_run", _run_bin(run))
        if run:
            self.hit("in_gap_stream", stream)

    def to_json(self) -> dict:
        return {g: dict(c) for g, c in self.hits.items()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1, sort_keys=True))


def merge(paths) -> dict[str, dict[str, int]]:
    total: dict[str, Counter] = {k: Counter() for k in BINS}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for g, bins in d.items():
            total[g].update(bins)
    return {g: {b: int(total[g][b]) for b in BINS[g]} for g in BINS}


def closure(merged: dict[str, dict[str, int]]) -> tuple[int, int, list[str]]:
    holes = [f"{g}.{b}" for g, bins in merged.items() for b, v in bins.items() if v == 0]
    total = sum(len(v) for v in merged.values())
    return total - len(holes), total, holes
