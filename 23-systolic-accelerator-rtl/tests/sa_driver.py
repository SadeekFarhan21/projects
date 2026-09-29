"""cocotb driver for sa_top.

One coroutine runs the whole handshake loop, sampling and driving on the
falling clock edge. All DUT ready signals and o_valid are functions of
registered state, so a value sampled at the falling edge is the value the DUT
sees at the next rising edge, and a transfer decided here happens there.

Input traffic is one ordered read channel: the weight and activation beats of
every command in schedule order. Only the head beat is offered, on its own
stream. Commands travel on a separate side channel.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from cocotb.triggers import FallingEdge, RisingEdge

from sysarray.coverage import Coverage
from sysarray.schedule import Cmd, golden, pad_to, schedule


@dataclass
class Policy:
    """Handshake timing policy.

    mode = "full"   inputs always valid, output always ready
    mode = "random" random valid gaps and random output backpressure
    mode = "bw"     token bucket of `bw` bytes per cycle per direction
    """
    mode: str = "full"
    p_in_gap: float = 0.0       # chance an idle input stays idle this cycle
    p_o_stall: float = 0.0      # chance o_ready is low this cycle
    burst_stall: int = 0        # max extra stall run started when stalling
    bw: float = math.inf
    rng: random.Random = field(default_factory=random.Random)

    @staticmethod
    def random_profile(rng: random.Random) -> "Policy":
        kind = rng.choice(["none", "light", "heavy", "bursty", "in_only", "out_only"])
        if kind == "none":
            return Policy("full", rng=rng)
        if kind == "light":
            return Policy("random", 0.1, 0.1, 0, rng=rng)
        if kind == "heavy":
            return Policy("random", 0.5, 0.6, 0, rng=rng)
        if kind == "bursty":
            return Policy("random", 0.05, 0.08, 24, rng=rng)
        if kind == "in_only":
            return Policy("random", 0.4, 0.0, 0, rng=rng)
        return Policy("random", 0.0, 0.4, 6, rng=rng)


def pack_i8(row: np.ndarray) -> int:
    return int.from_bytes(row.astype(np.int8).tobytes(), "little")


def unpack_i32(v: int, n: int) -> np.ndarray:
    return np.frombuffer(v.to_bytes(4 * n, "little"), dtype="<i4").copy()


class SADriver:
    def __init__(self, dut, n: int, adepth: int, cov: Coverage | None = None):
        self.dut = dut
        self.n = n
        self.adepth = adepth
        self.cov = cov or Coverage()
        self.fe = FallingEdge(dut.clk)
        self.re = RisingEdge(dut.clk)
        self.stability_errors = 0

    async def reset(self) -> None:
        d = self.dut
        for s in (d.cmd_valid, d.w_valid, d.a_valid, d.o_ready):
            s.value = 0
        d.cmd_m.value = 0
        d.cmd_load_w.value = 0
        d.cmd_acc.value = 0
        d.cmd_drain.value = 0
        d.w_data.value = 0
        d.a_data.value = 0
        d.rst_n.value = 0
        for _ in range(3):
            await self.re
        d.rst_n.value = 1
        await self.fe

    async def run_gemm(self, A: np.ndarray, B: np.ndarray, buf_rows: int,
                       policy: Policy, max_cycles: int = 2_000_000):
        """Run one GEMM. Returns (C, cycles) where cycles counts from the
        cycle the first command is offered to the cycle of the last output
        transfer, inclusive."""
        n = self.n
        M, K = A.shape
        Nout = B.shape[1]
        assert buf_rows <= self.adepth
        cmds: list[Cmd] = schedule(M, K, Nout, n, buf_rows)
        kt, nt = math.ceil(K / n), math.ceil(Nout / n)
        Ap = pad_to(A, M, kt * n)
        Bp = pad_to(B, kt * n, nt * n)

        # Ordered input channel: (stream, packed row)
        beats: list[tuple[str, int]] = []
        out_plan: list[tuple[int, int, int, int]] = []   # (row, n0, beat, beats)
        for c in cmds:
            if c.load_w:
                for k in range(n):
                    beats.append(("w", pack_i8(Bp[c.k0 + k, c.n0:c.n0 + n])))
            for r in range(c.m):
                beats.append(("a", pack_i8(Ap[c.m0 + r, c.k0:c.k0 + n])))
            if c.drain:
                for r in range(c.m):
                    out_plan.append((c.m0 + r, c.n0, r, c.m))
            self.cov.sample_cmd(c.load_w, c.acc, c.drain)

        Cp = np.zeros((M, nt * n), dtype=np.int32)
        d = self.dut
        h_cv, h_cr = d.cmd_valid, d.cmd_ready
        h_wv, h_wr, h_wd = d.w_valid, d.w_ready, d.w_data
        h_av, h_ar, h_ad = d.a_valid, d.a_ready, d.a_data
        h_ov, h_or, h_od = d.o_valid, d.o_ready, d.o_data
        rng = policy.rng
        S_in, S_out = n, 4 * n

        ci = bi = oi = 0
        cmd_on = beat_on = False            # a valid is being held
        cmd_fired = beat_fired = False      # transfer at the coming rising edge
        cur_cv = cur_wv = cur_av = False
        cur_or = 0
        credit_in, credit_out = float(S_in), float(S_out)
        in_gap = {"cmd": 0, "w": 0, "a": 0}
        o_stall = 0
        o_stall_left = 0
        prev_stall = False
        prev_od = None
        cycles = 0
        n_out = len(out_plan)

        while oi < n_out:
            await self.fe
            cycles += 1
            if cycles > max_cycles:
                raise TimeoutError(f"GEMM {M}x{K}x{Nout} did not finish in {max_cycles} cycles")
            cr = int(h_cr.value)
            wr = int(h_wr.value)
            ar = int(h_ar.value)
            ov = int(h_ov.value)

            # Python side handshake check (the RTL checks the same in SVA A1)
            if prev_stall:
                od_now = int(h_od.value)
                if not ov or od_now != prev_od:
                    self.stability_errors += 1


            # ---- retire transfers that happened at the last rising edge
            if cmd_fired:
                ci += 1
                cmd_on = cmd_fired = False
            if beat_fired:
                bi += 1
                beat_on = beat_fired = False

            # ---- command side channel
            if ci < len(cmds) and not cmd_on:
                if policy.mode == "random" and rng.random() < policy.p_in_gap:
                    if cr:
                        in_gap["cmd"] += 1
                else:
                    c = cmds[ci]
                    d.cmd_m.value = c.m
                    d.cmd_load_w.value = int(c.load_w)
                    d.cmd_acc.value = int(c.acc)
                    d.cmd_drain.value = int(c.drain)
                    cmd_on = True
            if cmd_on != cur_cv:
                h_cv.value = int(cmd_on)
                cur_cv = cmd_on
            if cmd_on and cr:
                cmd_fired = True
                self.cov.sample_in_gap("cmd", in_gap["cmd"])
                in_gap["cmd"] = 0

            # ---- ordered input channel (weights and activations)
            if bi < len(beats):
                stream, data = beats[bi]
                rdy = wr if stream == "w" else ar
                if not beat_on:
                    if policy.mode == "random":
                        ok = rng.random() >= policy.p_in_gap
                    elif policy.mode == "bw":
                        ok = credit_in >= S_in
                    else:
                        ok = True
                    if ok:
                        (h_wd if stream == "w" else h_ad).value = data
                        beat_on = True
                    elif rdy:
                        in_gap[stream] += 1
                wv = beat_on and stream == "w"
                av = beat_on and stream == "a"
                if beat_on and rdy:
                    beat_fired = True
                    self.cov.sample_in_gap(stream, in_gap[stream])
                    in_gap[stream] = 0
                    if policy.mode == "bw":
                        credit_in -= S_in
            else:
                wv = av = False
            if wv != cur_wv:
                h_wv.value = int(wv)
                cur_wv = wv
            if av != cur_av:
                h_av.value = int(av)
                cur_av = av

            # ---- output channel
            if policy.mode == "full":
                want = 1
            elif policy.mode == "bw":
                want = int(credit_out >= S_out)
            else:
                if o_stall_left > 0:
                    o_stall_left -= 1
                    want = 0
                elif rng.random() < policy.p_o_stall:
                    want = 0
                    if policy.burst_stall:
                        o_stall_left = rng.randint(0, policy.burst_stall)
                else:
                    want = 1
            if want != cur_or:
                h_or.value = want
                cur_or = want
            if ov and want:
                row, n0, beat, nb = out_plan[oi]
                Cp[row, n0:n0 + n] = unpack_i32(int(h_od.value), n)
                self.cov.sample_o_stall(o_stall, beat, nb)
                o_stall = 0
                oi += 1
                if policy.mode == "bw":
                    credit_out -= S_out
                prev_stall = False
            elif ov:
                o_stall += 1
                prev_stall = True
                prev_od = int(h_od.value)
            else:
                prev_stall = False

            # Token buckets refill after this cycle's transfers. Below one
            # beat of credit the carry is kept, so a stream of k beats ends
            # ceil((k-1)*S/bw) cycles after its first beat; a full bucket
            # (idle or waiting on ready) is clamped to exactly one beat.
            if policy.mode == "bw":
                credit_in = credit_in + policy.bw if credit_in < S_in else float(S_in)
                credit_out = credit_out + policy.bw if credit_out < S_out else float(S_out)

        # let the final transfer land, then idle the bus
        await self.re
        h_cv.value = 0
        h_wv.value = 0
        h_av.value = 0
        h_or.value = 0
        await self.fe
        C = Cp[:, :Nout]
        self.cov.sample_gemm(M, K, Nout, n, buf_rows, A, B, golden(A, B))
        return C, cycles
