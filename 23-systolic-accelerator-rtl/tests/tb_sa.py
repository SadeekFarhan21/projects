"""cocotb tests for sa_top. Run through tests/test_sa.py (pytest), which builds
the Verilator model and passes configuration through environment variables:

  SA_N, SA_ADEPTH   array size and buffer depth the model was built with
  SA_SEED           base random seed
  SA_NUM_GEMMS      GEMMs for test_random_gemms
  SA_COV_OUT        where to write functional coverage JSON
  SA_JOBS_IN        JSON list of GEMM jobs for test_cycle_jobs
  SA_CYCLES_OUT     where test_cycle_jobs writes measured cycles
  SA_ONLY_CASE      replay one random GEMM index (used for waveform replay)
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import cocotb
from cocotb.clock import Clock

from sa_driver import Policy, SADriver
from sysarray.coverage import Coverage
from sysarray.schedule import golden

N = int(os.environ.get("SA_N", "8"))
ADEPTH = int(os.environ.get("SA_ADEPTH", "64"))
SEED = int(os.environ.get("SA_SEED", "1"))


async def setup(dut) -> SADriver:
    Clock(dut.clk, 10, unit="ns").start()
    drv = SADriver(dut, N, ADEPTH)
    await drv.reset()
    return drv


def finish(dut, drv: SADriver, name: str) -> None:
    sva = int(dut.sva_errors.value)
    assert sva == 0, f"{sva} RTL assertion failures"
    assert drv.stability_errors == 0, f"{drv.stability_errors} handshake stability errors"
    out = os.environ.get("SA_COV_OUT")
    if out:
        drv.cov.save(f"{out}.{name}.json")


async def check(drv, A, B, buf_rows, policy, label):
    C, cycles = await drv.run_gemm(A, B, buf_rows, policy)
    ref = golden(A, B)
    bad = np.argwhere(C != ref)
    if len(bad):
        r, c = bad[0]
        raise AssertionError(
            f"{label}: {len(bad)} mismatches, first at ({r},{c}) got {C[r, c]} want {ref[r, c]}")
    return cycles


# ---------------------------------------------------------------- directed

@cocotb.test()
async def test_identity(dut):
    drv = await setup(dut)
    rng = np.random.default_rng(SEED)
    for M, K in [(N, N), (3, N), (2 * N + 1, 2 * N), (ADEPTH, N)]:
        A = rng.integers(-128, 128, (M, K), dtype=np.int8)
        B = np.eye(K, dtype=np.int8)
        await check(drv, A, B, ADEPTH, Policy("full"), f"identity {M}x{K}")
    finish(dut, drv, "identity")


@cocotb.test()
async def test_all_ones(dut):
    drv = await setup(dut)
    for M, K, Nn in [(N, N, N), (1, 1, 1), (5, 3 * N + 2, N + 3), (ADEPTH + 3, 2 * N, 2 * N)]:
        A = np.ones((M, K), dtype=np.int8)
        B = np.ones((K, Nn), dtype=np.int8)
        await check(drv, A, B, ADEPTH, Policy("full"), f"ones {M}x{K}x{Nn}")
    finish(dut, drv, "all_ones")


@cocotb.test()
async def test_int8_extremes(dut):
    drv = await setup(dut)
    rng = random.Random(SEED)
    cases = []
    for a, b in [(-128, -128), (127, 127), (-128, 127), (127, -128)]:
        cases.append((np.full((N, 32 * N), a, np.int8), np.full((32 * N, N), b, np.int8)))
    # random signs of the extremes, deep K so sums pass 2^20
    g = np.random.default_rng(SEED)
    A = g.choice(np.array([-128, 127], np.int8), (2 * N, 128))
    B = g.choice(np.array([-128, 127], np.int8), (128, 2 * N + 1))
    cases.append((A, B))
    for i, (A, B) in enumerate(cases):
        await check(drv, A, B, ADEPTH, Policy.random_profile(rng), f"extreme case {i}")
    finish(dut, drv, "int8_extremes")


@cocotb.test()
async def test_weight_reuse_and_chunks(dut):
    """K fits in one tile and M spans several buffer chunks, so later chunks
    run with load_w = 0 and reuse the weights already in the PEs."""
    drv = await setup(dut)
    rng = np.random.default_rng(SEED + 7)
    for M, K, Nn, buf in [(3 * 16, N, N, 16), (50, N - 1, 2 * N, 16), (2 * ADEPTH, N, N, ADEPTH)]:
        A = rng.integers(-128, 128, (M, K), dtype=np.int8)
        B = rng.integers(-128, 128, (K, Nn), dtype=np.int8)
        await check(drv, A, B, buf, Policy("full"), f"reuse {M}x{K}x{Nn} buf{buf}")
    finish(dut, drv, "weight_reuse")


# ---------------------------------------------------------------- random

def rand_dim(rng: random.Random, n: int, hi: int) -> int:
    kind = rng.random()
    if kind < 0.2:
        return rng.randint(1, max(1, n - 1))
    if kind < 0.35:
        return n
    if kind < 0.5:
        return n * rng.randint(2, 3)
    return rng.randint(n + 1, hi)


def rand_gemm(rng: random.Random, n: int, adepth: int):
    buf = rng.choice([8, 12, 16, 16, 32, adepth])
    buf = min(buf, adepth)
    r = rng.random()
    if r < 0.1:
        M = 1
    elif r < 0.25:
        M = rng.randint(2, max(2, n - 1))
    elif r < 0.35:
        M = n
    elif r < 0.6:
        M = rng.randint(n + 1, max(n + 1, buf))
    elif r < 0.7:
        M = buf
    elif r < 0.8:
        M = buf * rng.randint(2, 3)
    else:
        M = rng.randint(buf + 1, 3 * buf)
    K = rand_dim(rng, n, 3 * n + 3)
    Nn = rand_dim(rng, n, 3 * n + 3)
    vk = rng.random()
    g = np.random.default_rng(rng.getrandbits(32))
    if vk < 0.6:
        A = g.integers(-128, 128, (M, K), dtype=np.int8)
        B = g.integers(-128, 128, (K, Nn), dtype=np.int8)
    elif vk < 0.75:
        A = g.integers(-3, 4, (M, K), dtype=np.int8)
        B = g.integers(-3, 4, (K, Nn), dtype=np.int8)
    elif vk < 0.9:
        A = g.choice(np.array([-128, 127, 0], np.int8), (M, K))
        B = g.choice(np.array([-128, 127, 0], np.int8), (K, Nn))
    else:
        A = g.integers(-128, 128, (M, K), dtype=np.int8) * (g.random((M, K)) < 0.2)
        B = g.integers(-128, 128, (K, Nn), dtype=np.int8)
        A = A.astype(np.int8)
    return A, B, buf


@cocotb.test()
async def test_random_gemms(dut):
    drv = await setup(dut)
    count = int(os.environ.get("SA_NUM_GEMMS", "50"))
    only = os.environ.get("SA_ONLY_CASE")
    rng = random.Random(SEED)
    t0 = time.time()
    total_cycles = 0
    for i in range(count):
        A, B, buf = rand_gemm(rng, N, ADEPTH)
        policy = Policy.random_profile(random.Random(rng.getrandbits(32)))
        if only is not None and i != int(only):
            continue
        try:
            total_cycles += await check(
                drv, A, B, buf, policy,
                f"seed {SEED} case {i} shape {A.shape}x{B.shape[1]} buf {buf} {policy.mode}")
            sva = int(dut.sva_errors.value)
            assert sva == 0 and drv.stability_errors == 0, \
                f"case {i}: {sva} RTL assertion failures, {drv.stability_errors} stability errors"
        except (AssertionError, TimeoutError):
            if os.environ.get("SA_FAIL_OUT"):
                with open(os.environ["SA_FAIL_OUT"], "w") as f:
                    json.dump({"case": i, "seed": SEED}, f)
            raise
    dt = time.time() - t0
    dut._log.info(f"{count} random GEMMs, {total_cycles} cycles, {dt:.1f} s, {total_cycles / max(dt, 1e-9):.0f} cycles/s")
    if os.environ.get("SA_STATS_OUT"):
        with open(os.environ["SA_STATS_OUT"], "w") as f:
            json.dump({"n": N, "seed": SEED, "gemms": count, "mismatching_gemms": 0,
                       "sva_errors": int(dut.sva_errors.value),
                       "stability_errors": drv.stability_errors,
                       "cycles": total_cycles, "seconds": round(dt, 2)}, f)
    finish(dut, drv, "random")


# ---------------------------------------------------------------- cycles

@cocotb.test()
async def test_cycle_jobs(dut):
    """Run the GEMM jobs listed in SA_JOBS_IN under a bandwidth limit and record
    measured cycles for cycle model validation."""
    drv = await setup(dut)
    jobs_path = os.environ.get("SA_JOBS_IN")
    if not jobs_path:
        return
    jobs = json.loads(open(jobs_path).read())
    rows = []
    for j in jobs:
        g = np.random.default_rng(j["seed"])
        A = g.integers(-128, 128, (j["M"], j["K"]), dtype=np.int8)
        B = g.integers(-128, 128, (j["K"], j["N"]), dtype=np.int8)
        bw = float(j["bw"]) if j["bw"] is not None else float("inf")
        pol = Policy("bw", bw=bw) if bw != float("inf") else Policy("full")
        cycles = await check(drv, A, B, j["buf"], pol, f"job {j['id']}")
        rows.append({**j, "measured": cycles})
    with open(os.environ["SA_CYCLES_OUT"], "w") as f:
        json.dump(rows, f)
    finish(dut, drv, "cycles")
