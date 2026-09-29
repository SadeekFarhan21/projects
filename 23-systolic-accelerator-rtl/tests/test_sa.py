"""pytest entry points: build sa_top with Verilator and run the cocotb tests.

Environment knobs
  SA_REGRESSION_GEMMS   random GEMMs per shard in test_random_regression (default 25)
  SA_REGRESSION_SHARDS  shards per array size (default 2)
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
TESTS = ROOT / "tests"
BUILD = ROOT / "build"
SOURCES = [RTL / f for f in ("sa_pe.sv", "sa_delay.sv", "sa_sram.sv", "sa_array.sv", "sa_top.sv")]


def build(n: int, adepth: int, fault: bool = False):
    """Build (or reuse) a Verilator model. A file lock serialises concurrent
    xdist workers that want the same configuration."""
    tag = f"sim_n{n}_a{adepth}" + ("_fault" if fault else "")
    bdir = BUILD / tag
    BUILD.mkdir(exist_ok=True)
    with open(BUILD / f"{tag}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        runner = get_runner("verilator")
        runner.build(
            sources=SOURCES,
            hdl_toplevel="sa_top",
            parameters={"N": n, "ADEPTH": adepth},
            defines={"SA_FAULT_OUT_GLITCH": 1} if fault else {},
            build_args=["-Wno-fatal", "-O3", "--x-assign", "unique", "--x-initial", "unique"],
            build_dir=bdir,
            waves=True,            # tracing compiled in, only switched on for replays
            always=False,
        )
    return runner, bdir


def run(n: int, adepth: int, testcase: str, env: dict | None = None,
        fault: bool = False, waves: bool = False, tag: str = "",
        log_file: Path | None = None) -> None:
    runner, bdir = build(n, adepth, fault)
    tdir = bdir / f"run_{testcase}{tag}"
    tdir.mkdir(parents=True, exist_ok=True)
    extra = {"SA_N": str(n), "SA_ADEPTH": str(adepth), "PYTHONPATH": f"{TESTS}:{ROOT}"}
    extra.update(env or {})
    runner.test(
        test_module="tb_sa",
        hdl_toplevel="sa_top",
        hdl_toplevel_lang="verilog",
        testcase=testcase,
        build_dir=bdir,
        test_dir=tdir,
        extra_env=extra,
        waves=waves,
        results_xml=str(tdir / "results.xml"),
        log_file=log_file,
    )


def cov_path(name: str) -> str:
    d = BUILD / "coverage"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / name)


def run_with_replay(n: int, adepth: int, testcase: str, env: dict, tag: str) -> None:
    """Run a random test; on failure replay the failing case with a VCD dump."""
    fail_file = BUILD / f"fail_{testcase}{tag}.json"
    env = {**env, "SA_FAIL_OUT": str(fail_file)}
    fail_file.unlink(missing_ok=True)
    try:
        run(n, adepth, testcase, env, tag=tag)
    except SystemExit:
        if fail_file.exists():
            case = json.loads(fail_file.read_text())["case"]
            replay_env = {**env, "SA_ONLY_CASE": str(case)}
            try:
                run(n, adepth, testcase, replay_env, waves=True, tag=tag + "_replay")
            except SystemExit:
                pass
            wdir = BUILD / "waves"
            wdir.mkdir(exist_ok=True)
            src = BUILD / f"sim_n{n}_a{adepth}" / f"run_{testcase}{tag}_replay" / "dump.vcd"
            if src.exists():
                shutil.copy(src, wdir / f"{testcase}{tag}_case{case}.vcd")
        raise


# ------------------------------------------------------------------ directed

@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("testcase", ["test_identity", "test_all_ones",
                                      "test_int8_extremes", "test_weight_reuse_and_chunks"])
def test_directed(n, testcase):
    run(n, 64, testcase, {"SA_SEED": "11", "SA_COV_OUT": cov_path(f"n{n}_{testcase}")})


# ------------------------------------------------------------------ random

SHARDS = int(os.environ.get("SA_REGRESSION_SHARDS", "2"))
GEMMS = int(os.environ.get("SA_REGRESSION_GEMMS", "25"))


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("shard", range(SHARDS))
def test_random_regression(n, shard):
    seed = 1000 * n + shard
    run_with_replay(n, 64, "test_random_gemms",
                    {"SA_SEED": str(seed), "SA_NUM_GEMMS": str(GEMMS),
                     "SA_COV_OUT": cov_path(f"n{n}_random_s{shard}")},
                    tag=f"_s{shard}")


# ------------------------------------------------------------------ negative

def test_assertion_fires_on_injected_fault():
    """Build with SA_FAULT_OUT_GLITCH, which flips o_data while the sink
    stalls. The run must fail and the RTL handshake assertion A1 must be the
    thing that reports it, proving the checks are live."""
    log = BUILD / "fault_injection.log"
    log.unlink(missing_ok=True)
    with pytest.raises(SystemExit):
        run(8, 64, "test_random_gemms",
            {"SA_SEED": "5", "SA_NUM_GEMMS": "20"}, fault=True, log_file=log)
    assert "SVA A1 FAIL" in log.read_text()
