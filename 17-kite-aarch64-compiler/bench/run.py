#!/usr/bin/env python3
"""Benchmark harness.

For each benchmark NAME in bench/, builds
  kite-O0  (no IR optimisation, stack slots)
  kite-O1  (IR optimisation, stack slots)
  kite-O2  (IR optimisation + linear-scan register allocation)
  c-O0     (cc -O0 on NAME.c)
  c-O2     (cc -O2 on NAME.c)
checks that all five print the same output, then runs each RUNS times
and records wall-clock time. Writes:
  results/bench_raw.csv       every single run
  results/bench_summary.txt   median / min / max per configuration
  results/ir_counts.txt       static IR instruction counts, -O0 vs -O1

Usage: python3 bench/run.py [RUNS]
"""
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "bench")
RESULTS = os.path.join(ROOT, "results")
BUILD = os.path.join(BENCH, "build")
KITEC = os.path.join(ROOT, "_build", "default", "bin", "main.exe")
NAMES = ["fib", "sieve", "loops", "matmul"]
CONFIGS = ["kite-O0", "kite-O1", "kite-O2", "c-O0", "c-O2"]


def sh(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def build(name):
    exes = {}
    src = os.path.join(BENCH, name + ".kite")
    for lvl in ["0", "1", "2"]:
        exe = os.path.join(BUILD, f"{name}_kite_O{lvl}")
        sh([KITEC, "build", src, f"-O{lvl}", "-o", exe])
        exes[f"kite-O{lvl}"] = exe
    for lvl in ["0", "2"]:
        exe = os.path.join(BUILD, f"{name}_c_O{lvl}")
        sh(["cc", f"-O{lvl}", os.path.join(BENCH, name + ".c"), "-o", exe])
        exes[f"c-O{lvl}"] = exe
    return exes


def ir_count(src, lvl):
    out = sh([KITEC, "ir", src, f"-O{lvl}"])
    last = [l for l in out.splitlines() if l.startswith("// ")][-1]
    return int(last.split()[1])


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    raw = ["benchmark,config,run,seconds"]
    summary = {}
    for name in NAMES:
        exes = build(name)
        outputs = {c: sh([exes[c]]) for c in CONFIGS}  # also a warm-up run
        ref = outputs["c-O0"]
        for c in CONFIGS:
            if outputs[c] != ref:
                sys.exit(f"{name}: {c} printed {outputs[c]!r}, expected {ref!r}")
        for c in CONFIGS:
            times = []
            for r in range(runs):
                t0 = time.perf_counter()
                subprocess.run([exes[c]], check=True, stdout=subprocess.DEVNULL)
                dt = time.perf_counter() - t0
                times.append(dt)
                raw.append(f"{name},{c},{r + 1},{dt:.4f}")
            summary[(name, c)] = times
            print(f"{name:7s} {c:8s} median {statistics.median(times):.3f}s", flush=True)

    with open(os.path.join(RESULTS, "bench_raw.csv"), "w") as f:
        f.write("\n".join(raw) + "\n")

    cc_version = sh(["cc", "--version"]).splitlines()[0]
    lines = [
        "Kite benchmark summary (wall-clock seconds)",
        f"machine: {platform.machine()} {platform.platform()}, cpu: Apple M4 Pro",
        f"cc: {cc_version}",
        f"runs per configuration: {runs} (after one warm-up run)",
        "",
        f"{'benchmark':10s}{'config':10s}{'median':>9s}{'min':>9s}{'max':>9s}{'vs c-O2':>9s}",
    ]
    for name in NAMES:
        base = statistics.median(summary[(name, "c-O2")])
        for c in CONFIGS:
            t = summary[(name, c)]
            med = statistics.median(t)
            lines.append(f"{name:10s}{c:10s}{med:9.3f}{min(t):9.3f}{max(t):9.3f}{med / base:8.2f}x")
        lines.append("")
    lines.append("speedups from the Kite optimisation levels (median time ratio):")
    for name in NAMES:
        m = {c: statistics.median(summary[(name, c)]) for c in CONFIGS}
        lines.append(
            f"  {name:8s} O0/O1 = {m['kite-O0'] / m['kite-O1']:.2f}x   O1/O2 = {m['kite-O1'] / m['kite-O2']:.2f}x"
            f"   O0/O2 = {m['kite-O0'] / m['kite-O2']:.2f}x   kite-O2 vs c-O0 = {m['kite-O2'] / m['c-O0']:.2f}x"
        )
    with open(os.path.join(RESULTS, "bench_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # static IR sizes for benchmarks and every test program
    rows = [f"{'program':28s}{'IR -O0':>8s}{'IR -O1':>8s}{'change':>9s}"]
    progs = [os.path.join(BENCH, n + ".kite") for n in NAMES]
    tdir = os.path.join(ROOT, "tests", "programs")
    progs += sorted(os.path.join(tdir, f) for f in os.listdir(tdir) if f.endswith(".kite"))
    tot0 = tot1 = 0
    for p in progs:
        a, b = ir_count(p, 0), ir_count(p, 1)
        tot0 += a
        tot1 += b
        label = os.path.relpath(p, ROOT).replace("tests/programs/", "t/").replace("bench/", "bench/")
        rows.append(f"{label:28s}{a:8d}{b:8d}{(b - a) / a * 100:8.1f}%")
    rows.append(f"{'TOTAL':28s}{tot0:8d}{tot1:8d}{(tot1 - tot0) / tot0 * 100:8.1f}%")
    with open(os.path.join(RESULTS, "ir_counts.txt"), "w") as f:
        f.write("Static IR instruction counts (instructions + block terminators)\n\n")
        f.write("\n".join(rows) + "\n")
    print(f"IR total: O0 {tot0}, O1 {tot1}")


if __name__ == "__main__":
    main()
