"""End-to-end pipeline: decode TOPS and DEEP captures, convert to Parquet, run the quality suite.

Usage:
    uv run python scripts/pipeline.py --tops data/raw/20260918_IEXTP1_TOPS1.6.pcap.gz \
        --deep data/raw/20260918_IEXTP1_DEEP1.0.pcap.gz --name 20260918 [--until-ns NS] [--keep-bin]

Writes data/work/<name>/{TOPS,DEEP}/ (binary tables, deleted after conversion unless
--keep-bin), data/parquet/<name>/{tops,deep}/*.parquet, results/quality_<name>.{json,md}
and results/pipeline_<name>.json with stage timings and sizes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mdq import quality, to_parquet  # noqa: E402


def decode(tops: Path, deep: Path, work: Path, until_ns: int | None, decoder: Path) -> dict:
    procs = {}
    t0 = time.perf_counter()
    for feed, src in (("TOPS", tops), ("DEEP", deep)):
        out = work / feed
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(decoder), str(src), "--out", str(out)]
        if until_ns:
            cmd += ["--until-ns", str(until_ns)]
        log = open(work / f"{feed}.log", "w")
        procs[feed] = (subprocess.Popen(cmd, stderr=log, stdout=log), log)
    for feed, (p, log) in procs.items():
        if p.wait() != 0:
            raise RuntimeError(f"mdp_decode failed for {feed}, see {work / (feed + '.log')}")
        log.close()
    return {"decode_wall_seconds": round(time.perf_counter() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tops", required=True, type=Path)
    ap.add_argument("--deep", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--until-ns", type=int, default=None)
    ap.add_argument("--keep-bin", action="store_true")
    ap.add_argument("--skip-decode", action="store_true", help="reuse existing binary tables")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--decoder", type=Path, default=ROOT / "build" / "mdp_decode")
    a = ap.parse_args()

    work = a.data / "work" / a.name
    pq_root = a.data / "parquet" / a.name
    timings: dict = {"name": a.name, "tops_input": str(a.tops), "deep_input": str(a.deep),
                     "tops_input_bytes": a.tops.stat().st_size, "deep_input_bytes": a.deep.stat().st_size}
    if not a.skip_decode:
        timings.update(decode(a.tops, a.deep, work, a.until_ns, a.decoder))
    t0 = time.perf_counter()
    conv = {}
    for feed in ("TOPS", "DEEP"):
        conv[feed] = to_parquet.convert(work / feed, pq_root / feed.lower(), workers=a.workers,
                                        delete_bin=not a.keep_bin)
    timings["parquet_wall_seconds"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    suite = quality.Suite(pq_root, buckets=a.buckets)
    results = suite.run()
    meta = {"root": str(pq_root), "date": a.name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "common_event_window_end_ns": suite.t_end}
    quality.write_report(results, meta, a.results / f"quality_{a.name}")
    timings["quality_wall_seconds"] = round(time.perf_counter() - t0, 2)
    for feed in ("TOPS", "DEEP"):
        m = json.loads((pq_root / feed.lower() / "manifest.json").read_text())
        timings[feed.lower()] = {
            "capture_bytes_uncompressed": m["capture_bytes"],
            "packets": m["stats"]["packets"],
            "messages": m["stats"]["messages"],
            "decode_seconds": m["decode_seconds"],
            "min_event_ts": m["min_event_ts"],
            "max_event_ts": m["max_event_ts"],
            "input_truncated": m["input_truncated"],
            "parquet_bytes": sum(t["parquet_bytes"] for t in conv[feed]["tables"]),
            "bin_bytes": sum(t["bin_bytes"] for t in conv[feed]["tables"]),
            "tables": {t["table"]: {"rows": t["rows"], "parquet_bytes": t["parquet_bytes"]}
                       for t in conv[feed]["tables"]},
        }
    timings["checks"] = {r["check"]: r["status"] for r in results}
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"pipeline_{a.name}.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings["checks"]))
    return 0 if all(v == "pass" for v in timings["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
