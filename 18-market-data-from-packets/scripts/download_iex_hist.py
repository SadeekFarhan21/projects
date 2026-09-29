"""Download IEX HIST pcap.gz files for one date using parallel HTTP range requests.

Usage:
    uv run python scripts/download_iex_hist.py --date 20260918 --feeds TOPS DEEP
    uv run python scripts/download_iex_hist.py --date 20260918 --feeds TOPS --max-bytes 500000000

IEX publishes HIST data T+1 at https://iextrading.com/api/1.0/hist. Each entry
points at a Google Cloud Storage object that supports byte ranges, so we split
each file into chunks and fetch them concurrently with curl, then concatenate.
--max-bytes downloads only a prefix of the gzip stream (useful for a quick
look; gzip decodes a prefix fine and the decoder stops at the truncated tail).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

API = "https://iextrading.com/api/1.0/hist?date={date}"


def list_files(date: str) -> list[dict]:
    with urllib.request.urlopen(API.format(date=date), timeout=60) as r:
        data = json.load(r)
    if isinstance(data, dict):
        data = data.get(date, [])
    return data


def fetch_range(url: str, start: int, end: int, out: Path) -> int:
    expected = end - start + 1
    for attempt in range(5):
        if out.exists() and out.stat().st_size == expected:
            return expected
        subprocess.run(
            ["curl", "-s", "-L", "--retry", "5", "-r", f"{start}-{end}", "-o", str(out), url],
            check=False,
        )
        if out.exists() and out.stat().st_size == expected:
            return expected
    raise RuntimeError(f"range {start}-{end} failed for {url}")


def download(entry: dict, dest_dir: Path, chunk: int, workers: int, max_bytes: int | None) -> Path:
    url = entry["link"]
    size = int(entry["size"])
    total = min(size, max_bytes) if max_bytes else size
    name = f"{entry['date']}_IEXTP1_{entry['feed']}{entry['version']}.pcap.gz"
    if max_bytes:
        name = name.replace(".pcap.gz", f".prefix{total}.pcap.gz")
    final = dest_dir / name
    if final.exists() and final.stat().st_size == total:
        print(f"have {final} ({total} bytes)")
        return final
    parts_dir = dest_dir / (name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(s, min(s + chunk, total) - 1) for s in range(0, total, chunk)]
    print(f"{name}: {total/1e9:.2f} GB in {len(ranges)} chunks, {workers} workers", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {
            ex.submit(fetch_range, url, s, e, parts_dir / f"{i:06d}"): i
            for i, (s, e) in enumerate(ranges)
        }
        for f in cf.as_completed(futs):
            done += f.result()
            if len(ranges) >= 10 and futs[f] % max(1, len(ranges) // 10) == 0:
                print(f"  {name}: {done/1e9:.2f}/{total/1e9:.2f} GB", flush=True)
    tmp = final.with_suffix(".tmp")
    with open(tmp, "wb") as w:
        for i in range(len(ranges)):
            p = parts_dir / f"{i:06d}"
            with open(p, "rb") as r:
                while b := r.read(1 << 24):
                    w.write(b)
    os.replace(tmp, final)
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    print(f"wrote {final}")
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--feeds", nargs="+", default=["TOPS", "DEEP"])
    ap.add_argument("--dest", default="data/raw")
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-bytes", type=int, default=None)
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    entries = [e for e in list_files(args.date) if e["feed"] in args.feeds]
    if not entries:
        print(f"no files for {args.date} feeds {args.feeds}", file=sys.stderr)
        return 1
    for e in entries:
        download(e, dest, args.chunk_mb << 20, args.workers, args.max_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
