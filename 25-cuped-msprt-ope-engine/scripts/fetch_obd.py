"""Populate data/obd with the Open Bandit Dataset sample and the ZOZOTOWN BTS prior.

Default: the 10,000-row-per-policy sample that ships with obp (the smallest meaningful
version). Order of sources:
  1. an installed obp package (uv sync --group crosscheck), copied locally, no network
  2. the same files from the zr-obp GitHub repository

The full dataset (about 26 million rounds, a multi-gigabyte zip) is a later milestone:
    curl -LO https://research.zozo.com/data_release/open_bandit_dataset.zip
    unzip open_bandit_dataset.zip -d data/obd_full
Then pass --data-dir data/obd_full/open_bandit_dataset to experiments/07_obd_ope.py.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/st-tech/zr-obp/master"
FILES = [f"{p}/{c}/{name}" for p in ("random", "bts") for c in ("all",) for name in (f"{c}.csv", "item_context.csv")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "obd")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        import obp  # noqa: F401

        pkg = Path(obp.__file__).parent
        for f in FILES:
            dst = args.out / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pkg / "dataset" / "obd" / f, dst)
        shutil.copy(pkg / "policy" / "conf" / "prior_bts.yaml", args.out / "prior_bts.yaml")
        print(f"copied OBD sample from {pkg}")
    except ImportError:
        for f in FILES:
            dst = args.out / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{RAW}/obd/{f}", dst)
        urllib.request.urlretrieve(f"{RAW}/obp/policy/conf/prior_bts.yaml", args.out / "prior_bts.yaml")
        print("downloaded OBD sample from GitHub")
    for p in sorted(args.out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
