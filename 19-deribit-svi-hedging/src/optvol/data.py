"""Paths and cached loading of the reduced tardis day files."""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

import polars as pl

from . import chain

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
DEFAULT_DAY = "2026-09-01"


def day_path(day: str = DEFAULT_DAY) -> Path:
    return RAW / f"deribit_options_chain_{day.replace('-', '')}_1m.csv.gz"


@functools.cache
def load_day(day: str = DEFAULT_DAY) -> pl.DataFrame:
    p = day_path(day)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/download_tardis.sh {day}")
    return chain.load_minute_file(str(p))


def ts_us(day: str, hh: int, mm: int = 0) -> int:
    d = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
    return int((d + dt.timedelta(hours=hh, minutes=mm)).timestamp() * 1_000_000)


def clean_snapshot(day: str, t_us: int):
    df = load_day(day)
    c, counts = chain.clean(chain.snapshot(df, t_us))
    fwd = chain.fit_forwards(c)
    return chain.attach_iv(c, fwd), fwd, counts
