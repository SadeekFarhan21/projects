"""Thin local data layer: Binance public archive -> one tidy parquet file.

This is deliberately small. When project 08 (the shared market-data layer)
lands, `load_panel` is the only function the rest of the code calls, so the
migration is to reimplement `load_panel` on top of project 08 and delete the
downloader.
"""
from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PUBLIC = "https://data.binance.vision"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PANEL_PATH = DATA_DIR / "processed" / "klines_1d.parquet"


def _list(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """List an S3 prefix, following pagination. Returns (common_prefixes, keys)."""
    prefixes, keys, marker = [], [], ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = delimiter
        r = requests.get(S3, params=params, timeout=30)
        r.raise_for_status()
        txt = r.text
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", txt)[1:] if delimiter else []
        keys += re.findall(r"<Key>([^<]+)</Key>", txt)
        if "<IsTruncated>true</IsTruncated>" not in txt:
            break
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", txt)
        marker = nxt.group(1) if nxt else (keys[-1] if keys else prefixes[-1])
    # the first <Prefix> tag is the request prefix itself, skipped above
    return prefixes, keys


def list_usdt_symbols() -> list[str]:
    prefixes, _ = _list("data/spot/monthly/klines/")
    syms = sorted({p.rstrip("/").split("/")[-1] for p in prefixes})
    return [s for s in syms if s.endswith("USDT") and len(s) > 4]


def parse_kline_csv(raw: bytes, symbol: str) -> pd.DataFrame:
    """Parse one Binance kline CSV. Handles an optional header row and the
    2025 switch from millisecond to microsecond timestamps."""
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():  # header row present
        df = df.iloc[1:]
    t = pd.to_numeric(df["open_time"]).astype("int64").to_numpy()
    # Binance spot archives use microseconds from 2025-01 onward.
    unit_us = t > 10**14
    t_ms = np.where(unit_us, t // 1000, t)
    out = pd.DataFrame({
        "date": pd.to_datetime(t_ms, unit="ms").normalize(),
        "symbol": symbol,
        "open": pd.to_numeric(df["open"]).to_numpy(float),
        "high": pd.to_numeric(df["high"]).to_numpy(float),
        "low": pd.to_numeric(df["low"]).to_numpy(float),
        "close": pd.to_numeric(df["close"]).to_numpy(float),
        "volume": pd.to_numeric(df["volume"]).to_numpy(float),
        "quote_volume": pd.to_numeric(df["quote_volume"]).to_numpy(float),
    })
    return out


def _fetch(key: str, raw_dir: Path) -> Path:
    dest = raw_dir / Path(key).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(f"{PUBLIC}/{key}", timeout=60)
    r.raise_for_status()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(dest)
    return dest


def download(start: str = "2018-01", end: str = "2026-08", workers: int = 32,
             symbols: list[str] | None = None, verbose: bool = True) -> Path:
    """Download all monthly 1d kline zips for USDT pairs in [start, end] and
    write the combined panel to PANEL_PATH."""
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    symbols = symbols or list_usdt_symbols()
    if verbose:
        print(f"{len(symbols)} USDT symbols in archive")

    def keys_for(sym: str) -> list[str]:
        _, keys = _list(f"data/spot/monthly/klines/{sym}/1d/", delimiter=None)
        out = []
        for k in keys:
            m = re.search(r"-1d-(\d{4}-\d{2})\.zip$", k)
            if m and start <= m.group(1) <= end:
                out.append(k)
        return out

    keys: list[str] = []
    with ThreadPoolExecutor(workers) as ex:
        for ks in ex.map(keys_for, symbols):
            keys += ks
    if verbose:
        print(f"{len(keys)} monthly files to fetch (cached ones skipped)")

    paths: list[Path] = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch, k, raw_dir) for k in keys]
        for i, f in enumerate(as_completed(futs)):
            paths.append(f.result())
            if verbose and (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(keys)}")

    frames = []
    for p in paths:
        sym = p.name.split("-1d-")[0]
        with zipfile.ZipFile(p) as z:
            frames.append(parse_kline_csv(z.read(z.namelist()[0]), sym))
    panel = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(["date", "symbol"])
             .sort_values(["date", "symbol"], ignore_index=True))
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    if verbose:
        print(f"wrote {len(panel):,} rows, {panel.symbol.nunique()} symbols -> {PANEL_PATH}")
    return PANEL_PATH


def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    """Long panel with one row per (date, symbol). The single entry point the
    research code uses; swap this for project 08 later."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run python scripts/download_data.py`")
    return pd.read_parquet(path)
