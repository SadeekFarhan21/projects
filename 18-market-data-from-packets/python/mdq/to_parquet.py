"""Convert mdp_decode binary tables to Parquet.

Usage:
    uv run python -m mdq.to_parquet --work data/work/20260918/TOPS1.6 --out data/parquet/20260918/tops
    uv run python -m mdq.to_parquet --work ... --out ... --delete-bin --workers 6

Reads <work>/manifest.json, memory-maps each <table>.bin with the dtype from
schema.py, and writes <out>/<table>.parquet in row groups of --rows-per-group
rows with zstd compression. Symbols become dictionary-encoded strings, event
times become timestamp[ns, UTC], single-byte enums become 1-char strings.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mdq.schema import CHAR_COLUMNS, DELTA_COLUMNS, DTYPES, FLAG_COLUMNS, TABLE_DOCS, TIMESTAMP_COLUMNS

# 256-entry dictionary for single-byte enum columns.
_CHAR_DICT = pa.array([str(b) if b < 32 else chr(b) for b in range(256)], type=pa.string())
_TS_TYPE = pa.timestamp("ns", tz="UTC")


def _symbol_array(codes: np.ndarray) -> pa.DictionaryArray:
    uniq, inv = np.unique(codes, return_inverse=True)
    names = [int(u).to_bytes(8, "little").rstrip(b" \x00").decode("ascii", "replace") for u in uniq]
    return pa.DictionaryArray.from_arrays(pa.array(inv.astype(np.int32)), pa.array(names, type=pa.string()))


def _char_array(col: np.ndarray) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(col.astype(np.int16)), _CHAR_DICT)


def _to_arrow(table: str, rec: np.ndarray, feed: str) -> pa.Table:
    cols: dict[str, pa.Array] = {}
    n = len(rec)
    cols["feed"] = pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, np.int8)), pa.array([feed]))
    for name in rec.dtype.names:
        a = rec[name]
        if name in TIMESTAMP_COLUMNS:
            cols[name] = pa.array(a.astype("<i8", copy=False).view("datetime64[ns]"), type=_TS_TYPE)
        elif name == "symbol":
            cols[name] = _symbol_array(a)
        elif name in CHAR_COLUMNS:
            cols[name] = _char_array(a)
        elif name == "reason":
            cols[name] = pa.array([bytes(x).rstrip(b" \x00").decode("ascii", "replace") for x in a], type=pa.string())
        elif name.endswith("_size") or name in ("size", "paired_shares", "imbalance_shares", "round_lot"):
            cols[name] = pa.array(a.astype(np.int64))
        else:
            cols[name] = pa.array(a)
    for flag, (src, mask) in FLAG_COLUMNS.get(table, {}).items():
        cols[flag] = pa.array((rec[src] & mask) != 0)
    return pa.table(cols)


def convert_table(work: Path, out: Path, table: str, rows: int, record_size: int, feed: str,
                  rows_per_group: int, delete_bin: bool) -> dict:
    dtype = DTYPES[table]
    if dtype.itemsize != record_size:
        raise ValueError(f"{table}: numpy itemsize {dtype.itemsize} != C++ record size {record_size}")
    src = work / f"{table}.bin"
    dst = out / f"{table}.parquet"
    t0 = time.perf_counter()
    size = src.stat().st_size if src.exists() else 0
    if size != rows * record_size:
        raise ValueError(f"{table}: {src} has {size} bytes, manifest says {rows} rows x {record_size}")
    mm = np.memmap(src, dtype=dtype, mode="r") if rows else np.zeros(0, dtype=dtype)
    schema = _to_arrow(table, np.zeros(0, dtype=dtype), feed).schema
    schema = schema.with_metadata({"table": table, "doc": TABLE_DOCS.get(table, ""), "feed": feed,
                                   "price_unit": "1e-4 USD", "time_unit": "ns since epoch UTC"})
    dict_cols = [c for c in schema.names if c not in DELTA_COLUMNS]
    enc = {c: "DELTA_BINARY_PACKED" for c in schema.names if c in DELTA_COLUMNS}
    tmp = dst.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3, use_dictionary=dict_cols,
                          column_encoding=enc, write_statistics=True) as w:
        for start in range(0, rows, rows_per_group):
            chunk = np.asarray(mm[start:start + rows_per_group])
            w.write_table(_to_arrow(table, chunk, feed).cast(schema), row_group_size=rows_per_group)
    os.replace(tmp, dst)
    del mm
    if delete_bin and src.exists():
        src.unlink()
    return {"table": table, "rows": rows, "bin_bytes": size, "parquet_bytes": dst.stat().st_size,
            "seconds": round(time.perf_counter() - t0, 2)}


def convert(work: Path, out: Path, workers: int = 4, rows_per_group: int = 2_000_000,
            delete_bin: bool = False) -> dict:
    manifest = json.loads((work / "manifest.json").read_text())
    feed = "TOPS" if manifest["feed"].startswith("TOPS") else "DEEP"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(work / "manifest.json", out / "manifest.json")
    jobs = [(t, v["rows"], v["record_size"]) for t, v in manifest["tables"].items()
            if (work / f"{t}.bin").exists() or v["rows"] == 0]
    # Biggest first so the pool stays busy.
    jobs.sort(key=lambda j: -j[1] * j[2])
    results = []
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(convert_table, work, out, t, r, rs, feed, rows_per_group, delete_bin) for t, r, rs in jobs]
        for f in cf.as_completed(futs):
            res = f.result()
            results.append(res)
            print(f"  {feed} {res['table']}: {res['rows']:,} rows, {res['bin_bytes']/1e6:.1f} MB bin -> "
                  f"{res['parquet_bytes']/1e6:.1f} MB parquet in {res['seconds']}s", flush=True)
    summary = {"feed": feed, "work": str(work), "out": str(out), "seconds": round(time.perf_counter() - t0, 2),
               "tables": sorted(results, key=lambda r: r["table"])}
    (out / "parquet_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rows-per-group", type=int, default=2_000_000)
    ap.add_argument("--delete-bin", action="store_true")
    a = ap.parse_args()
    s = convert(a.work, a.out, a.workers, a.rows_per_group, a.delete_bin)
    print(json.dumps({k: v for k, v in s.items() if k != "tables"}))


if __name__ == "__main__":
    main()
