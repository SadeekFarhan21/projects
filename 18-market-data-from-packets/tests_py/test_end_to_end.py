"""End-to-end test on the checked-in slice (testdata/): C++ decode, Parquet, quality suite.

Requires the C++ build (cmake --build build). The slice is the first minutes of the
20260918 TOPS and DEEP captures cut with mdp_slice, see scripts/make_test_slice.sh.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "build" / "mdp_decode"
TOPS = ROOT / "testdata" / "slice_tops.pcap.gz"
DEEP = ROOT / "testdata" / "slice_deep.pcap.gz"
EXPECTED = ROOT / "testdata" / "expected.json"

sys.path.insert(0, str(ROOT / "python"))
from mdq import quality, to_parquet  # noqa: E402
from mdq.schema import DTYPES  # noqa: E402


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    if not DECODER.exists():
        pytest.fail("build/mdp_decode missing: run cmake -S . -B build -G Ninja && cmake --build build")
    tmp = tmp_path_factory.mktemp("e2e")
    work, pq_root = tmp / "work", tmp / "parquet"
    for feed, src in (("TOPS", TOPS), ("DEEP", DEEP)):
        subprocess.run([str(DECODER), str(src), "--out", str(work / feed), "--quiet"], check=True)
        to_parquet.convert(work / feed, pq_root / feed.lower(), workers=2, rows_per_group=50_000)
    suite = quality.Suite(pq_root, buckets=2, threads=2, memory="2GB")
    results = {r["check"]: r for r in suite.run()}
    return {"work": work, "parquet": pq_root, "results": results}


def test_all_quality_checks_pass(pipeline):
    failed = {k: v["summary"] for k, v in pipeline["results"].items() if v["status"] != "pass"}
    assert not failed, json.dumps(failed, indent=1, default=str)[:3000]


def test_row_counts_match_snapshot(pipeline):
    expected = json.loads(EXPECTED.read_text())
    got = {}
    for feed in ("tops", "deep"):
        for p in sorted((pipeline["parquet"] / feed).glob("*.parquet")):
            got[f"{feed}.{p.stem}"] = pq.ParquetFile(p).metadata.num_rows
    assert got == expected["row_counts"]


def test_every_message_lands_in_exactly_one_table(pipeline):
    for feed in ("TOPS", "DEEP"):
        m = json.loads((pipeline["work"] / feed / "manifest.json").read_text())
        rows = sum(v["rows"] for k, v in m["tables"].items() if k not in ("segments", "bbo_from_deep"))
        assert rows == m["stats"]["messages"]
        assert m["stats"]["framing_errors"] == 0
        assert m["stats"]["malformed_messages"] == 0
        assert m["tables"]["unknown_messages"]["rows"] == 0
        assert m["capture_format"] == "pcap"  # the slice is re-written as classic nanosecond pcap


def test_schema_types_and_units(pipeline):
    con = duckdb.connect()
    q = pipeline["parquet"] / "tops" / "quotes.parquet"
    types = dict(con.execute(f"DESCRIBE SELECT * FROM '{q}'").fetchall()[i][:2] for i in range(12))
    assert types["ts"] == "TIMESTAMP WITH TIME ZONE"
    assert types["bid_price"] == "BIGINT"
    assert types["symbol"] == "VARCHAR"
    # Prices are 1e-4 dollars: a nonzero quote on a real stock is between $0.0001 and $1,000,000.
    lo, hi = con.execute(f"SELECT min(bid_price), max(bid_price) FROM '{q}' WHERE bid_size > 0").fetchone()
    assert 0 < lo and hi < 10_000_000_000
    # Timestamps fall on the capture date.
    d = con.execute(f"SELECT DISTINCT CAST(ts AT TIME ZONE 'America/New_York' AS DATE) FROM '{q}'").fetchall()
    assert [str(x[0]) for x in d] == ["2026-09-18"]


def test_dtypes_match_cpp_record_sizes(pipeline):
    m = json.loads((pipeline["work"] / "DEEP" / "manifest.json").read_text())
    for table, v in m["tables"].items():
        assert DTYPES[table].itemsize == v["record_size"], table


def test_bbo_matches_book_rebuilt_in_python(pipeline):
    """Rebuild the DEEP book for a few symbols with plain Python dicts from deep_levels and compare
    against bbo_from_deep (independent implementation of the event-complete rule)."""
    con = duckdb.connect()
    root = pipeline["parquet"] / "deep"
    syms = [r[0] for r in con.execute(
        f"SELECT symbol FROM '{root}/deep_levels.parquet' GROUP BY 1 ORDER BY count(*) DESC LIMIT 5").fetchall()]
    for sym in syms:
        lv = con.execute(f"SELECT seq, side, price, size, event_complete FROM '{root}/deep_levels.parquet' "
                         f"WHERE symbol = '{sym}' ORDER BY seq").fetchall()
        bids, asks = {}, {}
        last = (0, 0, 0, 0)
        emitted = []
        for seq, side, px, sz, done in lv:
            book = bids if side == "B" else asks
            if sz == 0:
                book.pop(px, None)
            else:
                book[px] = sz
            if done:
                bb = max(bids) if bids else None
                ba = min(asks) if asks else None
                now = (bb or 0, bids.get(bb, 0), ba or 0, asks.get(ba, 0))
                if now != last:
                    emitted.append((seq, *now))
                    last = now
        got = con.execute(f"SELECT seq, bid_price, bid_size, ask_price, ask_size FROM "
                          f"'{root}/bbo_from_deep.parquet' WHERE symbol = '{sym}' ORDER BY seq").fetchall()
        assert got == emitted, sym


def test_parquet_roundtrip_of_binary_rows(tmp_path):
    """to_parquet on a synthetic binary table preserves every field."""
    work = tmp_path / "w"
    work.mkdir()
    dt = DTYPES["trades"]
    rec = np.zeros(1000, dtype=dt)
    rng = np.random.default_rng(1)
    rec["ts"] = 1789729540000000000 + np.arange(1000) * 1000
    rec["seq"] = np.arange(1, 1001)
    rec["symbol"] = np.frombuffer(b"SPY     ", dtype="<u8")[0]
    rec["price"] = rng.integers(1, 10**8, 1000)
    rec["trade_id"] = rng.integers(0, 2**62, 1000)
    rec["size"] = rng.integers(1, 10**6, 1000)
    rec["flags"] = rng.choice([0, 0x80, 0x20, 0xa8], 1000)
    rec["msg_type"] = ord("T")
    rec.tofile(work / "trades.bin")
    manifest = {"feed": "DEEP1.0", "tables": {"trades": {"rows": 1000, "record_size": dt.itemsize}}}
    (work / "manifest.json").write_text(json.dumps(manifest))
    to_parquet.convert(work, tmp_path / "out", workers=1, rows_per_group=300)
    t = pq.read_table(tmp_path / "out" / "trades.parquet").to_pydict()
    assert t["seq"] == list(range(1, 1001))
    assert t["price"] == rec["price"].tolist()
    assert t["trade_id"] == rec["trade_id"].tolist()
    assert set(t["symbol"]) == {"SPY"}
    assert set(t["msg_type"]) == {"T"}
    assert t["iso"] == ((rec["flags"] & 0x80) != 0).tolist()
    assert t["odd_lot"] == ((rec["flags"] & 0x20) != 0).tolist()
    ts = pq.read_table(tmp_path / "out" / "trades.parquet").column("ts").cast(pa.int64()).to_pylist()
    assert ts == rec["ts"].tolist()
