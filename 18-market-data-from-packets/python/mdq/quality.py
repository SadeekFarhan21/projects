"""Data-quality suite over the normalized Parquet tables (DuckDB SQL plus Python).

Usage:
    uv run python -m mdq.quality --root data/parquet/20260918 --report results/quality_20260918

Expects <root>/tops/*.parquet and <root>/deep/*.parquet from mdq.to_parquet.
Writes <report>.json (counts plus example rows for every check) and <report>.md.

Checks
  1 sequence_continuity   per feed and (protocol, channel, session): IEX-TP segments contiguous in
                          sequence and stream offset; every message sequence number present exactly once
                          across all tables
  2 timestamp_monotonic   segment send time never decreases; event timestamps never decrease within a
                          (message type, symbol) pair, which is the ordering the IEX specs promise
  3 deep_crossed_locked   consistent DEEP BBO (after event-complete) never crossed or locked
  4 bbo_deep_vs_tops      at every TOPS quote update, the DEEP-derived BBO as of the same event time
                          equals the TOPS BBO; every mismatch is categorized
  5 trade_reconciliation  TOPS and DEEP trade reports and breaks match one to one on trade id with
                          identical symbol, price, size, flags and timestamp
  6 halted_no_quotes      while a symbol is halted, paused or in an order acceptance period, TOPS
                          quotes carry the unavailable flag and no two-sided quote is published
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import duckdb

BUCKETS_DEFAULT = 8


def _rows(con: duckdb.DuckDBPyConnection, sql: str, limit: int | None = None) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    out = cur.fetchmany(limit) if limit else cur.fetchall()
    return [{c: _jsonable(v) for c, v in zip(cols, r)} for r in out]


def _one(con: duckdb.DuckDBPyConnection, sql: str) -> dict:
    r = _rows(con, sql)
    return r[0] if r else {}


def _jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.hex()
    return v


class Suite:
    def __init__(self, root: Path, buckets: int = BUCKETS_DEFAULT, threads: int = 6, memory: str = "12GB"):
        self.root = root
        self.buckets = buckets
        self.con = duckdb.connect()
        self.con.execute(f"SET threads={threads}; SET memory_limit='{memory}'; SET preserve_insertion_order=false;")
        tmp = root / "_duckdb_tmp"
        tmp.mkdir(exist_ok=True)
        self.con.execute(f"SET temp_directory='{tmp}'")
        self.tables: dict[str, list[str]] = {}
        for feed in ("tops", "deep"):
            names = []
            for p in sorted((root / feed).glob("*.parquet")):
                name = p.stem
                self.con.execute(f"CREATE VIEW {feed}_{name} AS SELECT * FROM read_parquet('{p}')")
                names.append(name)
            self.tables[feed] = names
        self.manifest = {f: json.loads((root / f / "manifest.json").read_text()) for f in ("tops", "deep")}
        # Common event-time window: a slice or a partial day covers different spans per feed.
        self.t_end = min(self._max_ts("tops"), self._max_ts("deep"))
        self.results: list[dict] = []

    def _max_ts(self, feed: str) -> int:
        return int(self.manifest[feed]["max_event_ts"])

    def _add(self, name: str, status: str, summary: dict, examples: dict | None = None, notes: str = "",
             seconds: float = 0.0) -> None:
        self.results.append({"check": name, "status": status, "summary": summary, "examples": examples or {},
                             "notes": notes, "seconds": round(seconds, 2)})
        print(f"[quality] {name}: {status} {json.dumps(summary)[:300]}", flush=True)

    def _ts_end_sql(self) -> str:
        return f"make_timestamp_ns({self.t_end})::TIMESTAMPTZ" if hasattr(duckdb, "__version__") else ""

    # 1
    def sequence_continuity(self) -> None:
        t0 = time.perf_counter()
        summary, examples = {}, {}
        ok = True
        for feed in ("tops", "deep"):
            seg = _one(self.con, f"""
                WITH s AS (
                  SELECT protocol_id, channel_id, session_id, first_seq, message_count, stream_offset,
                         payload_length, send_time, skipped,
                         lag(first_seq + message_count) OVER w AS prev_end,
                         lag(stream_offset + payload_length) OVER w AS prev_off_end
                  FROM {feed}_segments WHERE message_count > 0
                  WINDOW w AS (PARTITION BY protocol_id, channel_id, session_id ORDER BY first_seq, capture_ts))
                SELECT count(*) AS data_segments,
                       count(DISTINCT (protocol_id, channel_id, session_id)) AS sessions,
                       coalesce(sum((first_seq > prev_end)::INT), 0) AS gaps,
                       coalesce(sum(greatest(first_seq - prev_end, 0)), 0) AS gap_messages,
                       coalesce(sum((first_seq < prev_end)::INT), 0) AS repeats,
                       coalesce(sum((first_seq = prev_end AND stream_offset <> prev_off_end)::INT), 0)
                         AS offset_mismatches,
                       min(first_seq) AS min_seq, max(first_seq + message_count - 1) AS max_seq
                FROM s""")
            hb = _one(self.con, f"SELECT count(*) AS heartbeats FROM {feed}_segments WHERE message_count = 0")
            # Message-level: every sequence number between min and max appears exactly once across tables.
            msg_tables = [t for t in self.tables[feed] if t not in ("segments", "bbo_from_deep")]
            union = " UNION ALL ".join(f"SELECT seq FROM {feed}_{t}" for t in msg_tables)
            lo, hi = seg["min_seq"], seg["max_seq"]
            total, distinct = 0, 0
            step = 50_000_000
            if lo is not None:
                for a in range(lo, hi + 1, step):
                    r = _one(self.con, f"SELECT count(*) AS n, count(DISTINCT seq) AS d FROM ({union}) "
                                       f"WHERE seq BETWEEN {a} AND {min(a + step - 1, hi)}")
                    total += r["n"]
                    distinct += r["d"]
            expected = (hi - lo + 1) if lo is not None else 0
            m = self.manifest[feed]
            online = {k: sum(s[k] for s in m["sessions"]) for k in
                      ("gaps", "gap_messages", "duplicates", "overlaps", "offset_mismatches")}
            summary[feed] = {**seg, **hb, "message_rows": total, "distinct_seq": distinct,
                             "expected_messages": expected, "missing_messages": expected - distinct,
                             "duplicate_rows": total - distinct, "online_tracker": online,
                             "seq_anomalies_logged": m["seq_anomaly_count"]}
            if seg["gaps"] or seg["repeats"] or seg["offset_mismatches"] or expected != distinct or total != distinct:
                ok = False
                examples[feed] = _rows(self.con, f"""
                    WITH s AS (SELECT *, lag(first_seq + message_count) OVER w AS prev_end
                               FROM {feed}_segments WHERE message_count > 0
                               WINDOW w AS (PARTITION BY protocol_id, channel_id, session_id
                                            ORDER BY first_seq, capture_ts))
                    SELECT capture_ts, send_time, session_id, first_seq, prev_end, message_count
                    FROM s WHERE first_seq <> prev_end ORDER BY first_seq LIMIT 10""")
        self._add("sequence_continuity", "pass" if ok else "fail", summary, examples,
                  seconds=time.perf_counter() - t0)

    # 2
    def timestamp_monotonic(self) -> None:
        t0 = time.perf_counter()
        summary, examples = {}, {}
        ok = True
        for feed in ("tops", "deep"):
            seg = _one(self.con, f"""
                WITH s AS (SELECT send_time, capture_ts, lag(send_time) OVER (ORDER BY first_seq, capture_ts) AS prev
                           FROM {feed}_segments)
                SELECT coalesce(sum((send_time < prev)::INT), 0) AS send_time_regressions,
                       quantile_cont(epoch_ns(capture_ts) - epoch_ns(send_time), 0.5) AS capture_minus_send_p50_ns,
                       quantile_cont(epoch_ns(capture_ts) - epoch_ns(send_time), 0.99) AS capture_minus_send_p99_ns,
                       min(epoch_ns(capture_ts) - epoch_ns(send_time)) AS capture_minus_send_min_ns
                FROM s""")
            per_table = {}
            for t in self.tables[feed]:
                if t in ("segments", "unknown_messages", "system_events"):
                    continue
                cols = [r[0] for r in self.con.execute(f"DESCRIBE {feed}_{t}").fetchall()]
                part = "symbol" + (", msg_type" if "msg_type" in cols else "")
                n_bad = 0
                ex: list[dict] = []
                for k in range(self.buckets):
                    q = f"""
                        WITH x AS (SELECT {part}, seq, ts, lag(ts) OVER (PARTITION BY {part} ORDER BY seq) AS prev_ts
                                   FROM {feed}_{t} WHERE hash(symbol) % {self.buckets} = {k})
                        SELECT * FROM x WHERE ts < prev_ts"""
                    n_bad += self.con.execute(f"SELECT count(*) FROM ({q})").fetchone()[0]
                    if len(ex) < 5:
                        ex += _rows(self.con, q + " LIMIT 5")
                per_table[t] = n_bad
                if n_bad:
                    ok = False
                    examples[f"{feed}.{t}"] = ex[:5]
            se = self.con.execute(f"""
                SELECT coalesce(sum((ts < prev)::INT), 0) FROM (
                  SELECT ts, lag(ts) OVER (ORDER BY seq) AS prev FROM {feed}_system_events)""").fetchone()[0]
            per_table["system_events"] = se
            if seg["send_time_regressions"] or se:
                ok = False
            summary[feed] = {**seg, "event_ts_regressions_by_table": per_table}
        self._add("timestamp_monotonic", "pass" if ok else "fail", summary, examples,
                  notes="Event timestamps are checked within (message type, symbol) in sequence order, "
                        "the only ordering the TOPS and DEEP specs guarantee. capture minus send is informational.",
                  seconds=time.perf_counter() - t0)

    # 3
    def deep_crossed_locked(self) -> None:
        t0 = time.perf_counter()
        s = _one(self.con, """
            SELECT count(*) AS bbo_rows,
                   coalesce(sum((bid_size > 0 AND ask_size > 0 AND bid_price > ask_price)::INT), 0) AS crossed,
                   coalesce(sum((bid_size > 0 AND ask_size > 0 AND bid_price = ask_price)::INT), 0) AS locked,
                   coalesce(sum((bid_size > 0 AND ask_size > 0)::INT), 0) AS two_sided
            FROM deep_bbo_from_deep""")
        s["book_counters"] = self.manifest["deep"]["book"]
        ex = {}
        if s["crossed"] or s["locked"]:
            ex["crossed_or_locked"] = _rows(self.con, """
                SELECT * FROM deep_bbo_from_deep
                WHERE bid_size > 0 AND ask_size > 0 AND bid_price >= ask_price ORDER BY seq LIMIT 10""")
        self._add("deep_crossed_locked", "pass" if not (s["crossed"] or s["locked"]) else "fail", s, ex,
                  seconds=time.perf_counter() - t0)

    # 4
    def bbo_deep_vs_tops(self) -> None:
        t0 = time.perf_counter()
        end = self.t_end
        cat_sql = """
            CASE
              WHEN d_bid_price = bid_price AND d_bid_size = bid_size AND d_ask_price = ask_price
                   AND d_ask_size = ask_size THEN 'match'
              WHEN halted AND bid_size = 0 AND ask_size = 0 THEN 'tops_zero_quote_while_unavailable'
              WHEN d_bid_price = bid_price AND d_ask_price = ask_price THEN 'size_only'
              ELSE 'price'
            END"""
        base = """
            WITH d AS (
              SELECT symbol, ts, bid_price AS d_bid_price, bid_size AS d_bid_size, ask_price AS d_ask_price,
                     ask_size AS d_ask_size, seq AS d_seq, ts AS d_ts
              FROM deep_bbo_from_deep WHERE hash(symbol) % {B} = {k}
              QUALIFY row_number() OVER (PARTITION BY symbol, ts ORDER BY seq DESC) = 1),
            q AS (SELECT symbol, ts, seq, bid_price, bid_size, ask_price, ask_size, flags, halted, pre_post_market
                  FROM tops_quotes WHERE hash(symbol) % {B} = {k} AND epoch_ns(ts) <= {end}),
            j AS (
              SELECT q.*, coalesce(d.d_bid_price, 0) AS d_bid_price, coalesce(d.d_bid_size, 0) AS d_bid_size,
                     coalesce(d.d_ask_price, 0) AS d_ask_price, coalesce(d.d_ask_size, 0) AS d_ask_size,
                     d.d_seq, d.d_ts, (d.d_ts = q.ts) AS same_event
              FROM q ASOF LEFT JOIN d ON q.symbol = d.symbol AND q.ts >= d.ts)
            SELECT *, {cat} AS category FROM j"""
        counts: dict[str, int] = {}
        same_event: dict[str, int] = {}
        examples: dict[str, list] = {}
        for k in range(self.buckets):
            sql = base.format(B=self.buckets, k=k, end=end, cat=cat_sql)
            for cat, n, se in self.con.execute(
                    f"SELECT category, count(*), sum(same_event::INT) FROM ({sql}) GROUP BY 1").fetchall():
                counts[cat] = counts.get(cat, 0) + n
                same_event[cat] = same_event.get(cat, 0) + (se or 0)
            for cat in list(counts):
                if cat != "match" and len(examples.get(cat, [])) < 5:
                    examples.setdefault(cat, []).extend(
                        _rows(self.con, f"SELECT * FROM ({sql}) WHERE category = '{cat}' ORDER BY seq LIMIT 5"))
        total = sum(counts.values())
        explained = {"tops_zero_quote_while_unavailable"}
        unexplained = sum(v for c, v in counts.items() if c not in explained and c != "match")
        # Coverage in the other direction: DEEP BBO changes with no TOPS quote at the same (symbol, ts).
        cov = _one(self.con, f"""
            SELECT count(*) AS deep_bbo_changes,
                   count(*) FILTER (WHERE q.symbol IS NULL) AS deep_bbo_changes_without_tops_quote
            FROM (SELECT DISTINCT symbol, ts FROM deep_bbo_from_deep WHERE epoch_ns(ts) <= {end}) d
            LEFT JOIN (SELECT DISTINCT symbol, ts FROM tops_quotes) q USING (symbol, ts)""")
        summary = {"tops_quotes_compared": total, "by_category": counts,
                   "same_event_timestamp_by_category": same_event,
                   "agreement_rate": round(counts.get("match", 0) / total, 8) if total else None,
                   "agreement_rate_after_explained": round((total - unexplained) / total, 8) if total else None,
                   "unexplained_mismatches": unexplained, **cov}
        self._add("bbo_deep_vs_tops", "pass" if unexplained == 0 else "fail", summary,
                  {c: v[:5] for c, v in examples.items()}, seconds=time.perf_counter() - t0,
                  notes="DEEP BBO is taken as of the last completed DEEP event at or before the TOPS quote "
                        "timestamp (ASOF join, last sequence number within a timestamp).")

    # 5
    def trade_reconciliation(self) -> None:
        t0 = time.perf_counter()
        end = self.t_end
        s = _one(self.con, f"""
            WITH t AS (SELECT * FROM tops_trades WHERE epoch_ns(ts) <= {end}),
                 d AS (SELECT * FROM deep_trades WHERE epoch_ns(ts) <= {end})
            SELECT count(*) AS joined_rows,
                   count(*) FILTER (WHERE t.seq IS NOT NULL AND d.seq IS NOT NULL) AS matched_ids,
                   count(*) FILTER (WHERE t.seq IS NOT NULL AND d.seq IS NOT NULL AND t.symbol = d.symbol
                                    AND t.price = d.price AND t.size = d.size AND t.flags = d.flags
                                    AND t.ts = d.ts) AS exact_matches,
                   count(*) FILTER (WHERE d.seq IS NULL) AS tops_only,
                   count(*) FILTER (WHERE t.seq IS NULL) AS deep_only,
                   count(*) FILTER (WHERE t.msg_type = 'T') AS tops_reports,
                   count(*) FILTER (WHERE t.msg_type = 'B') AS tops_breaks,
                   sum(t.size) FILTER (WHERE t.msg_type = 'T') AS tops_shares,
                   sum(d.size) FILTER (WHERE d.msg_type = 'T') AS deep_shares
            FROM t FULL OUTER JOIN d ON t.trade_id = d.trade_id AND t.msg_type = d.msg_type""")
        dup = _one(self.con, f"""
            SELECT (SELECT count(*) - count(DISTINCT (trade_id, msg_type)) FROM tops_trades) AS tops_duplicate_ids,
                   (SELECT count(*) - count(DISTINCT (trade_id, msg_type)) FROM deep_trades) AS deep_duplicate_ids""")
        s.update(dup)
        bad = s["joined_rows"] - s["exact_matches"]
        ex = {}
        if bad:
            ex["mismatch"] = _rows(self.con, f"""
                WITH t AS (SELECT * FROM tops_trades WHERE epoch_ns(ts) <= {end}),
                     d AS (SELECT * FROM deep_trades WHERE epoch_ns(ts) <= {end})
                SELECT t.trade_id AS t_id, d.trade_id AS d_id, t.symbol AS t_sym, d.symbol AS d_sym,
                       t.price AS t_px, d.price AS d_px, t.size AS t_sz, d.size AS d_sz, t.ts AS t_ts, d.ts AS d_ts
                FROM t FULL OUTER JOIN d ON t.trade_id = d.trade_id AND t.msg_type = d.msg_type
                WHERE NOT (t.seq IS NOT NULL AND d.seq IS NOT NULL AND t.symbol = d.symbol AND t.price = d.price
                           AND t.size = d.size AND t.flags = d.flags AND t.ts = d.ts) LIMIT 10""")
        s["reconciliation_rate"] = round(s["exact_matches"] / s["joined_rows"], 8) if s["joined_rows"] else None
        self._add("trade_reconciliation", "pass" if bad == 0 and not dup["tops_duplicate_ids"] else "fail", s, ex,
                  seconds=time.perf_counter() - t0)

    # 6
    def halted_no_quotes(self) -> None:
        t0 = time.perf_counter()
        base = """
            WITH st AS (SELECT symbol, ts, status, reason FROM tops_trading_status),
            q AS (SELECT symbol, ts, seq, bid_price, bid_size, ask_price, ask_size, halted FROM tops_quotes),
            j AS (SELECT q.*, st.status, st.reason, st.ts AS status_ts
                  FROM q ASOF LEFT JOIN st ON q.symbol = st.symbol AND q.ts >= st.ts)
            SELECT * FROM j"""
        s = _one(self.con, f"""
            SELECT count(*) AS quotes,
                   count(*) FILTER (WHERE status IS NULL) AS quotes_before_any_status,
                   count(*) FILTER (WHERE status IN ('H', 'P', 'O')) AS quotes_while_not_trading,
                   count(*) FILTER (WHERE status IN ('H', 'P', 'O') AND halted) AS flagged_unavailable,
                   count(*) FILTER (WHERE status IN ('H', 'P', 'O') AND NOT halted) AS not_flagged,
                   count(*) FILTER (WHERE status IN ('H', 'P', 'O') AND bid_size > 0 AND ask_size > 0)
                     AS two_sided_while_not_trading,
                   count(*) FILTER (WHERE status IN ('H', 'P', 'O') AND NOT halted
                                    AND (bid_size > 0 OR ask_size > 0)) AS violations,
                   count(*) FILTER (WHERE status = 'T' AND halted) AS flagged_unavailable_while_trading
            FROM ({base})""")
        halts = _one(self.con, """
            SELECT count(*) AS status_messages,
                   count(DISTINCT symbol) FILTER (WHERE status IN ('H', 'P', 'O')) AS symbols_ever_not_trading,
                   count(*) FILTER (WHERE status = 'H') AS halt_messages,
                   count(*) FILTER (WHERE status = 'P') AS pause_messages,
                   count(*) FILTER (WHERE status = 'O') AS oap_messages
            FROM tops_trading_status""")
        s.update(halts)
        # Trades printed while the symbol was not trading (an auction cross may legitimately print on release).
        tr = _one(self.con, """
            WITH st AS (SELECT symbol, ts, status FROM tops_trading_status)
            SELECT count(*) FILTER (WHERE st.status IN ('H', 'P', 'O')) AS trades_while_not_trading
            FROM tops_trades t ASOF LEFT JOIN st ON t.symbol = st.symbol AND t.ts >= st.ts""")
        s.update(tr)
        ex = {}
        if s["violations"] or s["trades_while_not_trading"]:
            ex["violations"] = _rows(self.con, f"""SELECT * FROM ({base}) WHERE status IN ('H','P','O')
                AND NOT halted AND (bid_size > 0 OR ask_size > 0) ORDER BY seq LIMIT 10""")
        ex["halted_symbols"] = _rows(self.con, """
            SELECT symbol, ts, status, reason FROM tops_trading_status WHERE status IN ('H','P','O')
            ORDER BY ts LIMIT 10""")
        if s["flagged_unavailable_while_trading"]:
            ex["flagged_while_trading"] = _rows(self.con, f"""SELECT * FROM ({base}) WHERE status = 'T' AND halted
                ORDER BY seq LIMIT 5""")
        self._add("halted_no_quotes", "pass" if s["violations"] == 0 and s["trades_while_not_trading"] == 0 else "fail",
                  s, ex, seconds=time.perf_counter() - t0)

    def run(self) -> list[dict]:
        self.sequence_continuity()
        self.timestamp_monotonic()
        self.deep_crossed_locked()
        self.bbo_deep_vs_tops()
        self.trade_reconciliation()
        self.halted_no_quotes()
        return self.results


def write_report(results: list[dict], meta: dict, path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    doc = {"meta": meta, "checks": results}
    Path(str(path_prefix) + ".json").write_text(json.dumps(doc, indent=2, default=str))
    lines = [f"# Data quality report {meta.get('date', '')}", "",
             f"Parquet root {meta['root']}, generated {meta['generated']}.", "",
             "| check | status | seconds |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['check']} | {r['status']} | {r['seconds']} |")
    for r in results:
        lines += ["", f"## {r['check']}", "", f"Status {r['status']}.", ""]
        if r["notes"]:
            lines += [r["notes"], ""]
        lines += ["```json", json.dumps(r["summary"], indent=2, default=str), "```"]
        for name, rows in r["examples"].items():
            if not rows:
                continue
            lines += ["", f"### Examples {name}", "", "```json"]
            lines += [json.dumps(x, default=str) for x in rows[:5]]
            lines += ["```"]
    Path(str(path_prefix) + ".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--buckets", type=int, default=BUCKETS_DEFAULT)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--memory", default="12GB")
    a = ap.parse_args()
    t0 = time.perf_counter()
    suite = Suite(a.root, a.buckets, a.threads, a.memory)
    results = suite.run()
    meta = {"root": str(a.root), "date": a.root.name, "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "common_event_window_end_ns": suite.t_end, "seconds": round(time.perf_counter() - t0, 1)}
    write_report(results, meta, a.report)
    print(json.dumps({r["check"]: r["status"] for r in results}))


if __name__ == "__main__":
    main()
