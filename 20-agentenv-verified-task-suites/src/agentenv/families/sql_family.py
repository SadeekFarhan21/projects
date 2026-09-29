"""Data analysis tasks over SQLite (Chinook and a TPC-H SF0.01 sample).

Every task comes from a template: a question, a reference SQL query and a
parameter query that enumerates real values from the database. The ground
truth is whatever the reference SQL returns, checked with numeric tolerance
(or exact normalized text for "which X" questions). A candidate is rejected
when the answer is NULL or zero (a null agent submitting 0 must not pass),
or, for text answers, when the top two rows tie (ambiguous ground truth).
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from dataclasses import dataclass

from .. import PROJECT_ROOT
from ..task import Task

DBS = {"chinook": "data/chinook.sqlite", "tpch": "data/tpch.sqlite"}
DB_BLURB = {
    "chinook": "db.sqlite is the Chinook digital music store database (artists, albums, tracks, genres, "
               "playlists, customers, employees, invoices, invoice lines).",
    "tpch": "db.sqlite is a small TPC-H sample (region, nation, customer, supplier, part, partsupp, "
            "orders, lineitem). Dates are ISO strings like 1995-03-15.",
}


@dataclass(frozen=True)
class Template:
    name: str
    db: str
    difficulty: str
    quota: int
    kind: str  # number | text
    question: str
    sql: str
    params_sql: str
    tags: tuple[str, ...] = ()


def q(v) -> str:
    """SQL string literal."""
    return "'" + str(v).replace("'", "''") + "'"


TEMPLATES: list[Template] = [
    # ---------------- Chinook, easy (single table)
    Template("inv_count_country", "chinook", "easy", 4, "number",
             "How many invoices were billed to the country '{0}'?",
             "SELECT COUNT(*) FROM Invoice WHERE BillingCountry = {q0}",
             "SELECT BillingCountry FROM Invoice GROUP BY 1 ORDER BY 1", ("count", "filter")),
    Template("inv_sum_year", "chinook", "easy", 4, "number",
             "What is the sum of the Total column over all invoices dated in the year {0}?",
             "SELECT SUM(Total) FROM Invoice WHERE strftime('%Y', InvoiceDate) = {q0}",
             "SELECT DISTINCT strftime('%Y', InvoiceDate) FROM Invoice ORDER BY 1", ("sum", "date")),
    Template("cust_count_country", "chinook", "easy", 4, "number",
             "How many customers have Country equal to '{0}'?",
             "SELECT COUNT(*) FROM Customer WHERE Country = {q0}",
             "SELECT Country FROM Customer GROUP BY 1 ORDER BY 1", ("count", "filter")),
    Template("inv_avg_city", "chinook", "easy", 4, "number",
             "What is the average invoice Total for invoices with BillingCity '{0}'? Give at least 2 decimals.",
             "SELECT AVG(Total) FROM Invoice WHERE BillingCity = {q0}",
             "SELECT BillingCity FROM Invoice GROUP BY 1 ORDER BY 1", ("avg", "filter")),
    Template("track_long", "chinook", "easy", 4, "number",
             "How many tracks are strictly longer than {0} minutes?",
             "SELECT COUNT(*) FROM Track WHERE Milliseconds > {0} * 60000",
             "SELECT value FROM (SELECT 4 AS value UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 "
             "UNION SELECT 8 UNION SELECT 10 UNION SELECT 12 UNION SELECT 15 UNION SELECT 20)",
             ("count", "units")),
    # ---------------- Chinook, medium (one join)
    Template("genre_track_count", "chinook", "medium", 4, "number",
             "How many tracks belong to the genre named '{0}'?",
             "SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = {q0}",
             "SELECT Name FROM Genre ORDER BY 1", ("count", "join")),
    Template("artist_album_count", "chinook", "medium", 4, "number",
             "How many albums does the artist named '{0}' have?",
             "SELECT COUNT(*) FROM Album al JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = {q0}",
             "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
             "GROUP BY ar.ArtistId HAVING COUNT(*) >= 2 ORDER BY 1", ("count", "join")),
    Template("genre_revenue", "chinook", "medium", 4, "number",
             "What is the total revenue, summing UnitPrice * Quantity over invoice lines, for tracks in the genre '{0}'?",
             "SELECT SUM(il.UnitPrice * il.Quantity) FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
             "JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = {q0}",
             "SELECT Name FROM Genre ORDER BY 1", ("sum", "join")),
    Template("customer_spend", "chinook", "medium", 4, "number",
             "What is the total of all invoice Totals for the customer {0} {1}?",
             "SELECT SUM(i.Total) FROM Invoice i JOIN Customer c ON c.CustomerId = i.CustomerId "
             "WHERE c.FirstName = {q0} AND c.LastName = {q1}",
             "SELECT FirstName, LastName FROM Customer ORDER BY CustomerId", ("sum", "join")),
    Template("genre_avg_minutes", "chinook", "medium", 4, "number",
             "What is the average track length in minutes (Milliseconds / 60000.0) for tracks in the genre '{0}'? Give at least 2 decimals.",
             "SELECT AVG(t.Milliseconds / 60000.0) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = {q0}",
             "SELECT Name FROM Genre ORDER BY 1", ("avg", "join", "units")),
    # ---------------- Chinook, hard (multi join, group, argmax)
    Template("genre_top_artist", "chinook", "hard", 4, "text",
             "Which artist has the most tracks in the genre '{0}'? Submit the artist name exactly.",
             "SELECT ar.Name, COUNT(*) AS n FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId "
             "JOIN Artist ar ON ar.ArtistId = al.ArtistId JOIN Genre g ON g.GenreId = t.GenreId "
             "WHERE g.Name = {q0} GROUP BY ar.ArtistId ORDER BY n DESC",
             "SELECT Name FROM Genre ORDER BY 1", ("argmax", "join3")),
    Template("rep_revenue_year", "chinook", "hard", 4, "number",
             "What is the total of invoice Totals in {2} for customers whose support rep is the employee {0} {1}?",
             "SELECT SUM(i.Total) FROM Invoice i JOIN Customer c ON c.CustomerId = i.CustomerId "
             "JOIN Employee e ON e.EmployeeId = c.SupportRepId WHERE e.FirstName = {q0} AND e.LastName = {q1} "
             "AND strftime('%Y', i.InvoiceDate) = {q2}",
             "SELECT e.FirstName, e.LastName, y.yr FROM Employee e JOIN (SELECT DISTINCT SupportRepId FROM Customer) s "
             "ON s.SupportRepId = e.EmployeeId CROSS JOIN (SELECT DISTINCT strftime('%Y', InvoiceDate) AS yr FROM Invoice) y "
             "ORDER BY 1, 2, 3", ("sum", "join3", "date")),
    Template("artist_customers", "chinook", "hard", 4, "number",
             "How many distinct customers bought at least one track by the artist '{0}'?",
             "SELECT COUNT(DISTINCT i.CustomerId) FROM InvoiceLine il JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
             "JOIN Track t ON t.TrackId = il.TrackId JOIN Album al ON al.AlbumId = t.AlbumId "
             "JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = {q0}",
             "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId GROUP BY ar.ArtistId "
             "HAVING COUNT(*) >= 2 ORDER BY 1", ("count_distinct", "join4")),
    Template("genre_top_country", "chinook", "hard", 4, "text",
             "Which billing country has the highest total revenue (UnitPrice * Quantity over invoice lines) from tracks in the genre '{0}'? Submit the country name exactly.",
             "SELECT i.BillingCountry, SUM(il.UnitPrice * il.Quantity) AS rev FROM InvoiceLine il "
             "JOIN Invoice i ON i.InvoiceId = il.InvoiceId JOIN Track t ON t.TrackId = il.TrackId "
             "JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = {q0} GROUP BY 1 ORDER BY rev DESC",
             "SELECT Name FROM Genre ORDER BY 1", ("argmax", "join3")),
    Template("playlist_artists", "chinook", "hard", 4, "number",
             "How many distinct artists have at least one track in the playlist named '{0}'?",
             "SELECT COUNT(DISTINCT al.ArtistId) FROM Playlist p JOIN PlaylistTrack pt ON pt.PlaylistId = p.PlaylistId "
             "JOIN Track t ON t.TrackId = pt.TrackId JOIN Album al ON al.AlbumId = t.AlbumId WHERE p.Name = {q0}",
             "SELECT Name FROM Playlist GROUP BY Name HAVING COUNT(*) = 1 ORDER BY 1", ("count_distinct", "join3")),
    # ---------------- TPC-H, easy
    Template("orders_priority", "tpch", "easy", 5, "number",
             "How many orders have o_orderpriority equal to '{0}'?",
             "SELECT COUNT(*) FROM orders WHERE o_orderpriority = {q0}",
             "SELECT DISTINCT o_orderpriority FROM orders ORDER BY 1", ("count", "filter")),
    Template("qty_shipmode", "tpch", "easy", 4, "number",
             "What is the total l_quantity of line items shipped with l_shipmode '{0}'?",
             "SELECT SUM(l_quantity) FROM lineitem WHERE l_shipmode = {q0}",
             "SELECT DISTINCT l_shipmode FROM lineitem ORDER BY 1", ("sum", "filter")),
    Template("cust_segment", "tpch", "easy", 4, "number",
             "How many customers are in the market segment '{0}'?",
             "SELECT COUNT(*) FROM customer WHERE c_mktsegment = {q0}",
             "SELECT DISTINCT c_mktsegment FROM customer ORDER BY 1", ("count", "filter")),
    # ---------------- TPC-H, medium
    Template("cust_nation", "tpch", "medium", 5, "number",
             "How many customers belong to the nation '{0}'?",
             "SELECT COUNT(*) FROM customer c JOIN nation n ON n.n_nationkey = c.c_nationkey WHERE n.n_name = {q0}",
             "SELECT n_name FROM nation ORDER BY 1", ("count", "join")),
    Template("supp_bal_nation", "tpch", "medium", 5, "number",
             "What is the total s_acctbal of suppliers in the nation '{0}'? Give at least 2 decimals.",
             "SELECT SUM(s.s_acctbal) FROM supplier s JOIN nation n ON n.n_nationkey = s.s_nationkey WHERE n.n_name = {q0}",
             "SELECT n_name FROM nation ORDER BY 1", ("sum", "join")),
    Template("orders_seg_year", "tpch", "medium", 4, "number",
             "How many orders were placed in {1} by customers in the market segment '{0}'?",
             "SELECT COUNT(*) FROM orders o JOIN customer c ON c.c_custkey = o.o_custkey "
             "WHERE c.c_mktsegment = {q0} AND substr(o.o_orderdate, 1, 4) = {q1}",
             "SELECT s.c_mktsegment, y.yr FROM (SELECT DISTINCT c_mktsegment FROM customer) s CROSS JOIN "
             "(SELECT DISTINCT substr(o_orderdate, 1, 4) AS yr FROM orders) y ORDER BY 1, 2", ("count", "join", "date")),
    # ---------------- TPC-H, hard
    Template("region_revenue_year", "tpch", "hard", 5, "number",
             "What is the total revenue, sum of l_extendedprice * (1 - l_discount), of line items in orders placed in {1} by customers whose nation is in the region '{0}'? Give at least 2 decimals.",
             "SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) FROM lineitem l JOIN orders o ON o.o_orderkey = l.l_orderkey "
             "JOIN customer c ON c.c_custkey = o.o_custkey JOIN nation n ON n.n_nationkey = c.c_nationkey "
             "JOIN region r ON r.r_regionkey = n.n_regionkey WHERE r.r_name = {q0} AND substr(o.o_orderdate, 1, 4) = {q1}",
             "SELECT r.r_name, y.yr FROM region r CROSS JOIN (SELECT DISTINCT substr(o_orderdate, 1, 4) AS yr FROM orders) y "
             "ORDER BY 1, 2", ("sum", "join4", "date")),
    Template("nation_top_orders_year", "tpch", "hard", 4, "text",
             "Which nation's customers placed the most orders in {0}? Submit the nation name exactly.",
             "SELECT n.n_name, COUNT(*) AS cnt FROM orders o JOIN customer c ON c.c_custkey = o.o_custkey "
             "JOIN nation n ON n.n_nationkey = c.c_nationkey WHERE substr(o.o_orderdate, 1, 4) = {q0} "
             "GROUP BY 1 ORDER BY cnt DESC",
             "SELECT DISTINCT substr(o_orderdate, 1, 4) FROM orders ORDER BY 1", ("argmax", "join3", "date")),
    Template("brand_region_avg", "tpch", "hard", 4, "number",
             "What is the average of l_extendedprice * (1 - l_discount) over line items for parts of brand '{0}' supplied by suppliers located in the region '{1}'? Give at least 2 decimals.",
             "SELECT AVG(l.l_extendedprice * (1 - l.l_discount)) FROM lineitem l JOIN part p ON p.p_partkey = l.l_partkey "
             "JOIN supplier s ON s.s_suppkey = l.l_suppkey JOIN nation n ON n.n_nationkey = s.s_nationkey "
             "JOIN region r ON r.r_regionkey = n.n_regionkey WHERE p.p_brand = {q0} AND r.r_name = {q1}",
             "SELECT b.p_brand, r.r_name FROM (SELECT DISTINCT p_brand FROM part) b CROSS JOIN region r ORDER BY 1, 2",
             ("avg", "join4")),
]


def _render(t: Template, params: tuple) -> tuple[str, str]:
    fmt = {f"q{i}": q(p) for i, p in enumerate(params)}
    question = t.question.format(*params)
    sql = t.sql.format(*params, **fmt)
    return question, sql


def schema_text(con: sqlite3.Connection) -> str:
    """Compact schema, one line per table: Name(col1, col2, ...)."""
    out = []
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{name}')")]
        out.append(f"{name}({', '.join(cols)})")
    return "\n".join(out)


def _stable_rng(seed: int, name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:12], 16))


def generate(seed: int = 0) -> list[Task]:
    tasks: list[Task] = []
    cons = {k: sqlite3.connect(PROJECT_ROOT / v) for k, v in DBS.items()}
    schemas = {k: schema_text(c) for k, c in cons.items()}
    for t in TEMPLATES:
        con = cons[t.db]
        params = [tuple(r) for r in con.execute(t.params_sql).fetchall()]
        _stable_rng(seed, t.name).shuffle(params)
        kept = 0
        for p in params:
            if kept >= t.quota:
                break
            question, sql = _render(t, p)
            rows = con.execute(sql).fetchall()
            if not rows or rows[0][0] is None:
                continue
            if t.kind == "number":
                expected = float(rows[0][0])
                if abs(expected) < 1e-9:
                    continue
                ref_sql = sql
            else:
                if len(rows) >= 2 and rows[0][1] == rows[1][1]:
                    continue  # tie, ambiguous
                expected = str(rows[0][0])
                ref_sql = sql + " LIMIT 1"
            kept += 1
            fmt_hint = ("Submit only the number." if t.kind == "number" else "")
            prompt = (f"{DB_BLURB[t.db]}\nTables and columns\n{schemas[t.db]}\n\n"
                      f"Question: {question}\n{fmt_hint}\n"
                      "Use run_sql to compute the answer, then call submit with the answer.")
            tid = f"sql-{t.db}-{t.name}-{kept:02d}"
            tasks.append(Task(
                id=tid, family="sql", difficulty=t.difficulty, tags=[t.db, *t.tags],
                prompt=prompt,
                fixtures=[{"copy": DBS[t.db], "to": "db.sqlite"}],
                allowed_tools=["run_sql", "list_dir", "submit"],
                verifier={"type": "python", "function": "agentenv.verifiers:numeric_or_text_answer",
                          "args": {"expected": expected, "kind": t.kind, "rel_tol": 1e-4, "abs_tol": 0.01}},
                timeout_s=300, limits={"step_limit": 8, "token_budget": 1500},
                reference={"sql": ref_sql},
                metadata={"template": t.name, "params": list(p), "db": t.db, "answer_kind": t.kind},
            ).validate())
        if kept < t.quota:
            raise RuntimeError(f"template {t.name} produced only {kept}/{t.quota} tasks")
    return tasks
