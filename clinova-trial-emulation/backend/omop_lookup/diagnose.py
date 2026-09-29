"""
Diagnostic tool to debug lookup issues.
Run: python -m omop_lookup.diagnose
"""

import time
from pathlib import Path


def run_diagnostics():
    print("=" * 70)
    print("OMOP LOOKUP DIAGNOSTICS")
    print("=" * 70)

    # Step 1: Check if database file exists
    print("\n[1] Checking database file...")
    from .config import DEFAULT_CONFIG
    db_path = DEFAULT_CONFIG.db_path
    print(f"    Database path: {db_path}")
    print(f"    Exists: {db_path.exists()}")
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"    Size: {size_mb:.1f} MB")

    # Step 2: Connect and check tables
    print("\n[2] Checking database tables...")
    try:
        from .database import get_database
        db = get_database()

        tables_to_check = [
            'concept',
            'concept_synonym',
            'concept_relationship',
            'maps_to_lookup',
            'concept_search'
        ]

        for table in tables_to_check:
            try:
                count = db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"    {table}: {count:,} rows")
            except Exception as e:
                print(f"    {table}: ERROR - {e}")
    except Exception as e:
        print(f"    ERROR connecting: {e}")
        return

    # Step 3: Check concept_search structure
    print("\n[3] Checking concept_search table structure...")
    try:
        cols = db.conn.execute("DESCRIBE concept_search").fetchall()
        print("    Columns:")
        for col in cols:
            print(f"      - {col[0]}: {col[1]}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 4: Check if normalized text is populated
    print("\n[4] Checking normalized text column...")
    try:
        # Count nulls
        null_count = db.conn.execute("""
            SELECT COUNT(*) FROM concept_search
            WHERE search_text_normalized IS NULL
        """).fetchone()[0]

        empty_count = db.conn.execute("""
            SELECT COUNT(*) FROM concept_search
            WHERE search_text_normalized = ''
        """).fetchone()[0]

        total = db.conn.execute("SELECT COUNT(*) FROM concept_search").fetchone()[0]

        print(f"    Total rows: {total:,}")
        print(f"    NULL normalized: {null_count:,}")
        print(f"    Empty normalized: {empty_count:,}")
        print(f"    Has normalized: {total - null_count - empty_count:,}")

    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 5: Sample data
    print("\n[5] Sample data from concept_search...")
    try:
        samples = db.conn.execute("""
            SELECT concept_id, search_text, search_text_normalized, domain_id
            FROM concept_search
            LIMIT 5
        """).fetchall()

        for s in samples:
            print(f"    ID: {s[0]}")
            print(f"      Text: '{s[1][:50]}...' " if len(str(s[1])) > 50 else f"      Text: '{s[1]}'")
            print(f"      Normalized: '{s[2]}'")
            print(f"      Domain: {s[3]}")
            print()
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 6: Search for "heart" specifically
    print("\n[6] Searching for 'heart' in normalized text...")
    try:
        start = time.time()
        heart_count = db.conn.execute("""
            SELECT COUNT(*) FROM concept_search
            WHERE search_text_normalized LIKE '%heart%'
        """).fetchone()[0]
        elapsed = time.time() - start
        print(f"    Found {heart_count:,} rows containing 'heart'")
        print(f"    Query time: {elapsed*1000:.1f}ms")

        if heart_count > 0:
            samples = db.conn.execute("""
                SELECT concept_id, search_text, search_text_normalized
                FROM concept_search
                WHERE search_text_normalized LIKE '%heart%'
                LIMIT 3
            """).fetchall()
            print("    Samples:")
            for s in samples:
                print(f"      [{s[0]}] {s[1][:50]}")
                print(f"         normalized: '{s[2]}'")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 7: Test exact match query
    print("\n[7] Testing exact match for 'heart attack'...")
    try:
        start = time.time()
        exact = db.conn.execute("""
            SELECT concept_id, search_text, search_text_normalized
            FROM concept_search
            WHERE search_text_normalized = 'heart attack'
            LIMIT 5
        """).fetchall()
        elapsed = time.time() - start
        print(f"    Query time: {elapsed*1000:.1f}ms")
        print(f"    Results: {len(exact)}")
        for e in exact:
            print(f"      [{e[0]}] {e[1]}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 8: Check indexes
    print("\n[8] Checking indexes...")
    try:
        # DuckDB way to check indexes
        indexes = db.conn.execute("""
            SELECT index_name, table_name, is_unique
            FROM duckdb_indexes()
        """).fetchall()
        if indexes:
            for idx in indexes:
                print(f"    {idx[0]} on {idx[1]} (unique={idx[2]})")
        else:
            print("    No indexes found!")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Step 9: Test the actual search function
    print("\n[9] Testing search function directly...")
    try:
        from .normalize import normalize_text
        query = "heart attack"
        normalized = normalize_text(query)
        print(f"    Query: '{query}'")
        print(f"    Normalized: '{normalized}'")

        start = time.time()
        results = db.exact_match(normalized, domain_filter=None, limit=5)
        elapsed = time.time() - start
        print(f"    exact_match() returned: {len(results)} results")
        print(f"    Time: {elapsed*1000:.1f}ms")

        if results:
            for r in results[:3]:
                print(f"      [{r['concept_id']}] {r['concept_name']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()
