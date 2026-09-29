"""Build a TPC-H SF0.01 sample with DuckDB's dbgen and copy it into SQLite."""
import sqlite3
import sys
from pathlib import Path

import duckdb

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/tpch.sqlite")
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.01)")
tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
if out.exists():
    out.unlink()
lite = sqlite3.connect(out)
for t in tables:
    df = con.execute(f"SELECT * FROM {t}").df()
    for col in df.columns:  # dates to ISO strings, decimals to float
        if str(df[col].dtype).startswith("datetime"):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        elif df[col].dtype == object:
            df[col] = df[col].map(lambda v: float(v) if v.__class__.__name__ == "Decimal" else v)
    df.to_sql(t, lite, index=False)
    print(t, len(df))
lite.commit()
lite.close()
