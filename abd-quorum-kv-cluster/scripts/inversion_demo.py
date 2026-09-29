"""Run the deterministic new-old inversion scenario with and without read
write-back and record what each read returned and the checker's verdict.

Usage: uv run python scripts/inversion_demo.py   -> results/inversion_demo.txt
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvstore.scenarios import run_inversion_scenario  # noqa: E402

out = Path(__file__).resolve().parents[1] / "results" / "inversion_demo.txt"
lines = ["scenario: 3 nodes N=3 R=2 W=2; put(new) reaches n1 only; read via n1; crash n1; read via n2"]
for wb in (False, True):
    for trial in range(5):
        with tempfile.TemporaryDirectory() as d:
            r1, r2, ok = asyncio.run(run_inversion_scenario(d, writeback=wb))
        lines.append(f"writeback={wb} trial={trial} read1={r1} read2={r2} linearizable={ok}")
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
