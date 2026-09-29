import sys
from pathlib import Path

import pytest

from agentenv import PROJECT_ROOT

HAVE_DATA = (PROJECT_ROOT / "data" / "chinook.sqlite").exists() and (PROJECT_ROOT / "data" / "packages").exists()
HAVE_TASKS = (PROJECT_ROOT / "tasks" / "sql").exists()
needs_data = pytest.mark.skipif(not (HAVE_DATA and HAVE_TASKS),
                                reason="run scripts/download_data.sh and scripts/generate_tasks.py first")
PY = sys.executable


@pytest.fixture
def ws(tmp_path) -> Path:
    d = tmp_path / "ws"
    d.mkdir()
    return d
