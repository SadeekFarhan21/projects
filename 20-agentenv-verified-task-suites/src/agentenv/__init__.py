"""agentenv: verifiable agent environments, evals and (later) RL.

Paths in task files are resolved against PROJECT_ROOT, which defaults to the
repository directory that contains this package and can be overridden with the
AGENTENV_ROOT environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("AGENTENV_ROOT", Path(__file__).resolve().parents[2])
).resolve()

__all__ = ["PROJECT_ROOT"]
