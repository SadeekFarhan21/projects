"""
Context loading utilities.

Loads shared context files (rubric.md, SOTAstack.md) for agent prompts.
"""

from pathlib import Path


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_rubric() -> str:
    """
    Load the rubric.md file content.

    Returns:
        The content of rubric.md as a string
    """
    rubric_path = _get_project_root() / "rubric.md"
    with open(rubric_path, "r", encoding="utf-8") as f:
        return f.read()


def load_sotastack() -> str:
    """
    Load the SOTAstack.md file content.

    Returns:
        The content of SOTAstack.md as a string
    """
    sotastack_path = _get_project_root() / "SOTAstack.md"
    with open(sotastack_path, "r", encoding="utf-8") as f:
        return f.read()
