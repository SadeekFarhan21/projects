"""
File utilities for run folder management.

Handles creating run folders with slugified names and saving step outputs.
"""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path


def create_run_folder(query: str, base_path: str = "runs") -> str:
    """
    Create a run folder for the given query.

    The folder name is a slugified version of the query plus a hash suffix.

    Args:
        query: The input query string
        base_path: Base directory for runs (default: "runs")

    Returns:
        The full path to the created run folder
    """
    # Create slug from query (first 50 chars, lowercase, alphanumeric + dashes)
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower())[:50]
    slug = slug.strip("-")

    # Create hash suffix
    hash_part = hashlib.sha256(query.encode()).hexdigest()[:8]

    # Create folder name
    folder_name = f"{slug}+{hash_part}"

    # Get the project root (parent of agent directory)
    project_root = Path(__file__).parent.parent.parent
    run_folder = project_root / base_path / folder_name

    # Create the directory
    run_folder.mkdir(parents=True, exist_ok=True)

    return str(run_folder)


def save_step_output(run_folder: str, filename: str, content: str) -> str:
    """
    Save step output to a file in the run folder.

    Args:
        run_folder: Path to the run folder
        filename: Name of the file to save
        content: Content to write to the file

    Returns:
        The full path to the saved file
    """
    file_path = os.path.join(run_folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def append_to_log(run_folder: str, message: str) -> None:
    """
    Append a message to the run log.

    Args:
        run_folder: Path to the run folder
        message: Message to append to the log
    """
    log_path = os.path.join(run_folder, "run_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
