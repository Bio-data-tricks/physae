"""I/O helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_directory(base: Path, run_name: str | None = None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    final_name = run_name or timestamp
    run_dir = base / final_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


__all__ = ["create_run_directory"]
