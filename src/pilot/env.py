"""Auto-load environment variables from `.env` files.

Looks for `.env` first in the code/ root (one level up from this
package), then in the current working directory. Existing process
environment variables take precedence — `.env` only fills in missing
keys, never overrides what's already set.

This is called once at import time when any CLI module runs. The
load is idempotent.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """code/ root, two levels up from this file (src/pilot/env.py)."""
    return Path(__file__).resolve().parents[2]


def load_env() -> list[Path]:
    """Load .env files. Returns the list of files actually loaded."""
    candidates = [
        _project_root() / ".env",
        Path.cwd() / ".env",
    ]
    loaded: list[Path] = []
    for path in candidates:
        if path.exists() and path.resolve() not in {p.resolve() for p in loaded}:
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded
