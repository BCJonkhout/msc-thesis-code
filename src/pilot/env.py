"""Auto-load environment variables from `.env` files.

Looks for `.env` first in the code/ root (one level up from this
package), then in the current working directory.

**`.env` overrides process environment by default.** This is the
opposite of the standard dotenv default — but it's the right choice
here because the user iterates on `.env` (rotating keys, adding new
providers) while the parent shell may be hours or days old with
stale exported values. If you genuinely want process env to win,
pass `override=False`.

This is called once at import time when any CLI module runs. The
load is idempotent.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """code/ root, two levels up from this file (src/pilot/env.py)."""
    return Path(__file__).resolve().parents[2]


def load_env(*, override: bool = True) -> list[Path]:
    """Load .env files. Returns the list of files actually loaded."""
    candidates = [
        _project_root() / ".env",
        Path.cwd() / ".env",
    ]
    loaded: list[Path] = []
    for path in candidates:
        if path.exists() and path.resolve() not in {p.resolve() for p in loaded}:
            load_dotenv(path, override=override)
            loaded.append(path)

    # HuggingFace SDK reads HF_TOKEN; the .env stores the same value
    # under HUGGINGFACE_ACCESS_TOKEN to match the human-readable name
    # of the credential field in the provider dashboard. Mirror it
    # into HF_TOKEN so the SDK picks it up without further config.
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token

    return loaded
