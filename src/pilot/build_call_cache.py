"""Content-addressed cache for the (paid) per-call LLM responses made during a
document build — GraphRAG entity extraction + community reports, and RAPTOR
node summaries.

Why: a build of a very large document (e.g. NovelQA B42 at ~800k tokens) makes
several thousand provider calls. Without this, any interruption before the
build completes (a stall, a manual stop, a crash) discards every call and the
next attempt re-pays for all of them. Keying each response by the content of
its request lets a re-attempt reuse the calls already made and only pay for the
remainder, so a big build makes forward progress across restarts instead of
looping at full cost.

The cache is transparent: a hit returns the exact prior response, so the built
artefact is byte-identical to an uninterrupted build. It is content-addressed
(hash of kind + model + a discriminator + the request text), so it survives
code-version / cache-key changes and is shared across runs.

Cost accounting note: a cache hit makes no call and writes no ledger row, so
the append-only ledger still records each call exactly once across all
attempts (the primary run_index-0 deployment cost stays correct). The per-build
``build_meta`` captured on completion will, for a *resumed* build, omit the
rows from the earlier attempt(s); that only affects cross-candidate cost replay
(the optional Grok slice), not the primary cost.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

def _disabled() -> bool:
    return os.environ.get("PILOT_BUILD_CALL_CACHE", "").strip() == "0"


def _root() -> Path:
    return Path(
        os.environ.get("PILOT_BUILD_CALL_CACHE_DIR")
        or (Path("outputs") / "build_call_cache")
    )


def _key(kind: str, model: str, discriminator: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(kind.encode("utf-8"))
    h.update(b"\x1f")
    h.update((model or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(discriminator).encode("utf-8"))
    h.update(b"\x1f")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def get(kind: str, model: str, discriminator: str, text: str) -> str | None:
    """Return the cached response for this request, or None on miss."""
    if _disabled():
        return None
    h = _key(kind, model, discriminator, text)
    p = _root() / h[:2] / f"{h}.txt"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return None


def put(kind: str, model: str, discriminator: str, text: str, response: str) -> None:
    """Store ``response`` for this request. Best-effort; never raises."""
    if _disabled():
        return
    h = _key(kind, model, discriminator, text)
    d = _root() / h[:2]
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{h}.tmp"
        tmp.write_text(response, encoding="utf-8")
        os.replace(tmp, d / f"{h}.txt")
    except OSError:
        pass
