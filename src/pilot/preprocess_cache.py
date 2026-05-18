"""Persistent on-disk preprocessing cache for RAPTOR and GraphRAG.

The intra-process cache in ``pilot.cli.step_3_dry_run.run_dry_run``
keeps the per-paper preprocessing artefact (RAPTOR tree, GraphRAG
graph + community reports + entity vectors) alive across questions
inside ONE candidate's Python process. That is correct for cost
amortisation within a single candidate's run, but it is insufficient
when nine independent answerer candidates each launch their own
Python process: each process re-builds the artefact from scratch.
With a non-deterministic preprocessing-stage model (Gemini Flash Lite
at T=0 returns near- but not bit-identical summaries / entity
extractions across calls), the rebuilt artefacts diverge per
candidate, the retrieved context diverges, and the cross-candidate
comparison loses its measurement validity — the only thing that
should differ between candidates is the *answerer* call.

This module is the disk layer that sits below the intra-process
cache: intra-process miss → consult disk → if hit, reuse the exact
pickled artefact from whichever candidate built it first. The
artefact pickle is byte-identical across consumers, so RAPTOR and
GraphRAG produce byte-identical retrieved contexts per (paper_id,
question) cell regardless of which candidate's answerer is being
evaluated.

Cost-attribution invariant
--------------------------
Each candidate's ledger MUST continue to record the preprocessing
stage as if it had built the artefact itself. The cache is a
measurement-efficiency optimisation, not a cost discount:
``bar_C_deploy(c, n) = (C_off + sum C_on) / n`` is unchanged per
candidate. On a cache hit, ``replay_build_ledger`` writes synthetic
preprocess-stage rows reproducing the original build's
``uncached_input_tokens / cached_input_tokens / output_tokens /
wallclock_s / gpu_s_estimate``; new fields ``cache_loaded=true``
and ``source_run_id=<original_build_run_id>`` preserve the audit
trail so a reviewer can tell which rows were replayed from cache
without changing the cost arithmetic.

Cache key
---------
The cache key hashes every input that can possibly change the
artefact's content: architecture, paper_id, dataset, summary model
+ temperature, encoder model, chunk parameters, arch-specific
build parameters, ``seed``, and ``code_version_hash`` (the current
``git rev-parse HEAD`` of the ``code/`` worktree). Bumping any of
these invalidates every prior cache entry automatically, which is
the desired behaviour — a code edit that changes how the artefact
is constructed must not silently reuse a stale pickle.

Reproducibility prerequisite
----------------------------
The cache should be populated under
``OMP_NUM_THREADS=1`` and ``NUMBA_NUM_THREADS=1``. RAPTOR's UMAP+GMM
clustering and several of GraphRAG's NumPy/NetworkX paths are
non-deterministic across thread counts; without these pins the
artefact built on machine A is not guaranteed bit-identical to the
artefact built on machine B even at the same code version. The
RAPTOR and GraphRAG runners already set these env vars on import as
a defensive default, but cache producers running in non-standard
environments should set them explicitly before invoking the
pipeline.

Storage layout
--------------
``<root>/<arch>/<paper_id>/<key_hash>/``
    artifact.pkl       pickled state object (HIGHEST_PROTOCOL)
    build_meta.json    structured metadata + replay-able ledger rows
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pilot.ledger import CostLedger, Stage


# Pickle protocol pinned to highest available so artefact bytes are
# stable across consumer processes at the same Python minor version.
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def default_cache_root() -> Path:
    """Default disk-cache root: ``code/outputs/preprocess_cache``.

    Resolved relative to this module so callers from any working
    directory land in the same place. ``outputs/*`` is gitignored at
    the ``code/`` repo so no extra ignore entry is needed.
    """
    return Path(__file__).resolve().parents[2] / "outputs" / "preprocess_cache"


def code_version_hash() -> str:
    """Return ``git rev-parse HEAD`` for the ``code/`` worktree.

    The cache key includes this so any code edit (e.g. tweaking
    RAPTOR's leaf chunk size, GraphRAG's gleaning prompt) invalidates
    every prior cache entry. Falls back to a sentinel marker when git
    is not available or the working tree is not a git repo, which
    keeps tests runnable in tmpdirs without polluting key space; in
    that case the cache key is still deterministic per-process but
    cross-process determinism is the caller's responsibility.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "no-git-head"


class CacheRequiredMiss(RuntimeError):
    """Raised when ``--cache-required`` is set and a disk-cache lookup
    misses on RAPTOR or GraphRAG.

    Use case: the cross-candidate rerun must guarantee byte-identical
    contexts. A silent rebuild here would re-introduce the very
    determinism bug the disk cache exists to fix. The caller catches
    this at the orchestrator level and aborts the whole run rather
    than producing a half-valid result set.
    """


# ──────────────────────────────────────────────────────────────────────
# Cache key construction
# ──────────────────────────────────────────────────────────────────────

# Canonical default values, kept here so the cache key remains stable
# when callers don't override them. Mirrored from the architecture
# modules' module-level constants — when an arch module changes one of
# these, ``code_version_hash`` invalidates the prior entry on its
# own; we just need a fallback that matches today's behaviour.
_RAPTOR_DEFAULT_KEY_INPUTS = {
    "chunk_size_tokens": 100,       # tb_max_tokens, RAPTOR §3.1 leaf
    "chunk_overlap_tokens": 0,
    "max_layers": 5,                # tb_num_layers
    "summarization_max_tokens": 200,
    "tr_threshold": 0.5,
    "tr_top_k": 10,
    "retrieval_max_tokens": 2000,
    "selection_mode": "top_k",
}

_GRAPHRAG_DEFAULT_KEY_INPUTS = {
    "chunk_size_tokens": 600,       # Edge et al. §4.1.1
    "chunk_overlap_tokens": 100,
    "max_gleanings": 1,
    "top_k_entities": 10,
    "top_k_relationships": 10,
    "max_context_tokens": 8000,
    "community_prop": 0.15,
    "text_unit_prop": 0.50,
    "community_report_max_tokens": 2000,
}


def build_cache_key_inputs(
    *,
    architecture: str,
    paper_id: str,
    dataset: str,
    summary_model: str,
    summary_temperature: float,
    encoder_model: str,
    seed: int = 42,
    arch_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the dict that will be canonicalised + hashed into the key.

    ``arch_overrides`` lets a caller pin a non-default chunk size or
    retrieval budget for an experimental sweep; the override is folded
    into the cache key so the swept variants get separate cache
    entries. ``code_version_hash`` is added here so it lives next to
    the other key inputs in ``build_meta.json`` for forensics.
    """
    if architecture == "raptor":
        arch_inputs = dict(_RAPTOR_DEFAULT_KEY_INPUTS)
    elif architecture == "graphrag":
        arch_inputs = dict(_GRAPHRAG_DEFAULT_KEY_INPUTS)
    else:
        raise ValueError(
            f"disk cache only applies to raptor/graphrag, got {architecture!r}"
        )
    if arch_overrides:
        arch_inputs.update(arch_overrides)

    return {
        "architecture": architecture,
        "paper_id": paper_id,
        "dataset": dataset,
        "summary_model": summary_model,
        "summary_temperature": summary_temperature,
        "encoder_model": encoder_model,
        "seed": seed,
        "code_version_hash": code_version_hash(),
        **arch_inputs,
    }


def hash_cache_key(cache_key_inputs: dict[str, Any]) -> str:
    """SHA-256 of the canonical-JSON encoding of the key dict.

    ``sort_keys=True`` + ``separators=(",", ":")`` guarantees the same
    dict hashes the same regardless of insertion order or
    pretty-printing. 16 hex characters of the digest are enough to
    avoid collisions across the entire workspace's plausible cache
    population (~thousands of entries) and keep directory names short
    on Windows path-length-limited filesystems.
    """
    payload = json.dumps(
        cache_key_inputs, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────
# On-disk entry I/O
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """In-memory representation of one cache entry after load.

    ``state`` is the unpickled artefact (a ``_RaptorState`` or
    ``_GraphRAGState`` instance). ``build_meta`` is the parsed
    metadata JSON. ``entry_dir`` is the directory the entry was
    loaded from — useful for tests and forensics.
    """
    state: object
    build_meta: dict[str, Any]
    entry_dir: Path


def _entry_dir(
    cache_root: Path, architecture: str, paper_id: str, key_hash: str,
) -> Path:
    # Sanitise paper_id for cross-platform path safety: NovelQA novel
    # ids are short alnum tags (B01, B08, ...), QASPER paper ids are
    # alnum + hyphen — both pass through unchanged. Defensive replace
    # for any oddball input so we never write outside ``cache_root``.
    safe_paper = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in paper_id
    )
    return cache_root / architecture / safe_paper / key_hash


def load_cache_entry(
    *,
    architecture: str,
    paper_id: str,
    key_hash: str,
    cache_root: Path | None = None,
) -> CacheEntry | None:
    """Return the cached entry if present, else None.

    Both ``artifact.pkl`` and ``build_meta.json`` must exist for the
    entry to count as present; a half-written entry (process crashed
    mid-save) is treated as miss so the caller rebuilds cleanly.
    """
    root = cache_root if cache_root is not None else default_cache_root()
    entry = _entry_dir(root, architecture, paper_id, key_hash)
    pkl = entry / "artifact.pkl"
    meta = entry / "build_meta.json"
    if not (pkl.exists() and meta.exists()):
        return None
    try:
        with open(pkl, "rb") as fh:
            state = pickle.load(fh)
        with open(meta, encoding="utf-8") as fh:
            build_meta = json.load(fh)
    except (pickle.UnpicklingError, json.JSONDecodeError, EOFError, OSError):
        # Corrupt entry: treat as miss. Cleanup is left to operator —
        # silently deleting would mask a real disk-corruption signal.
        return None
    return CacheEntry(state=state, build_meta=build_meta, entry_dir=entry)


def save_cache_entry(
    *,
    architecture: str,
    paper_id: str,
    key_hash: str,
    state: object,
    build_meta: dict[str, Any],
    cache_root: Path | None = None,
) -> Path:
    """Atomically persist a built artefact + its build metadata.

    ``artifact.pkl`` is written first to a ``.tmp`` sibling and
    ``os.replace``-d into place; ``build_meta.json`` is written
    similarly. Atomic-replace semantics on a single filesystem
    guarantee a reader either sees both files or neither, so a
    concurrent reader can never observe a half-written entry.
    """
    root = cache_root if cache_root is not None else default_cache_root()
    entry = _entry_dir(root, architecture, paper_id, key_hash)
    entry.mkdir(parents=True, exist_ok=True)

    pkl = entry / "artifact.pkl"
    pkl_tmp = entry / "artifact.pkl.tmp"
    with open(pkl_tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=PICKLE_PROTOCOL)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except (OSError, AttributeError):
            pass
    os.replace(pkl_tmp, pkl)

    meta = entry / "build_meta.json"
    meta_tmp = entry / "build_meta.json.tmp"
    with open(meta_tmp, "w", encoding="utf-8") as fh:
        json.dump(build_meta, fh, indent=2, sort_keys=True, ensure_ascii=True)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except (OSError, AttributeError):
            pass
    os.replace(meta_tmp, meta)

    return entry


# ──────────────────────────────────────────────────────────────────────
# Ledger interaction: capture-on-build, replay-on-hit
# ──────────────────────────────────────────────────────────────────────

def _read_jsonl_tail(path: Path, from_byte_offset: int) -> list[dict[str, Any]]:
    """Return all JSON rows in ``path`` from byte ``from_byte_offset`` onwards.

    Used to snapshot the ledger rows produced during one runner
    invocation: snapshot the file size before the call, then read the
    tail after. Done at the byte level rather than the line count
    because concurrent ledger writes by other architectures in
    interleaved tests could otherwise miscount lines (in practice
    ``run_dry_run`` is serial, but byte-offset is robust either way).
    """
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        fh.seek(from_byte_offset)
        blob = fh.read()
    rows: list[dict[str, Any]] = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def ledger_byte_size(ledger: CostLedger) -> int:
    """Current size in bytes of the ledger file.

    The cache wrapper snapshots this before a build-phase runner call
    and uses ``capture_build_rows_since`` after the call returns to
    isolate exactly the rows the runner produced.
    """
    if not ledger.path.exists():
        return 0
    return ledger.path.stat().st_size


def capture_build_rows_since(
    *,
    ledger: CostLedger,
    architecture: str,
    from_byte_offset: int,
) -> list[dict[str, Any]]:
    """Return ``stage=preprocess`` rows for ``architecture`` written
    after ``from_byte_offset``.

    Only preprocess-stage rows are stored in ``build_meta``; the
    per-question retrieval + generate rows belong to the candidate's
    on-line cost and are not part of the cached artefact. Filtering by
    architecture protects against an interleaved write by another arch
    on the same ledger (defensive — ``run_dry_run`` is serial).
    """
    tail = _read_jsonl_tail(ledger.path, from_byte_offset)
    out: list[dict[str, Any]] = []
    for row in tail:
        if row.get("architecture") != architecture:
            continue
        if row.get("stage") != Stage.PREPROCESS.value:
            continue
        out.append(row)
    return out


def summarise_build_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, float | int]:
    """Aggregate the build-cost totals stored in ``build_meta.json``.

    These mirror what the per-candidate ledger replay will sum to,
    making it trivial to assert the replay is faithful in tests
    without re-parsing every row.
    """
    total_in = 0
    total_out = 0
    total_gpu = 0.0
    total_wall = 0.0
    for r in rows:
        total_in += int(r.get("uncached_input_tokens", 0) or 0)
        total_in += int(r.get("cached_input_tokens", 0) or 0)
        total_out += int(r.get("output_tokens", 0) or 0)
        total_gpu += float(r.get("gpu_s_estimate", 0.0) or 0.0)
        total_wall += float(r.get("wallclock_s", 0.0) or 0.0)
    return {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_gpu_seconds": total_gpu,
        "total_wallclock_seconds": total_wall,
    }


def replay_build_ledger(
    *,
    ledger: CostLedger,
    build_meta: dict[str, Any],
    target_run_index: int = 0,
) -> int:
    """Write the cached build rows into ``ledger`` as if the candidate
    had paid for them.

    Cost-attribution invariant: each candidate must show the
    preprocessing cost in its own ledger so the deployment-cost
    formula ``bar_C_deploy(c, n) = (C_off + sum C_on) / n`` evaluates
    to the same value whether the artefact was cached or rebuilt.

    Each replayed row gets:
      - a fresh ``timestamp`` (the moment of replay)
      - the caller's ``target_run_index`` (so resume runs at
        non-zero ``run_index`` stay consistent with the rest of the
        candidate's row stream)
      - ``cache_loaded=true``
      - ``source_run_id=<original build_run_id>``
      - ``source_timestamp=<original timestamp>`` — forensic trail
        for a reviewer reconciling cache hits against the originating
        build run.

    All other fields (model, stage, uncached/cached/output tokens,
    gpu_s_estimate, prompt_hash, response_hash, ...) are copied
    verbatim so the cost arithmetic is bit-identical.

    Returns the number of rows replayed.
    """
    source_run_id = build_meta.get("build_run_id", "unknown")
    rows = build_meta.get("ledger_rows", []) or []
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    with open(ledger.path, "a", encoding="utf-8") as fh:
        for row in rows:
            replayed = dict(row)
            replayed["run_index"] = target_run_index
            replayed["timestamp"] = now
            replayed["cache_loaded"] = True
            replayed["source_run_id"] = source_run_id
            if "timestamp" in row:
                replayed["source_timestamp"] = row["timestamp"]
            fh.write(json.dumps(replayed, separators=(",", ":")))
            fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except (OSError, AttributeError):
            pass
    return len(rows)


def make_build_meta(
    *,
    cache_key_inputs: dict[str, Any],
    build_run_id: str,
    summary_model: str,
    encoder_model: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the ``build_meta.json`` dict from build-phase rows.

    Stored fields mirror the spec in the cache module docstring; the
    raw rows are included verbatim so ``replay_build_ledger`` can
    reproduce them on hit.
    """
    totals = summarise_build_rows(rows)
    return {
        "build_timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "build_run_id": build_run_id,
        "summary_model": summary_model,
        "encoder_model": encoder_model,
        "code_version_hash": cache_key_inputs.get(
            "code_version_hash", code_version_hash()
        ),
        "cache_key_inputs": cache_key_inputs,
        **totals,
        "ledger_rows": rows,
    }


# ──────────────────────────────────────────────────────────────────────
# Stable artefact fingerprint (for byte-equality assertions)
# ──────────────────────────────────────────────────────────────────────

def artifact_fingerprint(path_or_bytes) -> str:
    """SHA-256 of an artefact file's contents, or of a raw bytes blob.

    Used by ``test_preprocess_cache`` to assert that two cache loads
    of the same key return byte-identical pickles. Comparing pickle
    bytes directly is brittle across Python minor versions; this
    function exists so the test can document the intent ("the
    artefact bytes on disk are the SAME bytes that were written")
    without coupling to pickle internals.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return hashlib.sha256(bytes(path_or_bytes)).hexdigest()
    p = Path(path_or_bytes)
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
