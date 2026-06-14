"""Robustness + speed-up regressions for the main-study build path.

Covers the three fixes made after the long-document run stalled:

  1. RAPTOR reclustering terminates on a degenerate (non-splitting) leaf
     set instead of recursing until ``RecursionError`` (the failure seen on
     large NovelQA novels B12/B24/B38/B42).
  2. GraphRAG entity extraction is byte-for-byte deterministic regardless of
     ``PILOT_BUILD_CONCURRENCY`` (concurrent fetch, fixed-order merge).
  3. ``_load_failed_builds`` recovers the (arch, paper) set so a known-failed
     build is not re-attempted (and re-paid) on the document's other
     questions or on resume.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pilot.providers.base import AnswererProvider, CacheControl, ProviderResult


# ──────────────────────────────────────────────────────────────────────
# 1. RAPTOR reclustering must terminate on a non-splitting cluster
# ──────────────────────────────────────────────────────────────────────

def test_recluster_terminates_on_degenerate_cluster(monkeypatch):
    # Importing the runner first puts the vendored ``raptor`` package on the
    # path (module-level sys.path.insert), exactly as production does.
    import pilot.architectures.raptor  # noqa: F401
    from raptor import cluster_utils
    from raptor.tree_structures import Node

    nodes = [
        Node(text=f"word{i} " * 25, index=i, children=set(),
             embeddings={"EMB": [float(i)] * 4})
        for i in range(8)
    ]

    # Force the low-level clusterer to return a single cluster equal to its
    # input on every call — the exact condition that made the original code
    # recurse on the same node set forever.
    monkeypatch.setattr(
        cluster_utils, "perform_clustering",
        lambda embeddings, dim, threshold: [np.array([0.0]) for _ in range(len(embeddings))],
    )

    # max_length_in_cluster tiny so the (un-splittable) cluster is "too big"
    # and the recursion branch is taken. Must return, not RecursionError.
    out = cluster_utils.RAPTOR_Clustering.perform_clustering(
        nodes, "EMB", max_length_in_cluster=5,
    )
    assert len(out) == 1, "degenerate set should collapse to one accepted cluster"
    assert len(out[0]) == 8, "no nodes should be lost when accepting the oversized cluster"


def test_recluster_depth_ceiling_is_below_python_limit():
    import pilot.architectures.raptor  # noqa: F401
    from raptor import cluster_utils
    # The ceiling must leave ample stack headroom under the interpreter limit.
    assert cluster_utils._MAX_RECLUSTER_DEPTH < 100


# ──────────────────────────────────────────────────────────────────────
# 2. GraphRAG extraction determinism under concurrency
# ──────────────────────────────────────────────────────────────────────

class _ChunkDeterministicAnswerer(AnswererProvider):
    """Stateless, thread-safe answerer: response depends only on which chunk
    text appears in the prompt, so it is order- and concurrency-independent."""

    name = "chunkdet"

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls = 0
        self._clock = __import__("threading").Lock()

    def call(self, prompt: str, *, model: str, max_tokens=None,
             temperature: float = 0.0, top_p: float = 1.0,
             cache_control: CacheControl = CacheControl.DISABLED) -> ProviderResult:
        with self._clock:
            self.calls += 1
        idx = next((i for i, c in enumerate(self.chunks) if c in prompt), None)
        if idx is None:
            payload: dict = {"entities": [], "relationships": []}
        else:
            payload = {
                "entities": [
                    {"name": f"E{idx}", "type": "concept", "description": f"desc-{idx}"},
                    {"name": "SHARED", "type": "concept", "description": f"from-{idx}"},
                ],
                "relationships": [
                    {"source": f"E{idx}", "target": "SHARED",
                     "description": f"rel-{idx}", "weight": 5},
                ],
            }
        text = json.dumps(payload)
        return ProviderResult(
            text=text, uncached_input_tokens=1, cached_input_tokens=0,
            output_tokens=1, provider_request_id="req", wallclock_s=0.0,
        )


def _run_extraction(chunks, concurrency, tmp_path, monkeypatch):
    monkeypatch.setenv("PILOT_BUILD_CONCURRENCY", str(concurrency))
    from pilot.architectures.graphrag import _extract_entities_per_chunk
    from pilot.ledger import CostLedger

    ledger = CostLedger(run_id=f"t-{concurrency}", root=tmp_path)
    entities, relationships = _extract_entities_per_chunk(
        chunks,
        answerer=_ChunkDeterministicAnswerer(chunks),
        answerer_model="m",
        ledger=ledger,
        run_index=0,
    )
    ents = sorted(
        (e.name, e.type, e.description, tuple(e.text_unit_ids)) for e in entities
    )
    rels = [
        (r.source, r.target, r.description, r.text_unit_id, r.strength)
        for r in relationships
    ]
    return ents, rels


def test_graphrag_extraction_concurrency_is_deterministic(tmp_path, monkeypatch):
    chunks = [f"CHUNKMARKER{i}xyz" for i in range(6)]
    seq = _run_extraction(chunks, 1, tmp_path / "seq", monkeypatch)
    par = _run_extraction(chunks, 4, tmp_path / "par", monkeypatch)
    assert seq == par, "concurrent extraction must match the sequential merge exactly"
    # Sanity: the cross-chunk SHARED entity accumulated every chunk, in order.
    shared = [e for e in seq[0] if e[0] == "SHARED"][0]
    assert shared[3] == tuple(range(6)), "SHARED entity must record every chunk, in order"


# ──────────────────────────────────────────────────────────────────────
# 3. Failed-build skip set
# ──────────────────────────────────────────────────────────────────────

def test_load_failed_builds_roundtrip_and_tolerates_torn_line(tmp_path):
    from pilot.cli.step_3_dry_run import _load_failed_builds

    run_dir = tmp_path
    path = run_dir / "build_failures.jsonl"
    path.write_text(
        json.dumps({"architecture": "raptor", "paper_id": "B12", "question_id": "q1"}) + "\n"
        + json.dumps({"architecture": "graphrag", "paper_id": "B12", "question_id": "q2"}) + "\n"
        + json.dumps({"architecture": "raptor", "paper_id": "B12", "question_id": "q3"}) + "\n"
        + '{"architecture": "raptor", "paper_id": "B99"',  # torn final line
        encoding="utf-8",
    )
    failed = _load_failed_builds(run_dir)
    assert failed == {("raptor", "B12"), ("graphrag", "B12")}


def test_load_failed_builds_missing_file(tmp_path):
    from pilot.cli.step_3_dry_run import _load_failed_builds
    assert _load_failed_builds(tmp_path) == set()


# ──────────────────────────────────────────────────────────────────────
# 4. Resumable build-call cache
# ──────────────────────────────────────────────────────────────────────

def test_build_call_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PILOT_BUILD_CALL_CACHE", "")          # enable
    monkeypatch.setenv("PILOT_BUILD_CALL_CACHE_DIR", str(tmp_path / "cc"))
    from pilot import build_call_cache as bcc
    assert bcc.get("k", "m", "", "req") is None
    bcc.put("k", "m", "", "req", "resp")
    assert bcc.get("k", "m", "", "req") == "resp"
    # Different request text / kind / model => independent entries.
    assert bcc.get("k", "m", "", "other") is None
    assert bcc.get("other", "m", "", "req") is None


def test_build_call_cache_resumes_extraction_without_respending(tmp_path, monkeypatch):
    """A second build over the same chunks reuses the cached calls — the whole
    point: an interrupted big-novel build resumes instead of re-paying."""
    monkeypatch.setenv("PILOT_BUILD_CALL_CACHE", "")          # enable
    monkeypatch.setenv("PILOT_BUILD_CALL_CACHE_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("PILOT_BUILD_CONCURRENCY", "1")
    from pilot.architectures.graphrag import _extract_entities_per_chunk
    from pilot.ledger import CostLedger

    chunks = [f"CHUNKMARKER{i}xyz" for i in range(4)]
    ans = _ChunkDeterministicAnswerer(chunks)

    led1 = CostLedger(run_id="cc1", root=tmp_path / "r1")
    e1, _ = _extract_entities_per_chunk(
        chunks, answerer=ans, answerer_model="m", ledger=led1, run_index=0)
    first = ans.calls
    assert first > 0

    # Re-run (fresh ledger, same cache dir): every call must be served from the
    # cache, so the provider is not hit again.
    led2 = CostLedger(run_id="cc2", root=tmp_path / "r2")
    e2, _ = _extract_entities_per_chunk(
        chunks, answerer=ans, answerer_model="m", ledger=led2, run_index=0)
    assert ans.calls == first, "second build must make zero new provider calls"

    norm = lambda ents: sorted((e.name, e.type, e.description, tuple(e.text_unit_ids)) for e in ents)
    assert norm(e1) == norm(e2), "cached build must be identical to the original"
