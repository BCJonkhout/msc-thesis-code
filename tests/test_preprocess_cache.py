"""Tests for the persistent on-disk preprocessing cache.

The cross-candidate determinism rerun depends on three contracts:

  1. An artefact saved by one process is loaded byte-identically by
     a different process at the same cache key.
  2. The cache key invalidates correctly when any of its inputs
     changes (so a code edit that alters how the artefact is built
     can never silently serve a stale pickle).
  3. On cache hit, the per-candidate ledger gets synthetic
     preprocess-stage rows whose token / gpu / wallclock totals
     reproduce the original build's numbers exactly — the cache is
     a measurement-efficiency optimisation, not a cost discount.

These tests exercise all three. Building real RAPTOR or GraphRAG
artefacts requires Ollama + an LLM, which is out of scope for unit
tests; we instead manufacture artefact-shaped Python objects (a
``_RaptorState`` carrying a tiny mock ``ra``; a ``_GraphRAGState``
carrying small graph data) and exercise the save/load/replay code
paths against those. The cache module treats artefacts opaquely
(pickle.dump / pickle.load) so artefact shape is irrelevant to the
behaviour under test.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from pilot.ledger import CostLedger, Stage
from pilot.preprocess_cache import (
    CacheRequiredMiss,
    artifact_fingerprint,
    build_cache_key_inputs,
    capture_build_rows_since,
    hash_cache_key,
    ledger_byte_size,
    load_cache_entry,
    make_build_meta,
    replay_build_ledger,
    save_cache_entry,
    summarise_build_rows,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic artefacts
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _SyntheticRaptorTree:
    """Stand-in for the ``raptor.RetrievalAugmentation`` instance.

    The real object is large and depends on UMAP/GMM internals; the
    cache treats it as opaque bytes (``pickle.HIGHEST_PROTOCOL``), so
    a small deterministic dataclass exercises the same code path.
    Concrete fields mirror the real artefact's character: a list of
    node texts + a list of fixed-length embedding vectors, both of
    which would diverge across rebuilds with a non-deterministic
    summariser, which is the divergence the cache exists to prevent.
    """
    node_texts: list[str]
    node_embeddings: list[list[float]]
    config_signature: str


@dataclass
class _SyntheticGraphRAGGraph:
    """Stand-in for the ``_GraphRAGState`` dataclass.

    A tiny entity + relationship + community + report bundle plus an
    entity-vector list. Same pickle round-trip as the real object.
    """
    chunks: list[str]
    entities: list[dict]
    relationships: list[dict]
    communities: list[list[str]]
    reports: list[dict]
    entity_vecs: list[list[float]]
    embed_dim: int


def _make_raptor_artifact() -> _SyntheticRaptorTree:
    return _SyntheticRaptorTree(
        node_texts=[
            "leaf chunk 1: the document opens with a definition.",
            "leaf chunk 2: section two recapitulates the method.",
            "cluster summary L1: definitions + method recap.",
        ],
        node_embeddings=[
            [0.1, 0.2, 0.3, 0.4],
            [0.11, 0.22, 0.33, 0.44],
            [0.105, 0.21, 0.315, 0.42],
        ],
        config_signature="tb_max_tokens=100|num_layers=5|sumlen=200",
    )


def _make_graphrag_artifact() -> _SyntheticGraphRAGGraph:
    return _SyntheticGraphRAGGraph(
        chunks=["chunk A text", "chunk B text"],
        entities=[
            {"name": "Alice", "type": "person", "description": "Author."},
            {"name": "QASPER", "type": "dataset", "description": "Eval set."},
        ],
        relationships=[
            {"source": "Alice", "target": "QASPER",
             "description": "Alice evaluates on QASPER."},
        ],
        communities=[["Alice", "QASPER"]],
        reports=[{"community_id": 0, "text": "## Title\nAlice + QASPER\n"}],
        entity_vecs=[[0.5, 0.6, 0.7], [0.51, 0.61, 0.71]],
        embed_dim=3,
    )


def _make_build_rows(architecture: str) -> list[dict]:
    """Synthetic preprocess-stage rows mirroring a real build trace."""
    return [
        {
            "timestamp": "2026-05-16T10:00:00.000000+00:00",
            "architecture": architecture,
            "stage": Stage.PREPROCESS.value,
            "model": "bge-m3",
            "run_index": 0,
            "uncached_input_tokens": 1250,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "wallclock_s": 1.234,
            "gpu_s_estimate": 0.85,
            "prompt_hash": "a" * 64,
            "response_hash": "b" * 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": None,
            "failed": False,
            "provider_request_id": None,
        },
        {
            "timestamp": "2026-05-16T10:00:01.000000+00:00",
            "architecture": architecture,
            "stage": Stage.PREPROCESS.value,
            "model": "gemini-flash-lite",
            "run_index": 0,
            "uncached_input_tokens": 800,
            "cached_input_tokens": 100,
            "output_tokens": 220,
            "wallclock_s": 2.5,
            "gpu_s_estimate": 0.0,
            "prompt_hash": "c" * 64,
            "response_hash": "d" * 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 500,
            "failed": False,
            "provider_request_id": "req-xyz",
        },
    ]


# ──────────────────────────────────────────────────────────────────────
# Cache key construction + invalidation
# ──────────────────────────────────────────────────────────────────────

class TestCacheKey:
    def test_same_inputs_produce_same_hash(self):
        a = build_cache_key_inputs(
            architecture="raptor", paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        b = build_cache_key_inputs(
            architecture="raptor", paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        assert hash_cache_key(a) == hash_cache_key(b)

    @pytest.mark.parametrize(
        "field,override",
        [
            ("paper_id", "B02"),
            ("dataset", "qasper"),
            ("summary_model", "gpt-4o-mini"),
            ("summary_temperature", 0.7),
            ("encoder_model", "text-embedding-3-small"),
        ],
    )
    def test_changing_any_input_changes_the_hash(self, field, override):
        baseline_kwargs = dict(
            architecture="raptor", paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        baseline = hash_cache_key(build_cache_key_inputs(**baseline_kwargs))
        modified_kwargs = dict(baseline_kwargs)
        modified_kwargs[field] = override
        modified = hash_cache_key(build_cache_key_inputs(**modified_kwargs))
        assert baseline != modified, f"hash failed to change when {field} changed"

    def test_raptor_and_graphrag_have_disjoint_keys(self):
        # Same paper, same models — different architectures must
        # never collide because the per-architecture parameter blocks
        # are different and ``architecture`` is itself part of the
        # key.
        common = dict(
            paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        r = hash_cache_key(
            build_cache_key_inputs(architecture="raptor", **common)
        )
        g = hash_cache_key(
            build_cache_key_inputs(architecture="graphrag", **common)
        )
        assert r != g

    def test_arch_override_invalidates_default_key(self):
        baseline = hash_cache_key(build_cache_key_inputs(
            architecture="raptor", paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        ))
        overridden = hash_cache_key(build_cache_key_inputs(
            architecture="raptor", paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
            arch_overrides={"max_layers": 7},
        ))
        assert baseline != overridden


# ──────────────────────────────────────────────────────────────────────
# Round-trip: save + load (in-process)
# ──────────────────────────────────────────────────────────────────────

class TestRaptorRoundTrip:
    def test_save_then_load_returns_equal_state(self, tmp_path: Path):
        state = _make_raptor_artifact()
        key_inputs = build_cache_key_inputs(
            architecture="raptor", paper_id="paperA", dataset="qasper",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        key_hash = hash_cache_key(key_inputs)
        build_meta = make_build_meta(
            cache_key_inputs=key_inputs,
            build_run_id="run-001",
            summary_model="gemini-flash-lite",
            encoder_model="bge-m3",
            rows=_make_build_rows("raptor"),
        )

        save_cache_entry(
            architecture="raptor", paper_id="paperA",
            key_hash=key_hash, state=state, build_meta=build_meta,
            cache_root=tmp_path,
        )

        entry = load_cache_entry(
            architecture="raptor", paper_id="paperA",
            key_hash=key_hash, cache_root=tmp_path,
        )
        assert entry is not None
        assert entry.state == state
        assert entry.build_meta["build_run_id"] == "run-001"
        assert entry.build_meta["summary_model"] == "gemini-flash-lite"
        assert entry.build_meta["encoder_model"] == "bge-m3"
        # Totals stored in build_meta must agree with the row-level sum.
        totals = summarise_build_rows(_make_build_rows("raptor"))
        assert entry.build_meta["total_input_tokens"] == totals["total_input_tokens"]
        assert entry.build_meta["total_output_tokens"] == totals["total_output_tokens"]
        assert entry.build_meta["total_gpu_seconds"] == totals["total_gpu_seconds"]

    def test_artifact_pkl_bytes_are_deterministic(self, tmp_path: Path):
        """Two saves of the same Python object produce byte-identical
        pickles. This is the core invariant the cross-candidate rerun
        depends on: every consumer that loads this artefact reads the
        SAME bytes off disk."""
        state = _make_raptor_artifact()
        meta = {"build_run_id": "r", "ledger_rows": []}
        save_cache_entry(
            architecture="raptor", paper_id="P", key_hash="k1",
            state=state, build_meta=meta, cache_root=tmp_path,
        )
        save_cache_entry(
            architecture="raptor", paper_id="P", key_hash="k2",
            state=state, build_meta=meta, cache_root=tmp_path,
        )
        fp1 = artifact_fingerprint(
            tmp_path / "raptor" / "P" / "k1" / "artifact.pkl"
        )
        fp2 = artifact_fingerprint(
            tmp_path / "raptor" / "P" / "k2" / "artifact.pkl"
        )
        assert fp1 == fp2

    def test_load_in_separate_process_returns_same_bytes(self, tmp_path: Path):
        """Subprocess loads the pickle and prints its sha256. This is
        the cross-process determinism the disk cache exists to
        provide: candidate B's process reads the exact bytes
        candidate A's process wrote."""
        state = _make_raptor_artifact()
        meta = {"build_run_id": "r", "ledger_rows": []}
        save_cache_entry(
            architecture="raptor", paper_id="P", key_hash="kX",
            state=state, build_meta=meta, cache_root=tmp_path,
        )
        artifact_path = tmp_path / "raptor" / "P" / "kX" / "artifact.pkl"
        in_process_fp = artifact_fingerprint(artifact_path)

        # Spawn a fresh Python interpreter, hand it the artefact
        # path, and have it print the sha256 of the bytes it sees.
        # Subprocess gets the same PYTHONPATH so it can import the
        # synthetic dataclass for unpickling.
        helper = (
            "import hashlib, sys; "
            "p = sys.argv[1]; "
            "print(hashlib.sha256(open(p, 'rb').read()).hexdigest())"
        )
        env = dict(os.environ)
        out = subprocess.run(
            [sys.executable, "-c", helper, str(artifact_path)],
            capture_output=True, text=True, check=True, env=env,
        )
        subprocess_fp = out.stdout.strip()
        assert subprocess_fp == in_process_fp


class TestGraphRagRoundTrip:
    def test_save_then_load_returns_equal_state(self, tmp_path: Path):
        state = _make_graphrag_artifact()
        key_inputs = build_cache_key_inputs(
            architecture="graphrag", paper_id="B01", dataset="novelqa",
            summary_model="gemini-flash-lite", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        key_hash = hash_cache_key(key_inputs)
        build_meta = make_build_meta(
            cache_key_inputs=key_inputs,
            build_run_id="run-graphrag-001",
            summary_model="gemini-flash-lite",
            encoder_model="bge-m3",
            rows=_make_build_rows("graphrag"),
        )

        save_cache_entry(
            architecture="graphrag", paper_id="B01",
            key_hash=key_hash, state=state, build_meta=build_meta,
            cache_root=tmp_path,
        )

        entry = load_cache_entry(
            architecture="graphrag", paper_id="B01",
            key_hash=key_hash, cache_root=tmp_path,
        )
        assert entry is not None
        assert entry.state == state
        assert entry.build_meta["build_run_id"] == "run-graphrag-001"

    def test_load_in_separate_process_returns_same_bytes(self, tmp_path: Path):
        state = _make_graphrag_artifact()
        meta = {"build_run_id": "r", "ledger_rows": []}
        save_cache_entry(
            architecture="graphrag", paper_id="P", key_hash="kG",
            state=state, build_meta=meta, cache_root=tmp_path,
        )
        artifact_path = tmp_path / "graphrag" / "P" / "kG" / "artifact.pkl"
        in_process_fp = artifact_fingerprint(artifact_path)
        helper = (
            "import hashlib, sys; "
            "print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())"
        )
        env = dict(os.environ)
        out = subprocess.run(
            [sys.executable, "-c", helper, str(artifact_path)],
            capture_output=True, text=True, check=True, env=env,
        )
        assert out.stdout.strip() == in_process_fp


class TestCacheMiss:
    def test_missing_entry_returns_none(self, tmp_path: Path):
        assert load_cache_entry(
            architecture="raptor", paper_id="nope",
            key_hash="deadbeef", cache_root=tmp_path,
        ) is None

    def test_half_written_entry_is_treated_as_miss(self, tmp_path: Path):
        """Only artifact.pkl present (build_meta.json missing) =>
        miss. Models the case where a producer crashed mid-save: the
        consumer must rebuild rather than load a half-valid entry."""
        entry_dir = tmp_path / "raptor" / "P" / "kHalf"
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / "artifact.pkl").write_bytes(b"\x80\x04N.")
        # build_meta.json deliberately not written.
        result = load_cache_entry(
            architecture="raptor", paper_id="P", key_hash="kHalf",
            cache_root=tmp_path,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# Ledger replay
# ──────────────────────────────────────────────────────────────────────

class TestLedgerReplay:
    def test_replay_reproduces_token_and_gpu_totals(self, tmp_path: Path):
        """The whole point of the replay: each candidate's ledger
        sums to the same per-paper preprocessing cost regardless of
        whether the candidate built the artefact or loaded it from
        cache. ``bar_C_deploy`` per the cost model is unchanged per
        candidate."""
        ledger = CostLedger(run_id="cand-A", root=tmp_path)
        original_rows = _make_build_rows("raptor")
        build_meta = make_build_meta(
            cache_key_inputs={"code_version_hash": "abc"},
            build_run_id="origin-run",
            summary_model="gemini-flash-lite",
            encoder_model="bge-m3",
            rows=original_rows,
        )

        n = replay_build_ledger(ledger=ledger, build_meta=build_meta)
        assert n == len(original_rows)

        replayed_rows = ledger.read()
        assert len(replayed_rows) == len(original_rows)
        # Token + gpu_s totals must match the build_meta summary
        # exactly. (CostLedger.read() returns CallRecord instances
        # which drop the cache_loaded/source_run_id auxiliary fields;
        # we re-read the raw JSONL for those.)
        replay_input = sum(
            r.uncached_input_tokens + r.cached_input_tokens
            for r in replayed_rows
        )
        replay_output = sum(r.output_tokens for r in replayed_rows)
        replay_gpu = sum(r.gpu_s_estimate for r in replayed_rows)
        assert replay_input == build_meta["total_input_tokens"]
        assert replay_output == build_meta["total_output_tokens"]
        assert replay_gpu == pytest.approx(build_meta["total_gpu_seconds"])

    def test_replay_marks_rows_as_cache_loaded(self, tmp_path: Path):
        ledger = CostLedger(run_id="cand-B", root=tmp_path)
        build_meta = make_build_meta(
            cache_key_inputs={"code_version_hash": "abc"},
            build_run_id="origin-run-XYZ",
            summary_model="gemini-flash-lite",
            encoder_model="bge-m3",
            rows=_make_build_rows("raptor"),
        )
        replay_build_ledger(ledger=ledger, build_meta=build_meta)

        # The CallRecord schema doesn't include cache_loaded /
        # source_run_id, so re-parse the JSONL to see them.
        raw = [json.loads(l) for l in ledger.path.read_text(
            encoding="utf-8").splitlines() if l.strip()]
        assert len(raw) == len(_make_build_rows("raptor"))
        for row in raw:
            assert row["cache_loaded"] is True
            assert row["source_run_id"] == "origin-run-XYZ"
            assert "source_timestamp" in row

    def test_capture_build_rows_filters_by_arch_and_stage(self, tmp_path: Path):
        """Snapshot-after-call captures must isolate the runner's
        preprocess rows from interleaved per-question retrieval +
        generate rows produced by the same runner call."""
        ledger = CostLedger(run_id="cap-test", root=tmp_path)
        offset_before = ledger_byte_size(ledger)
        with ledger.log_call(
            architecture="raptor", stage=Stage.PREPROCESS,
            model="bge-m3", prompt="leaf 1",
        ) as rec:
            rec.uncached_input_tokens = 10
            rec.output_tokens = 0
        with ledger.log_call(
            architecture="raptor", stage=Stage.RETRIEVAL,
            model="bge-m3", prompt="query",
        ) as rec:
            rec.uncached_input_tokens = 5
            rec.output_tokens = 0
        with ledger.log_call(
            architecture="naive_rag", stage=Stage.PREPROCESS,
            model="bge-m3", prompt="other-arch chunk",
        ) as rec:
            rec.uncached_input_tokens = 99
            rec.output_tokens = 0
        captured = capture_build_rows_since(
            ledger=ledger, architecture="raptor",
            from_byte_offset=offset_before,
        )
        assert len(captured) == 1  # only the raptor preprocess row
        assert captured[0]["uncached_input_tokens"] == 10
        assert captured[0]["stage"] == Stage.PREPROCESS.value


# ──────────────────────────────────────────────────────────────────────
# CLI --cache-required behaviour
# ──────────────────────────────────────────────────────────────────────

def _write_qasper_pool(data_root: Path, paper_id: str, qids: list[str]) -> None:
    qasper = data_root / "qasper"
    qasper.mkdir(parents=True, exist_ok=True)
    qas = [
        {"question_id": qid, "question": "?",
         "answers": [{"answer": {
             "free_form_answer": "yes",
             "evidence": [], "highlighted_evidence": [],
             "extractive_spans": [], "yes_no": True,
             "unanswerable": False,
         }}]}
        for qid in qids
    ]
    paper = {
        "paper_id": paper_id, "title": "paper", "abstract": "x",
        "full_text": [{"section_name": "s", "paragraphs": ["text"]}],
        "qas": qas,
    }
    (qasper / "dev.jsonl").write_text(
        json.dumps(paper) + "\n", encoding="utf-8"
    )
    (qasper / "calibration_pool.jsonl").write_text(
        "\n".join(
            json.dumps({"paper_id": paper_id, "question_id": qid, "question": "?"})
            for qid in qids
        ) + "\n",
        encoding="utf-8",
    )


class TestCacheRequiredFlag:
    def test_cache_required_miss_raises(self, tmp_path: Path, monkeypatch):
        """With --cache-required, a MISS on RAPTOR / GraphRAG must
        abort the run rather than silently rebuild. A silent rebuild
        would re-introduce the non-determinism the cache exists to
        prevent."""
        from unittest.mock import MagicMock
        from pilot.architectures import ArchitectureResult
        from pilot.cli import step_3_dry_run as cli

        data_root = tmp_path / "data"
        _write_qasper_pool(data_root, "paperA", ["q1"])

        # Runner stub: would build a fresh artefact on miss. With
        # --cache-required the dispatcher should raise BEFORE the
        # stub is ever called, so we record invocations and assert
        # the stub was never reached.
        invoked: list[bool] = []

        def fake_run_raptor(**kwargs):
            invoked.append(True)
            return ArchitectureResult(
                architecture="raptor", predicted_answer="z",
                preprocessing_state={"built": True},
            )

        monkeypatch.setattr(cli, "run_raptor", fake_run_raptor)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        with pytest.raises(CacheRequiredMiss):
            cli.run_dry_run(
                architectures=["raptor"],
                datasets=["qasper"],
                answerer_provider="gemini",
                answerer_model="m",
                embedder_model="bge-m3",
                naive_rag_top_k=8,
                data_root=data_root,
                out_dir=tmp_path / "out",
                cache_required=True,
                cache_root=tmp_path / "cache",
            )
        assert invoked == [], "runner should not be invoked when cache miss under --cache-required"

    def test_cache_required_hit_proceeds(self, tmp_path: Path, monkeypatch):
        """With --cache-required AND a populated entry, the
        dispatcher must load the artefact, replay the build rows,
        and call the runner with ``cached_state`` set."""
        from unittest.mock import MagicMock
        from pilot.architectures import ArchitectureResult
        from pilot.cli import step_3_dry_run as cli

        data_root = tmp_path / "data"
        _write_qasper_pool(data_root, "paperA", ["q1"])
        cache_root = tmp_path / "cache"

        # Pre-populate the cache for the (raptor, paperA) key the
        # dispatcher will compute. Mirror the exact key inputs the
        # dispatcher builds so the hash matches.
        key_inputs = build_cache_key_inputs(
            architecture="raptor", paper_id="paperA", dataset="qasper",
            summary_model="m", summary_temperature=0.0,
            encoder_model="bge-m3",
        )
        key_hash = hash_cache_key(key_inputs)
        cached_state = {"built_by": "candidate-A"}
        build_meta = make_build_meta(
            cache_key_inputs=key_inputs,
            build_run_id="origin-run",
            summary_model="m",
            encoder_model="bge-m3",
            rows=_make_build_rows("raptor"),
        )
        save_cache_entry(
            architecture="raptor", paper_id="paperA",
            key_hash=key_hash, state=cached_state, build_meta=build_meta,
            cache_root=cache_root,
        )

        # Runner stub records the cached_state it received.
        received_state: list[object] = []

        def fake_run_raptor(**kwargs):
            received_state.append(kwargs.get("cached_state"))
            return ArchitectureResult(
                architecture="raptor", predicted_answer="z",
                preprocessing_state=kwargs.get("cached_state") or {"built": True},
            )

        monkeypatch.setattr(cli, "run_raptor", fake_run_raptor)
        monkeypatch.setattr(cli, "get_provider", lambda name: MagicMock())

        summary = cli.run_dry_run(
            architectures=["raptor"],
            datasets=["qasper"],
            answerer_provider="gemini",
            answerer_model="m",
            embedder_model="bge-m3",
            naive_rag_top_k=8,
            data_root=data_root,
            out_dir=tmp_path / "out",
            cache_required=True,
            cache_root=cache_root,
        )
        assert received_state == [cached_state]
        assert summary["failures_count"] == 0
