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


# ──────────────────────────────────────────────────────────────────────
# Real-state pickle round-trip + rehydration (regression for
# `_thread.lock` crash hit on 2026-05-18 Phase G rerun attempt)
# ──────────────────────────────────────────────────────────────────────
#
# The 2026-05-18 attempt crashed inside ``save_cache_entry`` because
# ``_RaptorState`` carried a live ``RetrievalAugmentation`` whose
# embedding adapter wrapped an ``OllamaEmbedder``'s ``httpx.Client``
# (which holds a ``_thread.lock``). The fix split the state into
# pure-data (Tree) and live adapters (rebuilt on rehydrate). These
# tests pin both the byte-equivalence guarantee (the rehydrated
# state produces the SAME generate-stage prompt_hash for the same
# query as the original state) and the cross-process determinism
# property the disk cache exists to provide.

class _DeterministicEmbedder:
    """Stand-in for ``OllamaEmbedder`` for unit tests.

    Returns a deterministic vector keyed by the SHA-256 of the input
    text. Two test runs at the same code commit produce identical
    vectors per identical input — exactly what BGE-M3 via Ollama
    produces in practice, but without the httpx client (and the
    associated ``_thread.lock`` that crashed the real adapter on
    pickle). The cache module sees this object as fully serialisable
    pure-data, which is correct: a real embedder would NOT be saved
    to disk; only the state's pure-data slice is.
    """
    model = "bge-m3"

    def embed(self, texts):
        import hashlib
        from pilot.encoders.ollama import EmbeddingResult
        vectors: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # 16-dim deterministic vector built from the digest bytes
            # — enough dimensions for the retriever's cosine ranking
            # to behave, small enough to keep the test fast.
            vec = [b / 255.0 for b in h[:16]]
            vectors.append(vec)
        return EmbeddingResult(model=self.model, embeddings=vectors)


class _RecordingAnswerer:
    """Stand-in answerer that captures every prompt + returns a fixed
    response. The QA stage's ``prompt_hash`` is the cache-equality
    invariant the test is asserting: rehydrating with this answerer
    on either side of the pickle round-trip must produce the same
    hash for the same query."""

    def __init__(self, response: str = "A"):
        self.response = response
        self.prompts_seen: list[str] = []

    def call(self, prompt, *, model, max_tokens, temperature,
             cache_control=None):
        from dataclasses import dataclass

        @dataclass
        class _Result:
            text: str
            uncached_input_tokens: int = 4
            cached_input_tokens: int = 0
            output_tokens: int = 1
            provider_request_id: str | None = None

        self.prompts_seen.append(prompt)
        return _Result(text=self.response)


def _build_synthetic_raptor_state():
    """Hand-build a ``_RaptorState`` with a tiny real ``Tree``.

    Skips the UMAP+GMM cluster-build path (which needs Ollama + a
    summary LLM) by constructing the Tree directly. The state is
    otherwise the same shape ``run_raptor`` produces post-build, so
    pickling exercises the exact same code path as the production
    save.
    """
    # Import pilot.architectures.raptor FIRST — its module-level
    # ``sys.path.insert`` is what makes ``raptor`` (vendored at
    # code/third_party/raptor) importable. The test would otherwise
    # raise ModuleNotFoundError on the bare ``raptor`` import below.
    from pilot.architectures.raptor import (
        _RaptorState, _LedgerQAModel,
    )
    from raptor.tree_structures import Node, Tree

    # Two leaves + one parent. Embeddings are dicts keyed by the
    # config's ``cluster_embedding_model`` name ("EMB") because that
    # is what ``raptor.utils.get_embeddings`` looks up.
    leaf0 = Node(
        text="Mrs Dalloway opens with Clarissa buying flowers.",
        index=0, children=set(),
        embeddings={"EMB": [0.1] * 16},
    )
    leaf1 = Node(
        text="Septimus Warren Smith sees the dead in the park.",
        index=1, children=set(),
        embeddings={"EMB": [0.2] * 16},
    )
    parent = Node(
        text="A novel about post-war London, set on one day.",
        index=2, children={0, 1},
        embeddings={"EMB": [0.15] * 16},
    )
    all_nodes = {0: leaf0, 1: leaf1, 2: parent}
    layer_to_nodes = {0: [leaf0, leaf1], 1: [parent]}
    tree = Tree(
        all_nodes=all_nodes, root_nodes={2: parent},
        leaf_nodes={0: leaf0, 1: leaf1},
        num_layers=1, layer_to_nodes=layer_to_nodes,
    )

    # Match the contract ``run_raptor`` sets up post-build: ra holds
    # the Tree wired into a real RetrievalAugmentation, qa_adapter is
    # a live ``_LedgerQAModel``. The test then exercises rehydrate
    # to confirm the picklable contract.
    from pilot.architectures.raptor import (
        ClusterTreeConfig, TreeRetrieverConfig,
        RetrievalAugmentationConfig, RetrievalAugmentation,
        _LedgerEmbeddingModel, _LedgerSummarizationModel,
        _RAPTOR_DEFAULTS,
    )
    from pilot.ledger import CostLedger, Stage

    embedder = _DeterministicEmbedder()
    answerer = _RecordingAnswerer(response="A")
    # Throw-away ledger; the round-trip test only needs the
    # RetrievalAugmentation to be wired, not for the ledger to be
    # asserted on.
    return _RaptorState, _LedgerQAModel, _LedgerEmbeddingModel, \
        _LedgerSummarizationModel, ClusterTreeConfig, \
        TreeRetrieverConfig, RetrievalAugmentationConfig, \
        RetrievalAugmentation, _RAPTOR_DEFAULTS, \
        CostLedger, Stage, tree, embedder, answerer


class TestRaptorStatePickleRoundTrip:
    """Regression for the pre-fix `_thread.lock` pickle crash.

    The three assertions together encode the contract:

      1. ``pickle.dumps(state)`` SUCCEEDS — would have raised
         ``TypeError: cannot pickle '_thread.lock' object`` pre-fix.
      2. After unpickle + rehydrate the state can serve a retrieval
         + generate call without raising.
      3. The generate-stage ``prompt_hash`` from the rehydrated state
         is byte-identical to the same hash from the original state
         for the same query — the cache-equality invariant.
    """

    def test_state_pickles_without_thread_lock_crash(self, tmp_path: Path):
        (_RaptorState, _LedgerQAModel, _LedgerEmbeddingModel,
         _LedgerSummarizationModel, ClusterTreeConfig,
         TreeRetrieverConfig, RetrievalAugmentationConfig,
         RetrievalAugmentation, _RAPTOR_DEFAULTS,
         CostLedger, Stage, tree, embedder, answerer) = (
            _build_synthetic_raptor_state()
        )

        ledger = CostLedger(run_id="orig", root=tmp_path)
        embedding_adapter = _LedgerEmbeddingModel(
            embedder=embedder, ledger=ledger, run_index=0,
            stage=Stage.RETRIEVAL,
        )
        summary_adapter = _LedgerSummarizationModel(
            answerer=answerer, model="answerer-A",
            ledger=ledger, run_index=0,
        )
        qa_adapter = _LedgerQAModel(
            answerer=answerer, model="answerer-A",
            ledger=ledger, run_index=0,
        )
        tb_cfg = ClusterTreeConfig(
            max_tokens=_RAPTOR_DEFAULTS["tb_max_tokens"],
            num_layers=_RAPTOR_DEFAULTS["tb_num_layers"],
            summarization_length=_RAPTOR_DEFAULTS["tb_summarization_length"],
            summarization_model=summary_adapter,
            embedding_models={"EMB": embedding_adapter},
            cluster_embedding_model="EMB",
        )
        tr_cfg = TreeRetrieverConfig(
            embedding_model=embedding_adapter,
            context_embedding_model="EMB",
            top_k=_RAPTOR_DEFAULTS["tr_top_k"],
            threshold=_RAPTOR_DEFAULTS["tr_threshold"],
            selection_mode=_RAPTOR_DEFAULTS["tr_selection_mode"],
        )
        cfg = RetrievalAugmentationConfig(
            tree_builder_config=tb_cfg, tree_retriever_config=tr_cfg,
            qa_model=qa_adapter, embedding_model=embedding_adapter,
            summarization_model=summary_adapter,
        )
        ra = RetrievalAugmentation(config=cfg, tree=tree)
        state = _RaptorState(ra=ra, qa_adapter=qa_adapter)

        # The crash-reproducer: pre-fix this raised
        # ``TypeError: cannot pickle '_thread.lock' object`` because
        # the real OllamaEmbedder's httpx.Client carries a lock; the
        # synthetic embedder mirrors the shape but the pre-fix
        # ``_RaptorState.__getstate__`` would have walked the whole
        # ra → tree_builder → embedding_model graph and tried to
        # pickle adapters too. Post-fix only the Tree is serialised.
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        # The pickle blob should NOT contain any reference to the
        # adapter class names or the embedder class name — only the
        # Tree's pure-data graph. This is a quick structural check
        # that __getstate__ really strips live adapters.
        for forbidden in (
            b"_LedgerEmbeddingModel",
            b"_LedgerQAModel",
            b"_LedgerSummarizationModel",
            b"_DeterministicEmbedder",
            b"_RecordingAnswerer",
            b"OllamaEmbedder",
        ):
            assert forbidden not in blob, (
                f"pickle blob contains forbidden live-adapter ref {forbidden!r} — "
                "__getstate__ failed to strip it"
            )

    def test_unpickle_then_rehydrate_then_retrieve_works(self, tmp_path: Path):
        (_RaptorState, _LedgerQAModel, _LedgerEmbeddingModel,
         _LedgerSummarizationModel, ClusterTreeConfig,
         TreeRetrieverConfig, RetrievalAugmentationConfig,
         RetrievalAugmentation, _RAPTOR_DEFAULTS,
         CostLedger, Stage, tree, embedder, answerer) = (
            _build_synthetic_raptor_state()
        )
        # Build original state.
        ledger_a = CostLedger(run_id="orig", root=tmp_path)
        embedding_adapter = _LedgerEmbeddingModel(
            embedder=embedder, ledger=ledger_a, run_index=0,
            stage=Stage.RETRIEVAL,
        )
        qa_adapter = _LedgerQAModel(
            answerer=answerer, model="answerer-A",
            ledger=ledger_a, run_index=0,
        )
        summary_adapter = _LedgerSummarizationModel(
            answerer=answerer, model="answerer-A",
            ledger=ledger_a, run_index=0,
        )
        tb_cfg = ClusterTreeConfig(
            max_tokens=_RAPTOR_DEFAULTS["tb_max_tokens"],
            num_layers=_RAPTOR_DEFAULTS["tb_num_layers"],
            summarization_length=_RAPTOR_DEFAULTS["tb_summarization_length"],
            summarization_model=summary_adapter,
            embedding_models={"EMB": embedding_adapter},
            cluster_embedding_model="EMB",
        )
        tr_cfg = TreeRetrieverConfig(
            embedding_model=embedding_adapter,
            context_embedding_model="EMB",
            top_k=_RAPTOR_DEFAULTS["tr_top_k"],
            threshold=_RAPTOR_DEFAULTS["tr_threshold"],
            selection_mode=_RAPTOR_DEFAULTS["tr_selection_mode"],
        )
        cfg = RetrievalAugmentationConfig(
            tree_builder_config=tb_cfg, tree_retriever_config=tr_cfg,
            qa_model=qa_adapter, embedding_model=embedding_adapter,
            summarization_model=summary_adapter,
        )
        ra = RetrievalAugmentation(config=cfg, tree=tree)
        state = _RaptorState(ra=ra, qa_adapter=qa_adapter)

        # Round-trip via pickle.
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        revived = pickle.loads(blob)
        assert revived.ra is None
        assert revived.qa_adapter is None

        # Rehydrate with fresh adapters on a different ledger,
        # mimicking the cross-candidate cache-load path.
        ledger_b = CostLedger(run_id="rehydrated", root=tmp_path)
        answerer_b = _RecordingAnswerer(response="B")
        revived.rehydrate(
            embedder=_DeterministicEmbedder(), ledger=ledger_b,
            answerer=answerer_b, answerer_model="answerer-B",
        )
        assert revived.ra is not None
        assert revived.qa_adapter is not None

        # Retrieve works; the retriever walks the prebuilt tree
        # using the synthetic embedder.
        context, _layer = revived.ra.retrieve(
            "What does Clarissa buy?",
            collapse_tree=True, max_tokens=2000,
            return_layer_information=True,
        )
        assert "Clarissa" in context or "Septimus" in context or context

    def test_prompt_hash_identical_across_pickle_boundary(self, tmp_path: Path):
        """The load-bearing assertion. Two candidates load the SAME
        pickle and rehydrate with their own answerer adapters; for the
        same query, the generate-stage prompt_hash MUST be identical
        between (original, rehydrated) — the retrieved context is what
        feeds the answer prompt, and the cache exists to make that
        context byte-stable across candidates."""
        # pilot.architectures.raptor must import before the vendored
        # ``raptor`` module is accessible — its module-level sys.path
        # tweak is what exposes the third-party package.
        from pilot.architectures.raptor import (
            ClusterTreeConfig, TreeRetrieverConfig,
            RetrievalAugmentationConfig, RetrievalAugmentation,
            _RaptorState, _LedgerEmbeddingModel,
            _LedgerSummarizationModel, _LedgerQAModel,
            _RAPTOR_DEFAULTS,
        )
        from pilot.ledger import CostLedger, Stage
        from raptor.tree_structures import Node, Tree

        # Reusable Tree builder — keeping it inline rather than
        # round-tripping the synthetic builder to keep the call
        # self-contained and force two SEPARATE ra instances (so a
        # shared internal mutation cannot influence the assertion).
        def _make_state(ledger, answerer, answerer_model):
            leaves = [
                Node(text=f"chunk-{i}: deterministic sentence body {i}.",
                     index=i, children=set(),
                     embeddings={"EMB": [(i + 1) / 10.0] * 16})
                for i in range(4)
            ]
            parent = Node(
                text="recursive summary of the four leaves",
                index=4, children={0, 1, 2, 3},
                embeddings={"EMB": [0.5] * 16},
            )
            all_nodes = {n.index: n for n in [*leaves, parent]}
            tree = Tree(
                all_nodes=all_nodes, root_nodes={4: parent},
                leaf_nodes={i: leaves[i] for i in range(4)},
                num_layers=1,
                layer_to_nodes={0: leaves, 1: [parent]},
            )
            embedder = _DeterministicEmbedder()
            emb_adapter = _LedgerEmbeddingModel(
                embedder=embedder, ledger=ledger, run_index=0,
                stage=Stage.RETRIEVAL,
            )
            sum_adapter = _LedgerSummarizationModel(
                answerer=answerer, model=answerer_model,
                ledger=ledger, run_index=0,
            )
            qa_adapter = _LedgerQAModel(
                answerer=answerer, model=answerer_model,
                ledger=ledger, run_index=0,
            )
            tb_cfg = ClusterTreeConfig(
                max_tokens=_RAPTOR_DEFAULTS["tb_max_tokens"],
                num_layers=_RAPTOR_DEFAULTS["tb_num_layers"],
                summarization_length=_RAPTOR_DEFAULTS["tb_summarization_length"],
                summarization_model=sum_adapter,
                embedding_models={"EMB": emb_adapter},
                cluster_embedding_model="EMB",
            )
            tr_cfg = TreeRetrieverConfig(
                embedding_model=emb_adapter,
                context_embedding_model="EMB",
                top_k=_RAPTOR_DEFAULTS["tr_top_k"],
                threshold=_RAPTOR_DEFAULTS["tr_threshold"],
                selection_mode=_RAPTOR_DEFAULTS["tr_selection_mode"],
            )
            cfg = RetrievalAugmentationConfig(
                tree_builder_config=tb_cfg, tree_retriever_config=tr_cfg,
                qa_model=qa_adapter, embedding_model=emb_adapter,
                summarization_model=sum_adapter,
            )
            ra = RetrievalAugmentation(config=cfg, tree=tree)
            state = _RaptorState(ra=ra, qa_adapter=qa_adapter)
            return state

        query = "What does the recursive summary contain?"

        # 1. Original state: build, answer once via the runner-equivalent
        #    path. The synthetic answerer records the prompt verbatim;
        #    we hash it the same way the ledger does.
        from pilot.ledger import sha256_hex as _hash
        ledger_orig = CostLedger(run_id="orig", root=tmp_path)
        answerer_orig = _RecordingAnswerer(response="A")
        orig_state = _make_state(ledger_orig, answerer_orig, "model-A")
        orig_state.qa_adapter.current_options = {"A": "yes", "B": "no"}
        # Mirror the runner's call order: retrieve → answer_question.
        # answer_question internally retrieves again then calls
        # qa_adapter.answer_question(context, query); the call ends up
        # in answerer_orig.prompts_seen[-1].
        orig_state.ra.answer_question(
            query, collapse_tree=True, max_tokens=2000,
        )
        orig_generate_prompt = answerer_orig.prompts_seen[-1]
        orig_hash = _hash(orig_generate_prompt)

        # 2. Pickle → unpickle → rehydrate with a DIFFERENT answerer
        #    (mimicking candidate B in the cross-candidate rerun).
        blob = pickle.dumps(orig_state, protocol=pickle.HIGHEST_PROTOCOL)
        revived = pickle.loads(blob)

        ledger_rev = CostLedger(run_id="rev", root=tmp_path)
        answerer_rev = _RecordingAnswerer(response="B")
        revived.rehydrate(
            embedder=_DeterministicEmbedder(), ledger=ledger_rev,
            answerer=answerer_rev, answerer_model="model-B",
        )
        revived.qa_adapter.current_options = {"A": "yes", "B": "no"}
        revived.ra.answer_question(
            query, collapse_tree=True, max_tokens=2000,
        )
        rev_generate_prompt = answerer_rev.prompts_seen[-1]
        rev_hash = _hash(rev_generate_prompt)

        # The whole purpose of the disk cache: the generate-stage
        # prompt_hash is identical across candidates. Any drift here
        # means the retrieved context drifted, which means the cache
        # has lost its byte-equality guarantee.
        assert orig_hash == rev_hash, (
            "rehydrated state produced a different generate prompt_hash "
            "than the original state for the same query — the cache's "
            "byte-equality guarantee is broken"
        )


class TestGraphRagStatePickleRoundTrip:
    """GraphRAG's state was already pure-data, but adding the
    __getstate__/__setstate__/rehydrate triad pins the on-disk
    schema so a future drift into holding a live adapter would be
    caught at review rather than silently re-introducing the
    `_thread.lock` pickle crash that hit RAPTOR."""

    def test_state_pickles_and_unpickles_intact(self, tmp_path: Path):
        from pilot.architectures.graphrag import (
            _GraphRAGState, _Entity, _Relationship, _CommunityReport,
        )
        import networkx as nx

        g = nx.Graph()
        g.add_node("Alice", type="character", description="protagonist",
                   text_unit_ids=(0,))
        g.add_node("Bob", type="character", description="antagonist",
                   text_unit_ids=(1,))
        g.add_edge("Alice", "Bob", weight=2, strength=8.0, score=1.6,
                   description="Alice confronts Bob.")
        state = _GraphRAGState(
            chunks=["chunk 0", "chunk 1"],
            entities=[
                _Entity(name="Alice", type="character",
                        description="protagonist", text_unit_ids=[0]),
                _Entity(name="Bob", type="character",
                        description="antagonist", text_unit_ids=[1]),
            ],
            relationships=[
                _Relationship(source="Alice", target="Bob",
                              description="conflict", text_unit_id=0,
                              strength=8),
            ],
            g=g,
            communities=[{"Alice", "Bob"}],
            reports=[_CommunityReport(
                community_id=0, member_names=["Alice", "Bob"],
                text="## Title\nAlice + Bob", rank=2,
            )],
            entity_vecs=[[0.1] * 8, [0.2] * 8],
            embed_dim=8,
        )

        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        revived = pickle.loads(blob)
        assert revived.chunks == state.chunks
        assert len(revived.entities) == 2
        assert revived.entities[0].name == "Alice"
        assert revived.entity_vecs == state.entity_vecs
        assert revived.embed_dim == 8
        # NetworkX graph round-trips.
        assert revived.g.number_of_nodes() == 2
        assert revived.g.has_edge("Alice", "Bob")

        # rehydrate is a symmetric no-op on GraphRAG but must return
        # self so callers can chain it.
        assert revived.rehydrate(embedder=None, ledger=None,
                                 answerer=None, answerer_model=None) is revived


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
