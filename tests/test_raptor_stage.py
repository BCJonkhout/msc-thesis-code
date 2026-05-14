"""Tests for RAPTOR's embedding-adapter stage attribution.

The same ``_LedgerEmbeddingModel`` instance is wired into both
RAPTOR's tree builder (which embeds every leaf chunk and every
internal-cluster node) and the query-time TreeRetriever (which
embeds the user query against the prebuilt tree). The two phases
land in different cost-model buckets:

  - Build embeds → ``C_off^struct`` → ``Stage.PREPROCESS``
  - Query embeds → ``C_on`` → ``Stage.RETRIEVAL``

``run_raptor`` controls this by instantiating the adapter with
``stage=PREPROCESS`` and flipping it to ``RETRIEVAL`` immediately
after ``ra.add_documents()`` returns. These tests verify the
adapter respects whatever stage is current at the moment of the
embed call, so the contract ``run_raptor`` relies on is upheld.

This matters because the *total* amortised cost is invariant to
the stage choice (both buckets are summed into the same per-paper
total), but the OFFLINE / ONLINE SPLIT is what the break-even
analysis in project.tex §3.4.1 depends on. A mis-tag silently
inverts the break-even curve without changing the headline cost.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pilot.architectures.raptor import _LedgerEmbeddingModel
from pilot.ledger import CostLedger, Stage


class _StubEmbedder:
    model = "stub-embed"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        # Return a tiny non-empty vector per text so the adapter can
        # compute its response_hash on a non-empty fingerprint.
        return MagicMock(embeddings=[[0.1, 0.2, 0.3] for _ in texts])


class TestStageAttribute:
    def test_default_stage_is_preprocess(self, tmp_path: Path):
        """Newly-constructed adapter defaults to PREPROCESS so the
        tree-builder phase records build embeds correctly even if
        the caller forgets to set it explicitly."""
        ledger = CostLedger(run_id="r-default", root=tmp_path)
        adapter = _LedgerEmbeddingModel(
            embedder=_StubEmbedder(), ledger=ledger, run_index=0,
        )
        assert adapter.stage == Stage.PREPROCESS

    def test_explicit_preprocess_logs_as_preprocess(self, tmp_path: Path):
        ledger = CostLedger(run_id="r-preprocess", root=tmp_path)
        adapter = _LedgerEmbeddingModel(
            embedder=_StubEmbedder(), ledger=ledger, run_index=0,
            stage=Stage.PREPROCESS,
        )
        adapter.create_embedding("chunk text")
        rows = ledger.read()
        assert len(rows) == 1
        assert rows[0].stage == "preprocess"

    def test_flipping_to_retrieval_changes_subsequent_rows(
        self, tmp_path: Path
    ):
        """Build phase → embeds tagged PREPROCESS. Then the caller
        flips ``adapter.stage = RETRIEVAL``. Subsequent embeds
        (the per-query work) must land as RETRIEVAL — this is the
        exact transition ``run_raptor`` performs between
        ``ra.add_documents()`` and ``ra.retrieve()``.
        """
        ledger = CostLedger(run_id="r-flip", root=tmp_path)
        adapter = _LedgerEmbeddingModel(
            embedder=_StubEmbedder(), ledger=ledger, run_index=0,
            stage=Stage.PREPROCESS,
        )
        # Two build embeds.
        adapter.create_embedding("leaf chunk one")
        adapter.create_embedding("leaf chunk two")
        # Caller flips to retrieval mode.
        adapter.stage = Stage.RETRIEVAL
        # One query embed.
        adapter.create_embedding("the user query")

        rows = ledger.read()
        stages = [r.stage for r in rows]
        assert stages == ["preprocess", "preprocess", "retrieval"]

    def test_all_rows_tagged_raptor_architecture(self, tmp_path: Path):
        ledger = CostLedger(run_id="r-arch", root=tmp_path)
        adapter = _LedgerEmbeddingModel(
            embedder=_StubEmbedder(), ledger=ledger, run_index=0,
        )
        adapter.create_embedding("x")
        adapter.stage = Stage.RETRIEVAL
        adapter.create_embedding("y")
        rows = ledger.read()
        assert {r.architecture for r in rows} == {"raptor"}
