"""Tests for the Step 3 dry-run architecture dispatcher.

`pilot.cli.step_3_dry_run._invoke_architecture` is a pure dispatcher
that maps architecture name → runner. The test surface:

  - "flat" / "naive_rag" / "raptor" / "graphrag" each route to the
    correct runner with the right arguments
  - the embedder + chunker requirement is enforced (flat doesn't
    need them; the others do)
  - prompt_style is threaded through to the runners that accept it
  - unknown architecture names raise ValueError

The dispatcher's runners are monkey-patched so this test never
touches Ollama / Gemini / RAPTOR's vendored code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.architectures import ArchitectureResult
from pilot.cli import step_3_dry_run as cli


@pytest.fixture
def fake_item() -> dict[str, Any]:
    return {
        "dataset": "qasper",
        "paper_id": "paper42",
        "question_id": "q1",
        "question": "what?",
        "options": None,
        "document": "Document text.",
        "gold_answers": ["x"],
        "gold_evidence_sentences": [],
        "gold_label": None,
    }


@pytest.fixture
def stub_dependencies(monkeypatch):
    """Replace each runner with a recorder that captures kwargs and
    returns a sentinel ArchitectureResult."""
    calls: dict[str, list[dict[str, Any]]] = {
        "flat": [], "naive_rag": [], "raptor": [], "graphrag": [],
    }

    def make_recorder(name: str):
        def recorder(**kwargs):
            calls[name].append(kwargs)
            return ArchitectureResult(
                architecture=name,
                predicted_answer=f"answer-from-{name}",
            )
        return recorder

    monkeypatch.setattr(cli, "run_flat", make_recorder("flat"))
    monkeypatch.setattr(cli, "run_naive_rag", make_recorder("naive_rag"))
    monkeypatch.setattr(cli, "run_raptor", make_recorder("raptor"))
    monkeypatch.setattr(cli, "run_graphrag", make_recorder("graphrag"))
    return calls


# ──────────────────────────────────────────────────────────────────────
# Architecture routing
# ──────────────────────────────────────────────────────────────────────

class TestDispatcherRouting:
    def test_flat_routes_to_run_flat(self, fake_item, stub_dependencies):
        result = cli._invoke_architecture(
            "flat", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=None, chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
        )
        assert result.architecture == "flat"
        assert len(stub_dependencies["flat"]) == 1
        assert stub_dependencies["naive_rag"] == []

    def test_naive_rag_routes_to_run_naive_rag(self, fake_item, stub_dependencies):
        result = cli._invoke_architecture(
            "naive_rag", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=MagicMock(),
            ledger=MagicMock(), naive_rag_top_k=8,
        )
        assert result.architecture == "naive_rag"
        assert len(stub_dependencies["naive_rag"]) == 1

    def test_raptor_routes_to_run_raptor(self, fake_item, stub_dependencies):
        result = cli._invoke_architecture(
            "raptor", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
        )
        assert result.architecture == "raptor"
        assert len(stub_dependencies["raptor"]) == 1

    def test_graphrag_routes_to_run_graphrag(self, fake_item, stub_dependencies):
        result = cli._invoke_architecture(
            "graphrag", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
        )
        assert result.architecture == "graphrag"
        assert len(stub_dependencies["graphrag"]) == 1


# ──────────────────────────────────────────────────────────────────────
# Embedder requirement enforcement
# ──────────────────────────────────────────────────────────────────────

class TestEmbedderRequirement:
    def test_naive_rag_without_embedder_raises(self, fake_item, stub_dependencies):
        with pytest.raises(RuntimeError, match="naive_rag requires embedder"):
            cli._invoke_architecture(
                "naive_rag", fake_item,
                answerer=MagicMock(), answerer_model="m",
                embedder=None, chunker=MagicMock(),
                ledger=MagicMock(), naive_rag_top_k=8,
            )

    def test_naive_rag_without_chunker_raises(self, fake_item, stub_dependencies):
        with pytest.raises(RuntimeError, match="naive_rag requires embedder"):
            cli._invoke_architecture(
                "naive_rag", fake_item,
                answerer=MagicMock(), answerer_model="m",
                embedder=MagicMock(), chunker=None,
                ledger=MagicMock(), naive_rag_top_k=8,
            )

    def test_raptor_without_embedder_raises(self, fake_item, stub_dependencies):
        with pytest.raises(RuntimeError, match="raptor requires embedder"):
            cli._invoke_architecture(
                "raptor", fake_item,
                answerer=MagicMock(), answerer_model="m",
                embedder=None, chunker=None,
                ledger=MagicMock(), naive_rag_top_k=8,
            )

    def test_graphrag_without_embedder_raises(self, fake_item, stub_dependencies):
        with pytest.raises(RuntimeError, match="graphrag requires embedder"):
            cli._invoke_architecture(
                "graphrag", fake_item,
                answerer=MagicMock(), answerer_model="m",
                embedder=None, chunker=None,
                ledger=MagicMock(), naive_rag_top_k=8,
            )


# ──────────────────────────────────────────────────────────────────────
# prompt_style threading
# ──────────────────────────────────────────────────────────────────────

class TestPromptStyleThreading:
    def test_flat_receives_prompt_style(self, fake_item, stub_dependencies):
        cli._invoke_architecture(
            "flat", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=None, chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
            prompt_style="literature",
        )
        assert stub_dependencies["flat"][0]["prompt_style"] == "literature"

    def test_naive_rag_receives_prompt_style(self, fake_item, stub_dependencies):
        cli._invoke_architecture(
            "naive_rag", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=MagicMock(),
            ledger=MagicMock(), naive_rag_top_k=8,
            prompt_style="literature",
        )
        assert stub_dependencies["naive_rag"][0]["prompt_style"] == "literature"

    def test_default_prompt_style_is_pilot(self, fake_item, stub_dependencies):
        cli._invoke_architecture(
            "flat", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=None, chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
        )
        assert stub_dependencies["flat"][0]["prompt_style"] == "pilot"

    def test_raptor_does_not_receive_prompt_style(
        self, fake_item, stub_dependencies
    ):
        """RAPTOR uses its own paper-verbatim prompts; the dispatcher
        should not pass prompt_style to it (the call signature
        wouldn't accept it)."""
        cli._invoke_architecture(
            "raptor", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
            prompt_style="literature",
        )
        # The recorder captured all kwargs; prompt_style must not be in them.
        assert "prompt_style" not in stub_dependencies["raptor"][0]


# ──────────────────────────────────────────────────────────────────────
# Unknown architecture
# ──────────────────────────────────────────────────────────────────────

class TestUnknownArchitecture:
    def test_unknown_name_raises_valueerror(self, fake_item, stub_dependencies):
        with pytest.raises(ValueError, match="unsupported architecture"):
            cli._invoke_architecture(
                "nonexistent", fake_item,
                answerer=MagicMock(), answerer_model="m",
                embedder=MagicMock(), chunker=None,
                ledger=MagicMock(), naive_rag_top_k=8,
            )


# ──────────────────────────────────────────────────────────────────────
# Multi-provider routing for the summary stage (Phase F extension)
# ──────────────────────────────────────────────────────────────────────

class TestSummaryProviderRouting:
    """RAPTOR + GraphRAG accept ``summary_answerer`` + ``summary_model``
    so the summary stage can run on a different provider than the
    answerer (e.g. Grok answerer + Gemini Flash Lite summary). The
    dispatcher must thread both through to the runners; Flat and
    Naive RAG have no summary stage and should ignore both kwargs.
    """

    def test_raptor_receives_summary_provider_and_model(
        self, fake_item, stub_dependencies
    ):
        answerer = MagicMock(name="grok_answerer")
        summary = MagicMock(name="gemini_summary")
        cli._invoke_architecture(
            "raptor", fake_item,
            answerer=answerer, answerer_model="grok-4.20-0309-non-reasoning",
            embedder=MagicMock(), chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
            summary_answerer=summary,
            summary_model="gemini-3.1-flash-lite-preview",
        )
        kwargs = stub_dependencies["raptor"][0]
        assert kwargs["summary_answerer"] is summary
        assert kwargs["summary_model"] == "gemini-3.1-flash-lite-preview"
        assert kwargs["answerer"] is answerer
        assert kwargs["answerer_model"] == "grok-4.20-0309-non-reasoning"

    def test_graphrag_receives_summary_provider_and_model(
        self, fake_item, stub_dependencies
    ):
        answerer = MagicMock(name="grok_answerer")
        summary = MagicMock(name="gemini_summary")
        cli._invoke_architecture(
            "graphrag", fake_item,
            answerer=answerer, answerer_model="grok-4.20-0309-non-reasoning",
            embedder=MagicMock(), chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
            summary_answerer=summary,
            summary_model="gemini-3.1-flash-lite-preview",
        )
        kwargs = stub_dependencies["graphrag"][0]
        assert kwargs["summary_answerer"] is summary
        assert kwargs["summary_model"] == "gemini-3.1-flash-lite-preview"

    def test_summary_kwargs_default_to_none_when_not_passed(
        self, fake_item, stub_dependencies
    ):
        cli._invoke_architecture(
            "raptor", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
        )
        kwargs = stub_dependencies["raptor"][0]
        assert kwargs.get("summary_answerer") is None
        assert kwargs.get("summary_model") is None

    def test_flat_does_not_receive_summary_kwargs(
        self, fake_item, stub_dependencies
    ):
        cli._invoke_architecture(
            "flat", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=None, chunker=None,
            ledger=MagicMock(), naive_rag_top_k=8,
            summary_answerer=MagicMock(),
            summary_model="other",
        )
        # Flat has no summary stage; the runner signature does not
        # accept summary_* and would TypeError if forwarded.
        kwargs = stub_dependencies["flat"][0]
        assert "summary_answerer" not in kwargs
        assert "summary_model" not in kwargs

    def test_naive_rag_does_not_receive_summary_kwargs(
        self, fake_item, stub_dependencies
    ):
        cli._invoke_architecture(
            "naive_rag", fake_item,
            answerer=MagicMock(), answerer_model="m",
            embedder=MagicMock(), chunker=MagicMock(),
            ledger=MagicMock(), naive_rag_top_k=8,
            summary_answerer=MagicMock(),
            summary_model="other",
        )
        kwargs = stub_dependencies["naive_rag"][0]
        assert "summary_answerer" not in kwargs
        assert "summary_model" not in kwargs
