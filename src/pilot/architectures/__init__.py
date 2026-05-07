"""Pilot answerer architectures.

Four architectures are evaluated under unified cost accounting:

  - flat       — Flat full-context (document + question to answerer).
  - naive_rag  — Chunk + embed + top-k cosine retrieve + answer.
  - raptor     — RAPTOR (Sarthi et al. 2024): recursive cluster +
                 summarize → tree → collapsed-tree retrieval.
  - graphrag   — GraphRAG (Edge et al. 2024): entity extraction +
                 community detection (Leiden) + community summary.

Each architecture is a callable that takes (document_or_corpus,
question, answerer, ledger) and returns an `ArchitectureResult` with
the predicted answer + the retrieved evidence sentences (used for
QASPER Evidence-F1 diagnostic). All LLM and embedding calls flow
through the provided answerer (a `pilot.providers.AnswererProvider`)
and embedder (a `pilot.encoders.OllamaEmbedder`); both record into
the supplied `pilot.ledger.CostLedger` so cost accounting is unified
across architectures.
"""
from pilot.architectures.base import ArchitectureResult, run_flat
from pilot.architectures.naive_rag import run_naive_rag

__all__ = [
    "ArchitectureResult",
    "run_flat",
    "run_naive_rag",
]
