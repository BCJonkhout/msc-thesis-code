"""RAPTOR runner.

Wires the vendored official RAPTOR reference implementation
(``code/third_party/raptor``, MIT-licensed mirror of
parthsarthi03/raptor) into the pilot's cost-accounting and provider
abstractions. Three thin shim classes route every embedding,
summarisation, and QA call through ``CostLedger`` so RAPTOR's
per-call cost is observable on the same footing as the other
architectures.

**Native-thread requirement.** UMAP+GMM clustering inside the
vendored repo segfaults on Windows when numba and OpenMP threading
are at their default settings (the JIT triggers a SIGSEGV during
"Constructing Layer 0"). Setting ``OMP_NUM_THREADS=1`` and
``NUMBA_NUM_THREADS=1`` on import resolves it. We do this inside
the module so callers don't have to remember.

Per the RAPTOR research brief (`thesis-msc/notes/pilot_findings.md`,
Step 3 RAPTOR section), the configuration matches Sarthi et al.
2024 *over* the official-repo defaults where they diverge:

  - retrieval token budget: 2000 (paper §4) vs 3500 (repo default)
  - retrieval mode: collapsed-tree (paper headline)
  - leaf chunk size: 100 tokens (per ``raptor/utils.py::split_text``)
  - clustering: UMAP+GMM with BIC sweep, repo default params
  - summariser temperature: T=0 (pinned by the pilot, paper silent)
  - answerer temperature: T=0 (pilot lock; matches repo's GPT4QAModel)

Cost-accounting note: every leaf-chunk embed, every tree-node
embed, every summarisation call, and the final answer call lands
in the ledger as separate rows tagged
``stage=retrieval`` (embeddings), ``stage=preprocess``
(summarisation), and ``stage=generate`` (answer). The pilot's
deployment-cost rule (cost-attribution Option A, run_index=0 only)
applies uniformly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# UMAP+GMM clustering inside the vendored RAPTOR repo segfaults on
# Windows under default numba+OpenMP threading. Pinning to single
# thread resolves it without touching the upstream source. The pins
# are applied at import time so callers don't have to remember; if
# the user has already exported these variables, those values win.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pilot.architectures.base import ArchitectureResult
from pilot.encoders import OllamaEmbedder
from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.providers.base import AnswererProvider, CacheControl

# Vendored RAPTOR upstream lives at code/third_party/raptor and is
# imported by adding code/third_party to sys.path on first use. We
# do this on import so subsequent `import raptor` calls succeed.
_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from raptor import (  # noqa: E402  (import order: must follow sys.path tweak)
    BaseEmbeddingModel,
    BaseQAModel,
    BaseSummarizationModel,
    ClusterTreeConfig,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    TreeRetrieverConfig,
)


# ──────────────────────────────────────────────────────────────────────
# Embedder adapter
# ──────────────────────────────────────────────────────────────────────

class _LedgerEmbeddingModel(BaseEmbeddingModel):
    """Adapter routing every embed call through ``CostLedger``."""

    def __init__(
        self,
        *,
        embedder: OllamaEmbedder,
        ledger: CostLedger,
        run_index: int = 0,
    ) -> None:
        self.embedder = embedder
        self.ledger = ledger
        self.run_index = run_index

    def create_embedding(self, text):
        # Ollama's /api/embed accepts a list; we send a 1-element batch.
        # Token count is approximated by char/4 since Ollama doesn't
        # report token usage on embed (the same heuristic used by
        # naive_rag for fair cross-architecture cost comparison).
        with self.ledger.log_call(
            architecture="raptor",
            stage=Stage.RETRIEVAL,
            model=self.embedder.model,
            prompt=text,
            run_index=self.run_index,
        ) as rec:
            result = self.embedder.embed([text])
            vec = result.embeddings[0]
            rec.uncached_input_tokens = max(1, len(text) // 4)
            rec.cached_input_tokens = 0
            rec.output_tokens = 0
            rec.response_hash = sha256_hex(
                "|".join(f"{v:.6f}" for v in vec[:8])  # short fingerprint
            )
        return vec


# ──────────────────────────────────────────────────────────────────────
# Summarisation adapter
# ──────────────────────────────────────────────────────────────────────

# Verbatim from raptor/SummarizationModels.py prompt body. Preserved
# exactly so cost-vs-quality results are comparable to the reference
# implementation.
_RAPTOR_SUMMARY_PROMPT = (
    "Write a summary of the following, including as many key details "
    "as possible: {context}:"
)


class _LedgerSummarizationModel(BaseSummarizationModel):
    """Adapter routing summarisation calls through the answerer + ledger.

    The pilot fixes T=0 for all preprocessing calls per § 3.4.3.
    Summary length matches the paper-default ~200 tokens; the
    repo's API ``max_tokens=500`` cap is preserved as the
    answerer-side ceiling.
    """

    def __init__(
        self,
        *,
        answerer: AnswererProvider,
        model: str,
        ledger: CostLedger,
        run_index: int = 0,
        max_tokens: int = 500,
    ) -> None:
        self.answerer = answerer
        self.model = model
        self.ledger = ledger
        self.run_index = run_index
        self.max_tokens = max_tokens

    def summarize(self, context, max_tokens: int | None = None):
        prompt = _RAPTOR_SUMMARY_PROMPT.format(context=context)
        cap = max_tokens if max_tokens is not None else self.max_tokens
        with self.ledger.log_call(
            architecture="raptor",
            stage=Stage.PREPROCESS,
            model=self.model,
            prompt=prompt,
            run_index=self.run_index,
            temperature=0.0,
            max_tokens=cap,
        ) as rec:
            result = self.answerer.call(
                prompt,
                model=self.model,
                max_tokens=cap,
                temperature=0.0,
                cache_control=CacheControl.EPHEMERAL_5MIN,
            )
            rec.uncached_input_tokens = result.uncached_input_tokens
            rec.cached_input_tokens = result.cached_input_tokens
            rec.output_tokens = result.output_tokens
            rec.provider_request_id = result.provider_request_id
            rec.response_hash = sha256_hex(result.text or "")
        return result.text


# ──────────────────────────────────────────────────────────────────────
# QA adapter
# ──────────────────────────────────────────────────────────────────────

# Verbatim concatenation of the chat-API prompt used by the vendored
# `raptor/QAModels.GPT4QAModel.answer_question` — the class that
# Sarthi et al. 2024 used for the GPT-4 QASPER replication. The
# vendored class issues a chat-completion with system + user
# messages; our `AnswererProvider.call` accepts a single prompt
# string, so we inline the system message at the top.
#
# An earlier pilot draft of this file accidentally adopted the
# legacy `GPT3QAModel` prompt from the same vendored repo (the
# completion-API path with the "folloing" typo and the "answer in
# less than 5-7 words" cap). That prompt is for `text-davinci-003`,
# not GPT-4. The 5-7-word cap was structurally truncating QASPER
# answers and is the load-bearing reason RAPTOR's apparent F1 sat
# below Naive RAG on the calibration sweeps. Corrected to the
# modern chat-API prompt for paper-faithful comparison.
_RAPTOR_FREEFORM_PROMPT = (
    "You are Question Answering Portal\n\n"
    "Given Context: {context} Give the best full answer amongst the option "
    "to question {question}"
)
# The MC prompt has no equivalent in the vendored repo (RAPTOR's
# paper benchmarks are free-form QA). We mirror the system framing
# and add the option-letter constraint that NovelQA / QuALITY
# require for codabench-comparable scoring.
_RAPTOR_MC_PROMPT = (
    "You are Question Answering Portal\n\n"
    "Given Context: {context}\n\n"
    "Answer the following multiple-choice question by responding with the "
    "single option letter (A, B, C, or D) of the correct answer:\n\n"
    "{question}\n\n{options}"
)


def _format_options_block(options: dict[str, str] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


class _LedgerQAModel(BaseQAModel):
    """Adapter routing the answer call through the answerer + ledger.

    A single QA model instance handles both free-form (QASPER) and
    multiple-choice (NovelQA / QuALITY) questions. The active
    options dict is set on the instance immediately before each
    ``answer_question`` call by the architecture wrapper below.
    """

    def __init__(
        self,
        *,
        answerer: AnswererProvider,
        model: str,
        ledger: CostLedger,
        run_index: int = 0,
        max_tokens: int = 256,
    ) -> None:
        self.answerer = answerer
        self.model = model
        self.ledger = ledger
        self.run_index = run_index
        self.max_tokens = max_tokens
        self.current_options: dict[str, str] | None = None

    def answer_question(self, context, question):
        if self.current_options:
            prompt = _RAPTOR_MC_PROMPT.format(
                context=context,
                question=question,
                options=_format_options_block(self.current_options),
            )
        else:
            prompt = _RAPTOR_FREEFORM_PROMPT.format(
                context=context, question=question
            )
        with self.ledger.log_call(
            architecture="raptor",
            stage=Stage.GENERATE,
            model=self.model,
            prompt=prompt,
            run_index=self.run_index,
            temperature=0.0,
            max_tokens=self.max_tokens,
        ) as rec:
            result = self.answerer.call(
                prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                cache_control=CacheControl.EPHEMERAL_5MIN,
            )
            rec.uncached_input_tokens = result.uncached_input_tokens
            rec.cached_input_tokens = result.cached_input_tokens
            rec.output_tokens = result.output_tokens
            rec.provider_request_id = result.provider_request_id
            rec.response_hash = sha256_hex(result.text or "")
        return result.text


# ──────────────────────────────────────────────────────────────────────
# Top-level architecture entrypoint
# ──────────────────────────────────────────────────────────────────────

# Default RAPTOR config matching the paper-headline collapsed-tree
# retrieval at 2000-token budget. Per the research brief, these
# override several repo defaults (max_tokens=3500 → 2000, the
# embedding model handle is "EMB" so RAPTOR uses our injected
# embedder for both clustering and retrieval).
_RAPTOR_DEFAULTS = dict(
    tb_max_tokens=100,           # leaf chunk size; paper §3.1
    tb_num_layers=5,             # depth cap; paper §3.1
    tb_summarization_length=200, # summary target ≈ 200 tokens (pilot's lock)
    tr_top_k=10,                 # retrieval k (collapsed-tree picks by token budget)
    tr_threshold=0.5,
    tr_selection_mode="threshold",
    collapse_tree=True,
    retrieval_max_tokens=2000,   # paper §4 "approximately top-20 nodes"
)


def run_raptor(
    *,
    document: str,
    query: str,
    options: dict[str, str] | None,
    answerer: AnswererProvider,
    answerer_model: str,
    summary_model: str | None,
    embedder: OllamaEmbedder,
    ledger: CostLedger,
    run_index: int = 0,
    max_tokens: int = 256,
    summary_answerer: AnswererProvider | None = None,
) -> ArchitectureResult:
    """Build a RAPTOR tree over ``document`` and answer ``query``.

    The summary model defaults to the answerer model unless
    overridden. ``summary_answerer`` lets the summary stage run
    on a different provider than the final answerer (Phase F
    extension protocol: e.g. Grok answerer + Gemini Flash Lite
    summary). When ``summary_answerer`` is ``None`` the summary
    calls go through the same provider as the answerer.
    """
    summary_model = summary_model or answerer_model
    summary_answerer = summary_answerer or answerer

    embedding_adapter = _LedgerEmbeddingModel(
        embedder=embedder, ledger=ledger, run_index=run_index
    )
    summary_adapter = _LedgerSummarizationModel(
        answerer=summary_answerer, model=summary_model,
        ledger=ledger, run_index=run_index,
    )
    qa_adapter = _LedgerQAModel(
        answerer=answerer, model=answerer_model,
        ledger=ledger, run_index=run_index, max_tokens=max_tokens,
    )
    qa_adapter.current_options = options

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
        tree_builder_config=tb_cfg,
        tree_retriever_config=tr_cfg,
        qa_model=qa_adapter,
        embedding_model=embedding_adapter,
        summarization_model=summary_adapter,
    )
    ra = RetrievalAugmentation(config=cfg)

    try:
        ra.add_documents(document)
    except Exception as exc:
        return ArchitectureResult(
            architecture="raptor",
            predicted_answer="",
            failed=True,
            failure_reason=f"tree_build_failed: {exc!r}",
        )

    # Retrieve to capture the evidence sentences (collapsed-tree mode,
    # paper-default 2000-token budget) BEFORE answering, so we can
    # report what the architecture saw.
    retrieved_text, layer_info = ra.retrieve(
        query,
        collapse_tree=_RAPTOR_DEFAULTS["collapse_tree"],
        max_tokens=_RAPTOR_DEFAULTS["retrieval_max_tokens"],
        return_layer_information=True,
    )

    # Now answer. RetrievalAugmentation.answer_question retrieves
    # again then calls qa_model; we've already retrieved above for
    # the evidence record, but the second retrieve is fast (tree is
    # cached) and matches the official flow.
    predicted = ra.answer_question(
        query,
        collapse_tree=_RAPTOR_DEFAULTS["collapse_tree"],
        max_tokens=_RAPTOR_DEFAULTS["retrieval_max_tokens"],
    )

    # Extract evidence sentences from the retrieved layer information.
    # Each entry is {"node_index", "layer_number"}; the actual text is
    # in `retrieved_text` already, separated by paragraph breaks.
    evidence_sentences = [s.strip() for s in retrieved_text.split("\n\n") if s.strip()]

    return ArchitectureResult(
        architecture="raptor",
        predicted_answer=predicted,
        retrieved_evidence_sentences=evidence_sentences,
    )
