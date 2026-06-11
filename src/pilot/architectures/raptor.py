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
from dataclasses import dataclass
from pathlib import Path

# UMAP+GMM clustering inside the vendored RAPTOR repo segfaults on
# Windows under default numba+OpenMP threading. Pinning to single
# thread resolves it without touching the upstream source. The pins
# are applied at import time so callers don't have to remember; if
# the user has already exported these variables, those values win.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pilot.architectures.base import ArchitectureResult, _render_prompt
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

# The single source of truth for the clustering seed that governs tree
# determinism (UMAP random_state + numpy seed). Re-exported so the
# preprocessing cache key can bind to the REAL seed instead of a stale
# hardcoded constant — if this seed ever changes, every cached tree is
# correctly invalidated.
from raptor.cluster_utils import RANDOM_SEED as CLUSTERING_SEED  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Embedder adapter
# ──────────────────────────────────────────────────────────────────────

class _LedgerEmbeddingModel(BaseEmbeddingModel):
    """Adapter routing every embed call through ``CostLedger``.

    The same adapter instance is wired into BOTH the tree-builder
    (which embeds every leaf chunk and every internal-cluster node
    during ``ra.add_documents()``) AND the query-time TreeRetriever
    (which embeds the query against the prebuilt tree). The two phases
    must land in DIFFERENT cost buckets:

      - Build-phase embeddings are paid once per (paper, arch) and
        belong to ``C_off^struct`` → ``Stage.PREPROCESS``.
      - Query-phase embeddings are paid per question and belong to
        ``C_on`` → ``Stage.RETRIEVAL``.

    The ``stage`` attribute is mutable: ``run_raptor`` sets it to
    ``Stage.PREPROCESS`` before invoking ``add_documents`` and flips
    it to ``Stage.RETRIEVAL`` immediately after, before any retrieve
    call. On cache-hit paths the cached adapter already has
    ``stage = RETRIEVAL`` from the build that populated it.

    Cost-model rationale (project.tex §3.4.1): the break-even analysis
    against flat full-context requires a clean separation between
    offline build cost and online per-query cost; lumping
    ~1000 tree-build embeds into per-query RETRIEVAL keeps the
    *total* amortised cost correct but inverts the offline/online
    split, which is what break-even depends on.
    """

    def __init__(
        self,
        *,
        embedder: OllamaEmbedder,
        ledger: CostLedger,
        run_index: int = 0,
        stage: Stage = Stage.PREPROCESS,
    ) -> None:
        self.embedder = embedder
        self.ledger = ledger
        self.run_index = run_index
        self.stage = stage

    def create_embedding(self, text):
        # Ollama's /api/embed accepts a list; we send a 1-element batch.
        # Token count is approximated by char/4 since Ollama doesn't
        # report token usage on embed (the same heuristic used by
        # naive_rag for fair cross-architecture cost comparison).
        with self.ledger.log_call(
            architecture="raptor",
            stage=self.stage,
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

# RAPTOR's answer call goes through the SHARED prompt contract
# (``pilot.architectures.base._render_prompt``) so all four
# architectures answer under one prompt regime for a given
# ``prompt_style`` -- only the retrieved ``{context}`` differs.
#
# Prompt history (audit trail): an early pilot draft used the vendored
# ``GPT3QAModel`` completion prompt with a "less than 5-7 words" cap (a
# text-davinci-003 artifact) that truncated QASPER answers; it was
# replaced by the vendored ``GPT4QAModel`` "give the best full answer"
# chat prompt, which over-corrected into verbose, preamble-laden
# answers ("Based on the provided text...") that the QASPER token-F1
# scorer (no preamble stripping) penalised on precision -- the
# load-bearing reason RAPTOR's F1 sat far below the concise
# architectures. Both per-arch RAPTOR prompts are retired in favour of
# the shared ``_render_prompt`` templates (qa_freeform_literature /
# qa_multiplechoice_literature under prompt_style='literature'), so
# RAPTOR answers in the same concise, no-abstention format as flat /
# naive_rag / graphrag and the QASPER comparison measures content, not
# answer format.


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
        prompt_style: str = "pilot",
    ) -> None:
        self.answerer = answerer
        self.model = model
        self.ledger = ledger
        self.run_index = run_index
        self.max_tokens = max_tokens
        self.prompt_style = prompt_style
        self.current_options: dict[str, str] | None = None

    def answer_question(self, context, question):
        # Shared answer-prompt contract: same templates as flat /
        # naive_rag / graphrag, selected by ``prompt_style``. Only the
        # retrieved ``context`` differs across architectures.
        prompt = _render_prompt(
            context=context,
            query=question,
            options=self.current_options,
            prompt_style=self.prompt_style,
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
    tr_top_k=20,                 # paper §4 "approximately top-20 nodes" (was 10, which
                                 # capped RAPTOR's answer context below its own 2000-token
                                 # budget -- a per-architecture under-feeding confound)
    tr_threshold=0.5,
    # Sarthi 2024 §4 collapsed-tree retrieval picks nodes greedily
    # by similarity until the token budget hits; the threshold mode
    # would prune low-similarity nodes even when budget allows. The
    # paper's headline behaviour is top-k by similarity within budget,
    # which `top_k` selection_mode + 2000-token retrieval_max_tokens
    # delivers. Earlier pilot draft used "threshold" — paper-faithful
    # correction at audit (paper_implementation_audit.md).
    tr_selection_mode="top_k",
    collapse_tree=True,
    retrieval_max_tokens=2000,   # paper §4 "approximately top-20 nodes"
)


@dataclass
class _RaptorState:
    """Per-(run, paper) preprocessing artefact for RAPTOR.

    Built once on the first question for a given paper and reused on
    every subsequent question. ``ra`` carries the already-constructed
    UMAP+GMM tree; ``qa_adapter`` is the live adapter whose
    ``current_options`` slot is mutated per-question to switch between
    free-form and multiple-choice prompting.

    This is the load-bearing object behind the repeated-context
    amortisation claim: without it, ``C_off^struct`` would be paid
    once per question instead of once per document, inverting the
    cost-vs-quality Pareto and breaking the thesis premise.

    Pickle contract
    ---------------
    The intra-process cache only ever needed ``_RaptorState`` to live
    in memory; the on-disk preprocess cache (``pilot.preprocess_cache``)
    later started persisting it across processes by pickling.
    ``ra`` and ``qa_adapter`` both transitively hold an ``OllamaEmbedder``
    whose ``httpx.Client`` carries a ``_thread.lock`` — unpicklable.
    The fix splits pure-data from live adapters:

      - ``__getstate__`` serialises ONLY the pure-data tree
        (``raptor.tree_structures.Tree``: nodes + their embeddings + the
        layer index). The tree alone is what the retrieved-context
        determinism guarantee depends on — two candidates loading the
        same pickle and rehydrating with their own adapters will
        produce byte-identical prompt-hashes because the retrieval
        path walks the same tree under the same TreeRetrieverConfig
        defaults.
      - ``__setstate__`` parks the deserialised tree on
        ``self._restored_tree`` and leaves ``ra`` / ``qa_adapter`` as
        ``None``. Any retrieval / answer call on the half-restored state
        will fail loudly until ``rehydrate(...)`` is called.
      - ``rehydrate(...)`` rebuilds a fresh ``RetrievalAugmentation``
        from the restored tree and the caller-supplied live adapters
        (embedder, ledger, answerer, models). The embedding adapter is
        constructed with ``stage=Stage.RETRIEVAL`` because we are
        post-build by definition on a cache hit; any retrieval-side
        embed lands in the ``C_on`` retrieval bucket per the
        cost-attribution rule (project.tex § 3.4.1).

    Byte-equivalence guarantee
    --------------------------
    Two candidates that load the SAME pickle and rehydrate with
    different live adapters (different answerer, same encoder) MUST
    produce the same generate-stage ``prompt_hash`` for the same
    query. The shared piece is the retrieved context, which depends
    only on (the cached tree, the cached entity vectors, the query
    embedding). The query embedding is byte-identical across
    candidates because both use BGE-M3 via the same Ollama endpoint.
    """
    ra: object | None  # raptor.RetrievalAugmentation
    qa_adapter: "_LedgerQAModel | None"

    def __getstate__(self) -> dict:
        # Pure-data slice only: the already-built tree. Everything else
        # (config objects, retrievers, builders, adapters) gets
        # reconstructed on rehydrate from the caller's fresh adapters.
        tree = None
        ra = self.ra
        if ra is not None:
            tree = getattr(ra, "tree", None)
        return {"tree": tree}

    def __setstate__(self, state: dict) -> None:
        # Park the tree on a private slot; surface it through
        # ``rehydrate`` rather than reconstructing eagerly because
        # rebuilding the ``RetrievalAugmentation`` needs live adapters
        # the unpickler cannot supply.
        self.ra = None
        self.qa_adapter = None
        self._restored_tree = state.get("tree")

    def rehydrate(
        self,
        *,
        embedder: OllamaEmbedder,
        ledger: CostLedger,
        answerer: AnswererProvider,
        answerer_model: str,
        summary_answerer: AnswererProvider | None = None,
        summary_model: str | None = None,
        run_index: int = 0,
        max_tokens: int = 256,
        prompt_style: str = "pilot",
    ) -> "_RaptorState":
        """Rebuild ``ra`` + ``qa_adapter`` from the restored tree.

        Idempotent — calling twice with the same adapters is harmless;
        calling with different adapters re-wires the state to the new
        adapters. Returns ``self`` for ergonomic chaining.

        The summary adapter is wired in for completeness even though
        a rehydrated state never re-enters the build phase: a
        ``RetrievalAugmentation`` constructor requires the field
        populated. ``summary_answerer`` / ``summary_model`` default to
        the answerer side to mirror the intra-process build path.
        """
        if self.ra is not None and self.qa_adapter is not None:
            # Already hydrated — just refresh the answerer wiring in
            # case the caller is swapping adapters on the same state
            # instance.
            self.qa_adapter.answerer = answerer
            self.qa_adapter.model = answerer_model
            self.qa_adapter.ledger = ledger
            self.qa_adapter.run_index = run_index
            self.qa_adapter.max_tokens = max_tokens
            self.qa_adapter.prompt_style = prompt_style
            return self

        tree = getattr(self, "_restored_tree", None)
        if tree is None and self.ra is not None:
            tree = getattr(self.ra, "tree", None)
        if tree is None:
            raise RuntimeError(
                "_RaptorState.rehydrate called without a restored tree; "
                "the pickle was either corrupt or this state was never "
                "built. Rebuild from the source document instead."
            )

        s_answerer = summary_answerer or answerer
        s_model = summary_model or answerer_model

        embedding_adapter = _LedgerEmbeddingModel(
            embedder=embedder, ledger=ledger, run_index=run_index,
            # Post-build on every cache-hit path → every query-time
            # embed lands as RETRIEVAL not PREPROCESS.
            stage=Stage.RETRIEVAL,
        )
        summary_adapter = _LedgerSummarizationModel(
            answerer=s_answerer, model=s_model,
            ledger=ledger, run_index=run_index,
        )
        qa_adapter = _LedgerQAModel(
            answerer=answerer, model=answerer_model,
            ledger=ledger, run_index=run_index, max_tokens=max_tokens,
            prompt_style=prompt_style,
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
            tree_builder_config=tb_cfg,
            tree_retriever_config=tr_cfg,
            qa_model=qa_adapter,
            embedding_model=embedding_adapter,
            summarization_model=summary_adapter,
        )
        # Pass the restored ``Tree`` directly so RetrievalAugmentation
        # initialises its TreeRetriever against the prebuilt structure
        # without re-running the builder.
        self.ra = RetrievalAugmentation(config=cfg, tree=tree)
        self.qa_adapter = qa_adapter
        # Tree is now owned by ``self.ra``; drop the private slot to
        # avoid accidental drift between ``self._restored_tree`` and
        # ``self.ra.tree``.
        if hasattr(self, "_restored_tree"):
            del self._restored_tree
        return self


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
    prompt_style: str = "pilot",
    cached_state: _RaptorState | None = None,
) -> ArchitectureResult:
    """Build a RAPTOR tree over ``document`` and answer ``query``.

    The summary model defaults to the answerer model unless
    overridden. ``summary_answerer`` lets the summary stage run
    on a different provider than the final answerer (Phase F
    extension protocol: e.g. Grok answerer + Gemini Flash Lite
    summary). When ``summary_answerer`` is ``None`` the summary
    calls go through the same provider as the answerer.

    ``cached_state`` — when provided, skip the tree build entirely and
    reuse the prior call's ``RetrievalAugmentation``. The cost ledger
    therefore records the tree-build cost exactly once per
    (run, paper_id), and every subsequent question on that paper
    only pays retrieval + answer. This is the on-disk realisation of
    the ``C_off^struct / n`` amortisation in the cost model
    (project.tex § 3.4.1).
    """
    summary_model = summary_model or answerer_model
    summary_answerer = summary_answerer or answerer

    if cached_state is not None:
        # Cache hit: tree already built on an earlier question for
        # this paper. Mutate the qa_adapter's current_options slot to
        # switch between free-form / MC prompting, then retrieve +
        # answer. No preprocess-stage ledger rows on this path — the
        # embedding adapter inside ``ra`` already had its stage
        # flipped to RETRIEVAL at the end of the build that populated
        # this cache entry.
        #
        # When the state came off disk (loaded from preprocess_cache),
        # its live adapters were stripped on pickle and the rehydrate
        # path must run before the first retrieval call. A state that
        # was passed in directly from an earlier in-process call
        # already has ``ra`` + ``qa_adapter`` populated, so the
        # idempotent rehydrate refreshes the answerer wiring without
        # rebuilding anything.
        cached_state.rehydrate(
            embedder=embedder, ledger=ledger,
            answerer=answerer, answerer_model=answerer_model,
            summary_answerer=summary_answerer, summary_model=summary_model,
            run_index=run_index, max_tokens=max_tokens,
            prompt_style=prompt_style,
        )
        qa_adapter = cached_state.qa_adapter
        qa_adapter.current_options = options
        ra = cached_state.ra
        state = cached_state
    else:
        embedding_adapter = _LedgerEmbeddingModel(
            embedder=embedder, ledger=ledger, run_index=run_index,
            stage=Stage.PREPROCESS,
        )
        summary_adapter = _LedgerSummarizationModel(
            answerer=summary_answerer, model=summary_model,
            ledger=ledger, run_index=run_index,
        )
        qa_adapter = _LedgerQAModel(
            answerer=answerer, model=answerer_model,
            ledger=ledger, run_index=run_index, max_tokens=max_tokens,
            prompt_style=prompt_style,
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
            # Build phase: every leaf-chunk + internal-cluster embed
            # goes through ``embedding_adapter`` and lands in the
            # ledger as ``Stage.PREPROCESS`` (the adapter was
            # instantiated with that stage above).
            ra.add_documents(document)
        except Exception as exc:
            import traceback as _tb, sys as _sys
            print(
                f"[raptor] tree_build_failed query={query[:60]!r} exc={exc!r}",
                file=_sys.stderr,
            )
            _tb.print_exc(file=_sys.stderr)
            return ArchitectureResult(
                architecture="raptor",
                predicted_answer="",
                failed=True,
                failure_reason=f"tree_build_failed: {exc!r}",
            )

        # Build complete — flip the adapter's stage so subsequent
        # query-time embed calls (TreeRetriever embedding the user
        # query against the prebuilt tree) land as ``RETRIEVAL`` per
        # the cost model. This adapter instance is what every cache
        # hit on this paper will use.
        embedding_adapter.stage = Stage.RETRIEVAL

        state = _RaptorState(ra=ra, qa_adapter=qa_adapter)

    # Retrieve to capture the evidence sentences (collapsed-tree mode,
    # paper-default 2000-token budget) BEFORE answering, so we can
    # report what the architecture saw.
    retrieved_text, layer_info = ra.retrieve(
        query,
        collapse_tree=_RAPTOR_DEFAULTS["collapse_tree"],
        top_k=_RAPTOR_DEFAULTS["tr_top_k"],
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
        top_k=_RAPTOR_DEFAULTS["tr_top_k"],
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
        preprocessing_state=state,
    )
