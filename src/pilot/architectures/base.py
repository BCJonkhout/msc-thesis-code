"""Architecture base + flat full-context runner.

Per pilot plan § 3.4.1 every LLM call writes a `CostLedger` row with
the same schema regardless of architecture; this is what makes the
unified cost accounting possible. The shape of every architecture
runner is therefore the same: a function that takes
(document_or_corpus, query, options, answerer, ledger, run_index)
and returns an ``ArchitectureResult`` carrying the predicted answer
text and any retrieved evidence sentences (used for QASPER
Evidence-F1 and for prompt-construction debugging).

Flat full-context is the simplest case: concatenate the entire
document into the prompt's ``{context}`` slot and call the answerer
exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pilot.ledger import CostLedger, Stage, sha256_hex
from pilot.prompts.loader import load_template
from pilot.providers.base import AnswererProvider, CacheControl


@dataclass
class ArchitectureResult:
    """One (architecture, query) outcome.

    The cost ledger is the canonical record of *what each call cost*;
    this struct is the canonical record of *what the architecture
    answered*. The two are joined offline by ``run_id`` for analysis.
    """

    architecture: str
    predicted_answer: str
    retrieved_evidence_sentences: list[str] = field(default_factory=list)
    prompt_token_count: int | None = None
    response_token_count: int | None = None
    failed: bool = False
    failure_reason: str | None = None
    # Opaque per-architecture preprocessing artefact (RAPTOR tree,
    # GraphRAG knowledge graph + community reports, ...). The dispatcher
    # caches this per-(architecture, paper_id) and threads it back into
    # the next call on the same paper so RAPTOR + GraphRAG don't pay
    # the build cost N times when the calibration pool has N questions
    # on the same document. None for architectures with no preprocessing
    # (flat, naive_rag).
    preprocessing_state: object | None = None


def _format_options_block(options: dict[str, str] | None) -> str:
    """Render an options dict as 'A. text\nB. text\n...' for MC prompts."""
    if not options:
        return ""
    lines = []
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)


def _render_prompt(
    *,
    context: str,
    query: str,
    options: dict[str, str] | None,
    prompt_style: str = "pilot",
) -> str:
    """Render the right template for the task format.

    ``prompt_style`` selects between two free-form QA templates:

      - "pilot" (default): the conservative pilot template with the
        ``respond exactly with `I do not know``` clause. The MC template
        is paired with this style.
      - "literature": a literature-standard concise-answer template
        without the abstention instruction. Used for benchmark
        comparability runs (Step 5 prompt-wording ablation per pilot
        plan § 5 Step 5). MC questions also use the standard MC
        template — letter-choice format is unchanged.

    Multiple-choice questions always use the MC template regardless
    of ``prompt_style``; the MC format does not have the abstention
    issue that motivates the dual-prompt comparison for free-form.
    """
    if options:
        template = load_template("qa_multiplechoice")
        return template.render(
            context=context,
            query=query,
            options=_format_options_block(options),
        )
    if prompt_style == "literature":
        template = load_template("qa_freeform_literature")
    else:
        template = load_template("qa_freeform")
    return template.render(context=context, query=query)


def run_flat(
    *,
    document: str,
    query: str,
    options: dict[str, str] | None,
    answerer: AnswererProvider,
    answerer_model: str,
    ledger: CostLedger,
    run_index: int = 0,
    cache_control: CacheControl = CacheControl.EPHEMERAL_5MIN,
    max_tokens: int = 256,
    temperature: float = 0.0,
    prompt_style: str = "pilot",
) -> ArchitectureResult:
    """Flat full-context: send entire document + query to the answerer.

    The whole document goes into the prompt's `{context}` slot. No
    retrieval step. ``retrieved_evidence_sentences`` is left empty —
    flat full-context has no notion of evidence selection (the model
    sees everything and the cost-vs-quality tradeoff is the
    measurement we care about).

    ``prompt_style`` selects between the conservative pilot template
    and the literature-standard concise-answer template (Step 5
    prompt-wording ablation per pilot plan § 5 Step 5). MC questions
    are unaffected; both styles share the same MC template.
    """
    prompt = _render_prompt(
        context=document, query=query, options=options, prompt_style=prompt_style
    )

    with ledger.log_call(
        architecture="flat",
        stage=Stage.GENERATE,
        model=answerer_model,
        prompt=prompt,
        run_index=run_index,
        temperature=temperature,
        max_tokens=max_tokens,
    ) as rec:
        result = answerer.call(
            prompt,
            model=answerer_model,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_control=cache_control,
        )
        rec.uncached_input_tokens = result.uncached_input_tokens
        rec.cached_input_tokens = result.cached_input_tokens
        rec.output_tokens = result.output_tokens
        rec.provider_request_id = result.provider_request_id
        rec.provider_region = result.provider_region
        rec.response_hash = sha256_hex(result.text or "")

    return ArchitectureResult(
        architecture="flat",
        predicted_answer=result.text,
        retrieved_evidence_sentences=[],
        prompt_token_count=result.uncached_input_tokens + result.cached_input_tokens,
        response_token_count=result.output_tokens,
    )
