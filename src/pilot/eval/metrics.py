"""Evaluation metrics: Answer-F1, Evidence-F1, Accuracy, MC parsing.

The pilot evaluates two task formats with three metrics:

  - Answer-F1 (QASPER free-form short answers).
    Token-level F1 between predicted and gold answer text after
    QASPER's normalisation: lowercase, strip articles, strip
    punctuation, collapse whitespace. When a question has multiple
    annotator answers, the *max* F1 across annotators is taken
    (per the QASPER official evaluation script).

  - Evidence-F1 (QASPER, diagnostic only).
    Sentence-level F1 between the union of retrieved chunks (the
    architecture's evidence selection) and the union of gold
    `highlighted_evidence` sentences. Reported per pilot plan
    § 3.4.2 alongside Answer-F1; not used to drive decisions.

  - Accuracy (NovelQA + QuALITY multiple choice).
    1 if `parse_mc_answer(prediction) == gold_label` else 0. The
    parser handles ``"A"``, ``"(A)"``, ``"Option A"``, ``"A. text"``,
    and full-option-text fallback.

These metrics are pure deterministic functions; tests for the
parser live at ``tests/test_mc_postprocessor.py`` and tests for
the F1 metrics at ``tests/test_metrics.py``.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Sequence

# Re-export the MC parser so callers have a single import surface.
from pilot.sanity.mc_postprocessor import parse_mc_answer

__all__ = [
    "accuracy",
    "answer_f1",
    "answer_f1_against_references",
    "evidence_f1",
    "normalize_text",
    "parse_mc_answer",
    "token_f1",
]


# ──────────────────────────────────────────────────────────────────────
# Text normalisation (QASPER official-eval-script compatible)
# ──────────────────────────────────────────────────────────────────────

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """QASPER's standard normalisation: lowercase, strip articles +
    punctuation, collapse whitespace.

    Reproduces the normalisation used by the official QASPER
    `qasper_evaluator.py` so our token-F1 numbers are comparable
    to published QASPER results.
    """
    if text is None:
        return ""
    # QASPER's official evaluator removes punctuation entirely (not
    # replaces with space), so "U.S.A." normalises to "usa" and
    # "it's" normalises to "its". Replicating that behaviour keeps
    # our F1 numbers comparable to published QASPER results.
    s = text.lower()
    s = _PUNCT_RE.sub("", s)
    s = _ARTICLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ──────────────────────────────────────────────────────────────────────
# Token-level F1
# ──────────────────────────────────────────────────────────────────────

def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between two strings after QASPER normalisation.

    Returns 0.0 if either side is empty after normalisation, except
    when both are empty in which case the answer is "no answer" and
    F1 is 1.0 (matches QASPER convention for unanswerable questions).
    """
    p_tokens = normalize_text(prediction).split()
    g_tokens = normalize_text(gold).split()

    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0

    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, gold: str) -> float:
    """Single-reference QASPER Answer-F1 (token-level F1 with
    QASPER normalisation)."""
    return token_f1(prediction, gold)


def answer_f1_against_references(
    prediction: str, references: Sequence[str]
) -> float:
    """Multi-reference Answer-F1: max token-F1 across reference answers.

    QASPER provides up to N annotator answers per question; the
    official metric takes the maximum F1 over annotators so the
    architecture is not penalised for choosing a defensible
    non-canonical phrasing.
    """
    if not references:
        return 0.0
    return max(token_f1(prediction, ref) for ref in references)


# ──────────────────────────────────────────────────────────────────────
# Evidence-F1 (QASPER, sentence-level)
# ──────────────────────────────────────────────────────────────────────

def _sentence_set(sentences: Iterable[str]) -> set[str]:
    """Normalise + deduplicate a collection of sentences for set comparison."""
    out: set[str] = set()
    for s in sentences:
        if not s:
            continue
        norm = normalize_text(s)
        if norm:
            out.add(norm)
    return out


def evidence_f1(
    retrieved_sentences: Iterable[str],
    gold_sentences: Iterable[str],
) -> float:
    """Sentence-level F1 between retrieved and gold evidence sentences.

    The metric treats each sentence as a token in a multiset; we
    compute set-F1 (each sentence counted at most once on each side).
    A retrieved sentence "matches" a gold sentence if they normalise
    to the same string. This is stricter than the architecture-level
    "is there overlap" rule used by the encoder Recall@k experiment;
    Evidence-F1 is the headline metric for QASPER's diagnostic
    evidence-grounding question.

    Returns 0.0 if both sides are empty (no gold evidence and no
    retrieved evidence is treated as a vacuous case, not a perfect
    score, per QASPER's evaluator).
    """
    p_set = _sentence_set(retrieved_sentences)
    g_set = _sentence_set(gold_sentences)

    if not p_set and not g_set:
        return 0.0
    if not p_set or not g_set:
        return 0.0

    intersection = p_set & g_set
    if not intersection:
        return 0.0
    precision = len(intersection) / len(p_set)
    recall = len(intersection) / len(g_set)
    return 2 * precision * recall / (precision + recall)


# ──────────────────────────────────────────────────────────────────────
# Multiple-choice accuracy
# ──────────────────────────────────────────────────────────────────────

def accuracy(prediction_label: str | None, gold_label: str) -> float:
    """0/1 accuracy on a single (predicted_letter, gold_letter) pair.

    The caller is expected to have already passed the model's raw
    output through ``parse_mc_answer`` to extract the option letter.
    Accepts ``None`` predictions (parsing failed) and scores them 0.
    """
    if prediction_label is None:
        return 0.0
    return 1.0 if prediction_label.strip().upper() == gold_label.strip().upper() else 0.0
