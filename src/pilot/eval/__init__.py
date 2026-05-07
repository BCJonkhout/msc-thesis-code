"""Pilot evaluation metrics.

The pilot uses three primary answer-quality metrics, one per task
format (per pilot plan § 3.1 Tasks and § 3.4.2 Judging Method):

  - Answer-F1:    QASPER free-form short answers vs gold annotations.
  - Evidence-F1:  QASPER paragraph-level evidence retrieval vs gold
                  highlighted_evidence sentences (diagnostic only).
  - Accuracy:     NovelQA + QuALITY multiple-choice option selection.

Implementations follow the official QASPER and NovelQA evaluation
scripts where possible; deviations are documented inline.
"""
from pilot.eval.metrics import (
    accuracy,
    answer_f1,
    answer_f1_against_references,
    evidence_f1,
    normalize_text,
    parse_mc_answer,
    token_f1,
)

__all__ = [
    "accuracy",
    "answer_f1",
    "answer_f1_against_references",
    "evidence_f1",
    "normalize_text",
    "parse_mc_answer",
    "token_f1",
]
