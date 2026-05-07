"""Codabench submission pipeline for the NovelQA leaderboard.

The official NovelQA leaderboard at codabench.org/competitions/2727
accepts a `submission.zip` containing:

  res_mc/
    res_mc.json   — multiple-choice predictions, keyed by novel title

Submission schema per NovelQA's official instructions:
    {
      "<novel title>": ["ans0", "ans1", "ans2", ...],
      ...
    }
where the list is in the same QID order as the input JSON in
`Data/PublicDomain/B*.json` and each entry is the predicted option
letter ("A", "B", "C", or "D").

Two surfaces in this package:

  - format: build a res_mc.json + zip from the dry run's
    per-architecture predictions JSONL. Pure-offline; deterministic.
  - submit: drive the live REST sequence to upload + poll +
    fetch leaderboard. Requires a Codabench session (cookie file
    or env var) since the platform doesn't currently expose
    per-user API tokens for scripts.
"""
from pilot.codabench.format import (
    NOVELQA_PLACEHOLDER_LETTER,
    build_res_mc,
    write_submission_zip,
)
from pilot.codabench.extract_score import extract_mc_accuracy

__all__ = [
    "NOVELQA_PLACEHOLDER_LETTER",
    "build_res_mc",
    "extract_mc_accuracy",
    "write_submission_zip",
]
