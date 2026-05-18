"""MC answer post-processor: 100% on the synthetic test set (decision #3)."""
from __future__ import annotations

import pytest

from pilot.sanity import parse_mc_answer

OPTIONS = ["The Eiffel Tower", "Big Ben", "The Statue of Liberty", "The Pyramid"]


# Each tuple: (raw model output, expected option letter or None)
SYNTHETIC_CASES: list[tuple[str, str | None]] = [
    # Bare letter
    ("A", "A"),
    ("B", "B"),
    ("C", "C"),
    ("D", "D"),
    # Bare letter with surrounding whitespace
    ("  A  ", "A"),
    ("\nB\n", "B"),
    # Parenthesised letter
    ("(A)", "A"),
    ("( B )", "B"),
    # Letter with punctuation prefix
    ("A.", "A"),
    ("A:", "A"),
    ("A)", "A"),
    # Letter + period + text
    ("A. The Eiffel Tower", "A"),
    ("B. Big Ben is a clock tower in London.", "B"),
    ("D) The Pyramid", "D"),
    # "Option A" style
    ("Option A", "A"),
    ("option b", "B"),
    ("Answer: C", "C"),
    ("The answer is D", "D"),
    # Full option text echoed
    ("The Eiffel Tower", "A"),
    ("Big Ben", "B"),
    # Loose parenthesised
    ("(option A)", "A"),
    # Leading "Answer:" prefix that some models emit
    ("Answer: A", "A"),
    ("Answer: A. The Eiffel Tower", "A"),
]

INVALID_CASES: list[str] = [
    "",
    "   ",
    "I do not know",
    "The model could not determine the answer.",
    "42",  # not a letter
]


@pytest.mark.parametrize("raw, expected", SYNTHETIC_CASES)
def test_synthetic_set_100pct(raw: str, expected: str) -> None:
    """All 23 synthetic cases parse correctly."""
    got = parse_mc_answer(raw, options=OPTIONS)
    assert got == expected, f"input={raw!r}: expected {expected}, got {got}"


@pytest.mark.parametrize("raw", INVALID_CASES)
def test_invalid_returns_none(raw: str) -> None:
    """Cases the parser cannot resolve return None rather than guessing."""
    assert parse_mc_answer(raw, options=OPTIONS) is None


def test_full_text_substring_match() -> None:
    """A free-form answer that contains an option's text resolves to that letter."""
    assert parse_mc_answer("The answer is the Eiffel Tower in Paris.", options=OPTIONS) == "A"


# --- Regression tests for Phase G mis-scoring bugs ---------------------------


def test_options_plural_does_not_yield_S() -> None:
    """Pattern 4 must not capture the 's' in 'options' under IGNORECASE.

    Earlier the keyword pattern `\\b(?:option|answer|choice)...([A-Z])` would,
    on input mentioning "options", capture the trailing 's' and uppercase to
    'S'. Adding `\\b` after the keyword group prevents the partial match.
    """
    raw = (
        "A\n\nThe relevant passage occurs in chapter three. "
        "The other options do not match the description."
    )
    assert parse_mc_answer(raw, options=OPTIONS) == "A"


def test_letter_newline_explanation_reasoning_model() -> None:
    """Reasoning-tuned models emit '<letter>\\n\\n<justification>'.

    Verifies the new early pattern (letter followed by one or more newlines)
    resolves the letter before any keyword pattern can fire on the body.
    """
    raw = "D\n\nThe context describes the construction date and architect."
    assert parse_mc_answer(raw, options=OPTIONS) == "D"


def test_letter_newline_variants() -> None:
    """Single-newline and trailing-content variants also resolve."""
    assert parse_mc_answer("A\nBecause ...", options=OPTIONS) == "A"
    assert parse_mc_answer("  B  \n\nReasoning follows.", options=OPTIONS) == "B"


def test_substring_fallback_word_boundary_yes_no() -> None:
    """'no' must not match inside 'not' for yes/no option sets.

    Before the word-boundary fix, "I do not know" with options ["Yes", "No"]
    incorrectly resolved to 'B' (No) via plain substring containment.
    """
    yn_options = ["Yes", "No"]
    assert parse_mc_answer("I do not know", options=yn_options) is None


def test_substring_fallback_multi_word_still_matches() -> None:
    """Multi-word option text still resolves via the word-boundary fallback."""
    opts = ["Spanish necklace", "Italian ring", "French brooch"]
    assert (
        parse_mc_answer("She found a Spanish necklace in the drawer.", options=opts)
        == "A"
    )
