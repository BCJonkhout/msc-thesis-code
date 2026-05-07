"""Tests for pilot.eval.metrics.

Coverage targets:
  - normalize_text matches QASPER's standard normalisation
  - token_f1 / answer_f1 against single and multi-reference cases,
    including unanswerable (empty / empty) handling
  - evidence_f1 sentence-level set-based matching
  - accuracy + parse_mc_answer integration
"""
from __future__ import annotations

from pilot.eval import (
    accuracy,
    answer_f1,
    answer_f1_against_references,
    evidence_f1,
    normalize_text,
    parse_mc_answer,
    token_f1,
)


# ──────────────────────────────────────────────────────────────────────
# normalize_text
# ──────────────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("HELLO World") == "hello world"

    def test_strips_articles(self):
        assert normalize_text("the cat and the dog") == "cat and dog"

    def test_strips_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("hello    \n\n  world") == "hello world"

    def test_handles_none(self):
        assert normalize_text(None) == ""

    def test_handles_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_handles_unicode_punct(self):
        # Curly quotes and em-dashes are common in QASPER.
        assert normalize_text("it's") == "its"


# ──────────────────────────────────────────────────────────────────────
# token_f1 / answer_f1
# ──────────────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_identical_after_normalisation(self):
        # Articles + punctuation differ but normalise to the same string.
        assert token_f1("The Cat, sat.", "the cat sat") == 1.0

    def test_partial_overlap(self):
        # pred=2 tokens, gold=3 tokens, common=2 (no articles)
        # precision=2/2=1.0, recall=2/3=0.666, F1=0.8
        result = token_f1("cat sat", "the cat sat there")
        # pred normalises to "cat sat" (2 tokens)
        # gold normalises to "cat sat there" (3 tokens; "the" stripped)
        # common = 2; precision=1.0, recall=2/3
        assert abs(result - 0.8) < 1e-6

    def test_no_overlap(self):
        assert token_f1("apple banana", "cat dog") == 0.0

    def test_both_empty_after_normalisation(self):
        # QASPER convention: "no answer" matches "no answer" → F1 = 1.0
        assert token_f1("", "") == 1.0
        assert token_f1("the", "the") == 1.0  # both normalise to ""

    def test_one_empty(self):
        assert token_f1("", "cat") == 0.0
        assert token_f1("cat", "") == 0.0

    def test_punctuation_only_pred(self):
        # Pred is just punctuation, normalises to empty string.
        assert token_f1(".", "cat") == 0.0


class TestAnswerF1:
    def test_single_reference_alias(self):
        # answer_f1 is a direct alias of token_f1.
        assert answer_f1("cat", "cat") == 1.0


class TestAnswerF1AgainstReferences:
    def test_takes_max_across_references(self):
        # Pred matches the second reference perfectly.
        result = answer_f1_against_references("cat", ["dog", "cat", "fish"])
        assert result == 1.0

    def test_zero_for_no_match(self):
        result = answer_f1_against_references("cat", ["dog", "fish"])
        assert result == 0.0

    def test_empty_references_returns_zero(self):
        assert answer_f1_against_references("cat", []) == 0.0

    def test_partial_match_returns_best(self):
        # ref1: 2 tokens overlap, ref2: 1 token overlap.
        result = answer_f1_against_references(
            "the quick brown fox", ["quick brown", "the cat"]
        )
        # vs "quick brown": pred=3 ("quick brown fox"), gold=2 ("quick brown"),
        # common=2, P=2/3, R=1.0, F1=0.8.
        # vs "the cat": gold normalises to "cat", pred has no "cat", F1=0.
        assert abs(result - 0.8) < 1e-6


# ──────────────────────────────────────────────────────────────────────
# evidence_f1
# ──────────────────────────────────────────────────────────────────────

class TestEvidenceF1:
    def test_perfect_match(self):
        retrieved = ["The cat sat.", "On the mat."]
        gold = ["The cat sat.", "On the mat."]
        assert evidence_f1(retrieved, gold) == 1.0

    def test_partial_overlap(self):
        # 1 of 2 retrieved matches 1 of 2 gold; precision=0.5, recall=0.5, F1=0.5
        retrieved = ["The cat sat.", "Quick brown fox."]
        gold = ["The cat sat.", "On the mat."]
        result = evidence_f1(retrieved, gold)
        assert abs(result - 0.5) < 1e-6

    def test_no_overlap(self):
        assert evidence_f1(["A"], ["B"]) == 0.0

    def test_both_empty(self):
        # Vacuous case: not a perfect score, per QASPER convention.
        assert evidence_f1([], []) == 0.0

    def test_one_empty(self):
        assert evidence_f1([], ["A"]) == 0.0
        assert evidence_f1(["A"], []) == 0.0

    def test_dedup_within_each_side(self):
        # Two copies of the same sentence on the retrieved side count once.
        retrieved = ["The cat sat.", "The cat sat."]
        gold = ["The cat sat.", "On the mat."]
        # set-based: precision=1/1=1.0, recall=1/2=0.5, F1=0.667
        result = evidence_f1(retrieved, gold)
        assert abs(result - 2 / 3) < 1e-6

    def test_normalised_match(self):
        # Punctuation differs but normalises identically.
        retrieved = ["the cat sat"]
        gold = ["The Cat sat."]
        assert evidence_f1(retrieved, gold) == 1.0


# ──────────────────────────────────────────────────────────────────────
# accuracy + parse_mc_answer integration
# ──────────────────────────────────────────────────────────────────────

class TestAccuracy:
    def test_correct_label(self):
        assert accuracy("A", "A") == 1.0

    def test_incorrect_label(self):
        assert accuracy("B", "A") == 0.0

    def test_case_insensitive(self):
        assert accuracy("a", "A") == 1.0

    def test_strips_whitespace(self):
        assert accuracy("  A  ", "A") == 1.0

    def test_none_prediction_scores_zero(self):
        # Parser failed to extract a letter — treated as a wrong answer,
        # not as an exception, so the eval harness keeps running.
        assert accuracy(None, "A") == 0.0


class TestParseMcAnswerIntegration:
    """Integration tests across parse_mc_answer + accuracy. The parser
    itself is exhaustively tested in test_mc_postprocessor.py; here we
    verify the eval-pipeline end-to-end on a few representative shapes.
    """

    def test_letter(self):
        assert accuracy(parse_mc_answer("A"), "A") == 1.0

    def test_parens(self):
        assert accuracy(parse_mc_answer("(B)"), "B") == 1.0

    def test_letter_with_text(self):
        assert accuracy(parse_mc_answer("A. The Eiffel Tower"), "A") == 1.0

    def test_full_option_text(self):
        options = ["The Eiffel Tower", "The Statue of Liberty"]
        assert accuracy(parse_mc_answer("The Eiffel Tower", options), "A") == 1.0

    def test_unparseable(self):
        # Model output didn't match any rule → parser returns None → score 0.
        assert accuracy(parse_mc_answer("I don't know"), "A") == 0.0
