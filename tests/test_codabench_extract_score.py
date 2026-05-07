"""Tests for pilot.codabench.extract_score.

Coverage targets:

  - The internal parser correctly extracts per-novel T/F strings
    from a Codabench scoring_stdout-shaped sample.
  - Calibration-position slicing aligns the platform's per-novel
    string with the pilot's per-novel question-id ordering and
    counts T's only at calibration positions.
  - Title normalisation matches the platform's "lowercase no
    punctuation no whitespace" convention.
  - Empty / malformed inputs surface an explicit error rather
    than a silent zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.codabench import extract_score as es


# ──────────────────────────────────────────────────────────────────────
# Internal parser
# ──────────────────────────────────────────────────────────────────────

class TestNovelLineParser:
    def test_extracts_one_line(self):
        text = '"mrsdalloway": "TFTFT",\n'
        out = dict(es._NOVEL_LINE_RE.findall(text))
        assert out == {"mrsdalloway": "TFTFT"}

    def test_extracts_multiple_lines(self):
        text = (
            'log header here\n'
            '"mrsdalloway": "TFTFT", \n'
            '"whitefang": "FFFTT",\n'
            '"mainstreet": "TTTTT"\n'
            'tail noise\n'
        )
        out = dict(es._NOVEL_LINE_RE.findall(text))
        assert out == {
            "mrsdalloway": "TFTFT",
            "whitefang": "FFFTT",
            "mainstreet": "TTTTT",
        }

    def test_keys_must_start_with_lowercase(self):
        # The Codabench scoring script lowercase-no-punctuates titles.
        # Lines starting with uppercase or digit are not treated as
        # novel-correctness rows by our regex.
        text = (
            '"Mrs. Dalloway": "TFTFT",\n'
            '"123novel": "TT",\n'
            '"validkey": "FFFT"\n'
        )
        out = dict(es._NOVEL_LINE_RE.findall(text))
        assert out == {"validkey": "FFFT"}

    def test_skips_lines_with_non_TF_chars(self):
        text = (
            '"validkey": "TFTFT",\n'
            '"another": "ABCDE",\n'   # not T/F → ignored
            '"third": "TTTT"\n'
        )
        out = dict(es._NOVEL_LINE_RE.findall(text))
        assert out == {"validkey": "TFTFT", "third": "TTTT"}


# ──────────────────────────────────────────────────────────────────────
# Title normalisation
# ──────────────────────────────────────────────────────────────────────

class TestNormTitle:
    def test_lowercases(self):
        assert es._norm_title("Mrs. Dalloway") == "mrsdalloway"

    def test_strips_all_punctuation_and_whitespace(self):
        assert es._norm_title("White Fang!") == "whitefang"
        assert es._norm_title("A Tale of Two Cities") == "ataleoftwocities"

    def test_unicode_punctuation_stripped(self):
        # Curly quotes, em-dashes, accents — the platform lowercases and
        # drops anything that isn't [a-z0-9].
        assert es._norm_title("À la recherche") == "larecherche"


# ──────────────────────────────────────────────────────────────────────
# Calibration-position slicing
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_data_root(tmp_path: Path) -> Path:
    """Build a tiny NovelQA-shaped layout on disk for slicing tests."""
    novelqa = tmp_path / "novelqa"
    novelqa.mkdir(parents=True)
    bookmeta = {
        "B01": {"title": "Mrs. Dalloway"},
        "B08": {"title": "White Fang"},
    }
    (novelqa / "bookmeta.json").write_text(json.dumps(bookmeta))

    # Question order in the platform's reference: B01 has 5 Qs, B08 has 4.
    # Calibration pool picks 2 from each, at known positions.
    questions = [
        {"novel_id": "B01", "question_id": "Q01"},
        {"novel_id": "B01", "question_id": "Q02"},
        {"novel_id": "B01", "question_id": "Q03"},
        {"novel_id": "B01", "question_id": "Q04"},
        {"novel_id": "B01", "question_id": "Q05"},
        {"novel_id": "B08", "question_id": "Q10"},
        {"novel_id": "B08", "question_id": "Q11"},
        {"novel_id": "B08", "question_id": "Q12"},
        {"novel_id": "B08", "question_id": "Q13"},
    ]
    with (novelqa / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q))
            fh.write("\n")

    # Calibration pool: B01 picks Q02, Q05; B08 picks Q10, Q12.
    cal = [
        {"novel_id": "B01", "question_id": "Q02"},
        {"novel_id": "B01", "question_id": "Q05"},
        {"novel_id": "B08", "question_id": "Q10"},
        {"novel_id": "B08", "question_id": "Q12"},
    ]
    with (novelqa / "calibration_pool.jsonl").open("w", encoding="utf-8") as fh:
        for c in cal:
            fh.write(json.dumps(c))
            fh.write("\n")

    return tmp_path


class TestCalibrationIndices:
    def test_returns_correct_positions_per_novel(self, fake_data_root: Path):
        idx = es._calibration_indices(fake_data_root)
        # B01 calibration QIDs Q02 (pos 1) and Q05 (pos 4)
        # B08 calibration QIDs Q10 (pos 0) and Q12 (pos 2)
        assert idx == {"B01": [1, 4], "B08": [0, 2]}

    def test_missing_pool_returns_empty(self, tmp_path: Path):
        # No calibration_pool.jsonl on disk → empty mapping.
        novelqa = tmp_path / "novelqa"
        novelqa.mkdir(parents=True)
        (novelqa / "questions.jsonl").write_text("")
        assert es._calibration_indices(tmp_path) == {}


# ──────────────────────────────────────────────────────────────────────
# extract_mc_accuracy with mocked log fetch
# ──────────────────────────────────────────────────────────────────────

class TestExtractMcAccuracy:
    def test_overall_and_calibration_accuracy(
        self, fake_data_root: Path, monkeypatch
    ):
        """Full integration: stub fetch_correctness_strings to return
        our two-novel sample, verify both overall and calibration
        slices."""
        # B01: 5 questions, T's at positions 1 and 3 (index 1=Q02, index 3=Q04)
        # B08: 4 questions, all T
        platform_strings = {
            "mrsdalloway": "FTTFT",   # pos 0 F, 1 T, 2 T, 3 F, 4 T
            "whitefang": "TTTT",       # all T
        }
        monkeypatch.setattr(
            es, "fetch_correctness_strings",
            lambda submission_id: platform_strings,
        )

        out = es.extract_mc_accuracy(submission_id=999, data_root=fake_data_root)
        assert "error" not in out
        assert out["submission_id"] == 999
        # Overall: 3 + 4 = 7 T's out of 5 + 4 = 9.
        assert out["total_correct"] == 7
        assert out["total_questions"] == 9
        # extract_mc_accuracy rounds to 4 decimals; allow tolerance.
        assert abs(out["overall_accuracy"] - 7 / 9) < 1e-4
        # Calibration:
        #   B01 positions [1, 4] → "T" at 1, "T" at 4 → 2/2
        #   B08 positions [0, 2] → "T" at 0, "T" at 2 → 2/2
        assert out["calibration_total_correct"] == 4
        assert out["calibration_total_questions"] == 4
        assert out["calibration_accuracy"] == 1.0
        assert out["calibration_breakdown"]["B01"]["accuracy"] == 1.0
        assert out["calibration_breakdown"]["B08"]["accuracy"] == 1.0

    def test_partial_calibration_signal(
        self, fake_data_root: Path, monkeypatch
    ):
        """Some calibration positions correct, some not: the slicing
        should report the right per-novel counts."""
        platform_strings = {
            "mrsdalloway": "FFFFF",   # nothing correct
            "whitefang": "TFTF",       # 2 of 4 T (pos 0 + 2)
        }
        monkeypatch.setattr(
            es, "fetch_correctness_strings",
            lambda submission_id: platform_strings,
        )
        out = es.extract_mc_accuracy(submission_id=1, data_root=fake_data_root)
        # B01 calibration positions [1, 4]: both F → 0/2
        # B08 calibration positions [0, 2]: T, T → 2/2
        assert out["calibration_total_correct"] == 2
        assert out["calibration_total_questions"] == 4
        assert out["calibration_accuracy"] == 0.5
        assert out["calibration_breakdown"]["B01"]["accuracy"] == 0.0
        assert out["calibration_breakdown"]["B08"]["accuracy"] == 1.0

    def test_empty_log_returns_explicit_error(
        self, fake_data_root: Path, monkeypatch
    ):
        monkeypatch.setattr(
            es, "fetch_correctness_strings",
            lambda submission_id: {},
        )
        out = es.extract_mc_accuracy(submission_id=1, data_root=fake_data_root)
        assert "error" in out
        assert "scoring_stdout" in out["error"] or "T-F" in out["error"]

    def test_calibration_position_out_of_range_handled(
        self, fake_data_root: Path, monkeypatch
    ):
        """If the platform's correctness string is shorter than the
        calibration pool expects (e.g., the platform truncated for
        some reason), positions beyond the string length are skipped."""
        platform_strings = {
            "mrsdalloway": "FT",       # only 2 chars; cal positions are 1, 4
            "whitefang": "TFTF",        # 4 chars; cal positions 0, 2 OK
        }
        monkeypatch.setattr(
            es, "fetch_correctness_strings",
            lambda submission_id: platform_strings,
        )
        out = es.extract_mc_accuracy(submission_id=1, data_root=fake_data_root)
        # B01: position 1 in "FT" is "T" → 1 correct of 1 in-range
        #      position 4 is out of range → not counted
        # B08: positions 0, 2 → "T", "T" → 2 of 2
        assert out["calibration_breakdown"]["B01"]["correct"] == 1
        assert out["calibration_breakdown"]["B01"]["total"] == 1
        assert out["calibration_breakdown"]["B08"]["correct"] == 2
        assert out["calibration_breakdown"]["B08"]["total"] == 2
