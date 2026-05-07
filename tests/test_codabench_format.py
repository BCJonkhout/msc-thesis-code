"""Tests for pilot.codabench.format.

Coverage targets:
  - res_mc.json schema matches NovelQA's leaderboard expectation
    (keyed by novel title, lists in QID order, letters A/B/C/D)
  - placeholder fill-in keeps the question count per novel exact
  - covered/missing stats are correct
  - the zip contains exactly one entry, ``res_mc/res_mc.json``
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pilot.codabench.format import (
    NOVELQA_PLACEHOLDER_LETTER,
    build_res_mc,
    write_submission_zip,
)


@pytest.fixture
def fake_data_root(tmp_path: Path) -> Path:
    """Construct a tiny NovelQA-shaped dataset on disk."""
    novelqa = tmp_path / "novelqa"
    (novelqa / "full_texts").mkdir(parents=True)
    (novelqa / "full_texts" / "B01.txt").write_text("Mrs Dalloway text.")
    (novelqa / "full_texts" / "B08.txt").write_text("Rebecca text.")

    bookmeta = {
        "B01": {"title": "Mrs. Dalloway", "tokenlen": 91706},
        "B08": {"title": "Rebecca of Sunnybrook Farm", "tokenlen": 106734},
    }
    (novelqa / "bookmeta.json").write_text(json.dumps(bookmeta))

    # questions.jsonl: 3 questions for B01, 2 for B08, in QID order.
    questions = [
        {"novel_id": "B01", "question_id": "Q0001", "Question": "?", "Options": {}},
        {"novel_id": "B01", "question_id": "Q0002", "Question": "?", "Options": {}},
        {"novel_id": "B01", "question_id": "Q0003", "Question": "?", "Options": {}},
        {"novel_id": "B08", "question_id": "Q0100", "Question": "?", "Options": {}},
        {"novel_id": "B08", "question_id": "Q0101", "Question": "?", "Options": {}},
    ]
    with (novelqa / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q))
            fh.write("\n")
    return tmp_path


@pytest.fixture
def predictions_jsonl(tmp_path: Path) -> Path:
    """Predictions for 1 of 3 B01 questions and 1 of 2 B08 questions."""
    rows = [
        {"dataset": "novelqa", "novel_id": "B01", "question_id": "Q0002",
         "predicted_letter": "B", "predicted_answer": "B"},
        {"dataset": "novelqa", "novel_id": "B08", "question_id": "Q0101",
         "predicted_letter": "C", "predicted_answer": "C"},
        # A QASPER row to make sure it's filtered out.
        {"dataset": "qasper", "paper_id": "1234", "question_id": "Qx",
         "predicted_answer": "irrelevant"},
    ]
    p = tmp_path / "predictions.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r))
            fh.write("\n")
    return p


def test_build_res_mc_keys_by_title(fake_data_root: Path, predictions_jsonl: Path):
    res_mc, _ = build_res_mc(predictions_jsonl, data_root=fake_data_root)
    assert set(res_mc.keys()) == {"Mrs. Dalloway", "Rebecca of Sunnybrook Farm"}


def test_letters_are_in_qid_order(fake_data_root: Path, predictions_jsonl: Path):
    res_mc, _ = build_res_mc(predictions_jsonl, data_root=fake_data_root)
    # B01: only Q0002 has a prediction (B); Q0001 + Q0003 fall back to placeholder.
    assert res_mc["Mrs. Dalloway"] == ["A", "B", "A"]
    # B08: only Q0101 has a prediction (C); Q0100 falls back to placeholder.
    assert res_mc["Rebecca of Sunnybrook Farm"] == ["A", "C"]


def test_stats_count_covered_and_missing(
    fake_data_root: Path, predictions_jsonl: Path
):
    _, stats = build_res_mc(predictions_jsonl, data_root=fake_data_root)
    assert stats["novels"] == 2
    assert stats["questions_total"] == 5
    assert stats["covered_by_predictions"] == 2
    assert stats["filled_with_placeholder"] == 3
    assert stats["placeholder_letter"] == NOVELQA_PLACEHOLDER_LETTER


def test_unparseable_letters_are_treated_as_missing(
    fake_data_root: Path, tmp_path: Path
):
    rows = [
        {"dataset": "novelqa", "novel_id": "B01", "question_id": "Q0001",
         "predicted_letter": None},
        {"dataset": "novelqa", "novel_id": "B01", "question_id": "Q0002",
         "predicted_letter": "X"},  # invalid letter
        {"dataset": "novelqa", "novel_id": "B01", "question_id": "Q0003",
         "predicted_letter": "B"},  # the only valid one
    ]
    p = tmp_path / "predictions_with_bad.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r))
            fh.write("\n")
    res_mc, stats = build_res_mc(p, data_root=fake_data_root)
    assert res_mc["Mrs. Dalloway"] == ["A", "A", "B"]
    assert stats["covered_by_predictions"] == 1


def test_write_submission_zip_creates_correct_layout(
    fake_data_root: Path, predictions_jsonl: Path, tmp_path: Path
):
    out = tmp_path / "submission.zip"
    stats = write_submission_zip(
        predictions_jsonl=predictions_jsonl,
        output_zip=out,
        data_root=fake_data_root,
        include_gen_stub=False,
    )
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert names == ["res_mc/res_mc.json"]
        body = json.loads(zf.read("res_mc/res_mc.json").decode("utf-8"))
    assert "Mrs. Dalloway" in body
    assert body["Mrs. Dalloway"][1] == "B"  # the one we predicted
    assert stats["output_zip"] == str(out)
    assert stats["output_zip_bytes"] > 0


def test_write_submission_zip_with_gen_stub(
    fake_data_root: Path, predictions_jsonl: Path, tmp_path: Path
):
    """The default include_gen_stub=True adds res_gen/ to work around
    the platform's scoring-script crash on res_mc-only submissions."""
    out = tmp_path / "submission_with_gen.zip"
    stats = write_submission_zip(
        predictions_jsonl=predictions_jsonl,
        output_zip=out,
        data_root=fake_data_root,
    )
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
        assert names == [
            "res_gen/key",
            "res_gen/res_gen.json",
            "res_mc/res_mc.json",
        ]
        gen = json.loads(zf.read("res_gen/res_gen.json").decode("utf-8"))
        # Empty strings, one per question, keyed by novel title.
        assert gen["Mrs. Dalloway"] == ["", "", ""]
        assert gen["Rebecca of Sunnybrook Farm"] == ["", ""]
    assert stats["include_gen_stub"] is True


def test_explicit_placeholder_override(
    fake_data_root: Path, predictions_jsonl: Path
):
    res_mc, stats = build_res_mc(
        predictions_jsonl, data_root=fake_data_root, placeholder="D"
    )
    # Predictions: B01 Q0002 -> "B", B08 Q0101 -> "C". Rest -> "D".
    assert res_mc["Mrs. Dalloway"] == ["D", "B", "D"]
    assert stats["placeholder_letter"] == "D"
