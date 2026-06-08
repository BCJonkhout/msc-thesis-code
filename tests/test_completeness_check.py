"""Tests for the completeness gate (pilot.cli.completeness_check)."""
from __future__ import annotations

import json
from pathlib import Path

from pilot.cli import completeness_check as cc


def _make_qasper_pool(data_root: Path, pairs: list[tuple[str, str]]) -> None:
    qasper = data_root / "qasper"
    qasper.mkdir(parents=True, exist_ok=True)
    papers, cal = [], []
    for pid, qid in pairs:
        papers.append({
            "paper_id": pid, "title": pid, "abstract": "x",
            "full_text": [{"section_name": "s", "paragraphs": ["t"]}],
            "qas": [{"question_id": qid, "question": "?",
                     "answers": [{"answer": {"free_form_answer": "y",
                                             "highlighted_evidence": []}}]}],
        })
        cal.append({"paper_id": pid, "question_id": qid, "question": "?"})
    (qasper / "dev.jsonl").write_text(
        "\n".join(json.dumps(p) for p in papers) + "\n", encoding="utf-8")
    (qasper / "calibration_pool.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cal) + "\n", encoding="utf-8")


def _write_rows(run_dir: Path, arch: str, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / f"{arch}_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_expected_cells_is_full_grid(tmp_path: Path):
    data_root = tmp_path / "data"
    _make_qasper_pool(data_root, [("p0", "q0"), ("p1", "q1")])
    cells = cc.expected_cells(
        data_root=data_root, datasets=["qasper"],
        architectures=["flat", "raptor"], num_runs=3,
    )
    # 2 papers x 2 archs x 3 runs = 12 cells.
    assert len(cells) == 12
    assert ("flat", "p0", "q0", 0) in cells
    assert ("raptor", "p1", "q1", 2) in cells


def test_missing_detects_gap(tmp_path: Path):
    data_root = tmp_path / "data"
    _make_qasper_pool(data_root, [("p0", "q0"), ("p1", "q1")])
    run_dir = tmp_path / "runs" / "r"
    # Only one of two papers done at run_index 0 for flat; raptor empty.
    _write_rows(run_dir, "flat", [
        {"paper_id": "p0", "question_id": "q0", "run_index": 0, "answer_f1": 0.5},
    ])
    missing = cc.missing_cells(
        run_dir=run_dir, data_root=data_root, datasets=["qasper"],
        architectures=["flat", "raptor"], num_runs=1,
    )
    # flat missing (p1,q1,0); raptor missing both.
    assert ("flat", "p1", "q1", 0) in missing
    assert ("raptor", "p0", "q0", 0) in missing
    assert ("raptor", "p1", "q1", 0) in missing
    assert ("flat", "p0", "q0", 0) not in missing
    assert len(missing) == 3


def test_complete_run_has_no_missing(tmp_path: Path):
    data_root = tmp_path / "data"
    _make_qasper_pool(data_root, [("p0", "q0")])
    run_dir = tmp_path / "runs" / "r"
    for arch in ("flat", "raptor"):
        _write_rows(run_dir, arch, [
            {"paper_id": "p0", "question_id": "q0", "run_index": 0, "answer_f1": 1.0},
            {"paper_id": "p0", "question_id": "q0", "run_index": 1, "answer_f1": 1.0},
        ])
    missing = cc.missing_cells(
        run_dir=run_dir, data_root=data_root, datasets=["qasper"],
        architectures=["flat", "raptor"], num_runs=2,
    )
    assert missing == set()


def test_torn_trailing_line_tolerated(tmp_path: Path):
    data_root = tmp_path / "data"
    _make_qasper_pool(data_root, [("p0", "q0")])
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    with (run_dir / "flat_predictions.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"paper_id": "p0", "question_id": "q0",
                             "run_index": 0, "answer_f1": 1.0}) + "\n")
        fh.write('{"paper_id": "p0", "questio')  # torn
    present = cc.present_cells(run_dir, ["flat"])
    assert present == {("flat", "p0", "q0", 0)}
