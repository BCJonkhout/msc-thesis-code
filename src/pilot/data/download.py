"""Dataset download orchestrator.

Acquires QASPER and QuALITY automatically; emits clear instructions
for NovelQA which is manually-gated. Idempotent: re-running with
existing files is a no-op unless ``--force`` is set.

Storage layout follows pilot plan § 2:

  code/data/qasper/{train,dev,test}.jsonl
  code/data/quality/{train,dev}.jsonl
  code/data/novelqa/{full_texts/, questions.jsonl}   (manual)

Run as:

  python -m pilot.data.download                 # all datasets
  python -m pilot.data.download --dataset qasper

Exit codes:
  0  All requested datasets present (downloaded or already on disk).
  1  At least one dataset failed to acquire.
  2  NovelQA needs manual action (informational; not a hard failure
     unless --strict is passed).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from pilot.env import load_env


_DEFAULT_DATA_ROOT = Path("data")


def _project_data_root() -> Path:
    """Default data root: code/data/. The CLI accepts an override."""
    here = Path(__file__).resolve()
    # src/pilot/data/download.py → walk up 3 to reach code/
    code_root = here.parents[3]
    return code_root / "data"


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to path.jsonl.tmp then rename, so partial writes do not corrupt the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(path)


def _splits_present(target_dir: Path, split_names: list[str]) -> bool:
    return all((target_dir / f"{name}.jsonl").exists() for name in split_names)


# ──────────────────────────────────────────────────────────────────────
# QASPER
# ──────────────────────────────────────────────────────────────────────

# Allen AI's S3 release of QASPER. The HuggingFace mirror was a
# script-based loader, which `datasets` v3+ no longer supports —
# pulling raw release tarballs directly is the canonical path.
_QASPER_RELEASES = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}


def download_qasper(data_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Download QASPER train/dev/test splits to data_root/qasper/.

    Pulls the release tarballs from Allen AI's S3 bucket (the
    canonical source) and converts them to JSONL.
    """
    import io
    import tarfile

    target_dir = data_root / "qasper"
    out_names = ["train", "dev", "test"]

    if not force and _splits_present(target_dir, out_names):
        return {
            "dataset": "qasper",
            "status": "already_present",
            "path": str(target_dir),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    # Each release tarball contains a single JSON file with a dict
    # keyed by paper_id mapping to that paper's QA annotations. The
    # canonical JSONL representation is one row per paper (preserving
    # the dict-shaped value as the row body).
    for release_key, url in _QASPER_RELEASES.items():
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = resp.read()
        except Exception as exc:
            return {
                "dataset": "qasper",
                "status": "error",
                "error": f"download {release_key} failed: {exc!r}",
            }

        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    fh = tar.extractfile(member)
                    if fh is None:
                        continue
                    obj = json.loads(fh.read().decode("utf-8"))

                    name_lower = member.name.lower()
                    if "train" in name_lower:
                        split = "train"
                    elif "dev" in name_lower:
                        split = "dev"
                    elif "test" in name_lower:
                        split = "test"
                    else:
                        continue
                    rows = [
                        {"paper_id": pid, **paper}
                        for pid, paper in obj.items()
                    ]
                    _atomic_write_jsonl(target_dir / f"{split}.jsonl", rows)
                    counts[split] = len(rows)
        except Exception as exc:
            return {
                "dataset": "qasper",
                "status": "error",
                "error": f"extract {release_key} failed: {exc!r}",
            }

    missing = [s for s in out_names if s not in counts]
    if missing:
        return {
            "dataset": "qasper",
            "status": "error",
            "error": f"splits missing after extract: {missing}",
        }

    return {
        "dataset": "qasper",
        "status": "downloaded",
        "path": str(target_dir),
        "row_counts": counts,
    }


# ──────────────────────────────────────────────────────────────────────
# QuALITY
# ──────────────────────────────────────────────────────────────────────

# Official QuALITY release archive on the nyu-mll repo. The htmlstripped
# variant is what most reproductions use (HTML markup is removed; raw
# text only, which is what flat full-context expects).
_QUALITY_ARCHIVE_URL = (
    "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/"
    "QuALITY.v1.0.1.zip"
)


def download_quality(data_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Download QuALITY train/dev splits to data_root/quality/.

    QuALITY ships as a zip with one JSONL per split. Test labels are
    not public (held by NYU); only train and dev are usable. Dev is
    the operational evaluation set per pilot plan § 2.3.
    """
    target_dir = data_root / "quality"
    if not force and _splits_present(target_dir, ["train", "dev"]):
        return {
            "dataset": "quality",
            "status": "already_present",
            "path": str(target_dir),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "_release.zip"

    try:
        with urllib.request.urlopen(_QUALITY_ARCHIVE_URL, timeout=60) as resp:
            zip_path.write_bytes(resp.read())
    except Exception as exc:
        return {
            "dataset": "quality",
            "status": "error",
            "error": f"download failed: {exc!r}",
        }

    counts: dict[str, int] = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            for member in members:
                lower = member.lower()
                # Pilot uses the htmlstripped variant: HTML markup
                # replaced by line breaks. The non-stripped variant
                # is also in the zip but contains raw HTML which
                # would inflate the token count on flat full-context.
                if "htmlstripped" not in lower:
                    continue
                if lower.endswith(".train"):
                    out_name = "train.jsonl"
                elif lower.endswith(".dev"):
                    out_name = "dev.jsonl"
                elif lower.endswith(".test"):
                    out_name = "test.jsonl"  # labels-stripped; kept for completeness
                else:
                    continue
                with zf.open(member) as src:
                    raw = src.read().decode("utf-8")
                # str.splitlines() also splits on unicode line
                # separators (U+2028, U+2029, etc.) that legitimately
                # appear inside QuALITY article text. JSONL is
                # \n-delimited; split on \n only.
                rows = [json.loads(line) for line in raw.split("\n") if line.strip()]
                _atomic_write_jsonl(target_dir / out_name, rows)
                counts[out_name.removesuffix(".jsonl")] = len(rows)
    except Exception as exc:
        return {
            "dataset": "quality",
            "status": "error",
            "error": f"extract failed: {exc!r}",
        }
    finally:
        zip_path.unlink(missing_ok=True)

    return {
        "dataset": "quality",
        "status": "downloaded",
        "path": str(target_dir),
        "row_counts": counts,
    }


# ──────────────────────────────────────────────────────────────────────
# NovelQA
# ──────────────────────────────────────────────────────────────────────

def download_novelqa(data_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Acquire NovelQA via HuggingFace `NovelQA/NovelQA`.

    Per arXiv:2403.12766, the dataset is released on the HuggingFace
    dataset `NovelQA/NovelQA` under Apache-2.0. The repo is a flat
    file collection (no Parquet schema) so `load_dataset` does not
    work; we use `snapshot_download` to pull the repo and then
    extract `NovelQA.zip` which contains:

      Books/PublicDomain/B*.txt       — 61 public-domain novel texts
      Data/PublicDomain/B*.json       — questions for those novels
      Books/CopyrightProtected/       — empty (texts withheld)
      Data/CopyrightProtected/B*.json — questions for withheld novels

    The pilot uses only the public-domain subset (61 novels) since
    the copyright-protected texts cannot be sent to the answerer.
    Per pilot plan § 2.2, novel B48 (The History of Rome, 2.58M
    tokens) is excluded from main evaluation as an outlier.

    Output layout:
      data/novelqa/full_texts/B*.txt
      data/novelqa/questions.jsonl   — one row per (novel_id, question_id)
      data/novelqa/bookmeta.json     — verbatim metadata
    """
    import zipfile

    target_dir = data_root / "novelqa"
    questions_path = target_dir / "questions.jsonl"
    full_texts_dir = target_dir / "full_texts"
    if (
        not force
        and questions_path.exists()
        and full_texts_dir.exists()
        and any(full_texts_dir.iterdir())
    ):
        return {
            "dataset": "novelqa",
            "status": "already_present",
            "path": str(target_dir),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    full_texts_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return {
            "dataset": "novelqa",
            "status": "error",
            "error": f"huggingface_hub not installed: {exc}",
        }

    try:
        snapshot_dir = Path(
            snapshot_download(
                repo_id="NovelQA/NovelQA",
                repo_type="dataset",
                allow_patterns=["NovelQA.zip", "bookmeta.json", "README.md"],
            )
        )
    except Exception as exc:
        msg = repr(exc)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            return {
                "dataset": "novelqa",
                "status": "manual_action_required",
                "error": (
                    "NovelQA is a gated HuggingFace dataset. Visit "
                    "https://huggingface.co/datasets/NovelQA/NovelQA "
                    "and click 'Agree and access' on the dataset card "
                    "with the same HF account whose token is in "
                    "HUGGINGFACE_ACCESS_TOKEN, then re-run."
                ),
            }
        return {
            "dataset": "novelqa",
            "status": "error",
            "error": f"snapshot_download failed: {msg}",
        }

    zip_path = snapshot_dir / "NovelQA.zip"
    if not zip_path.exists():
        return {
            "dataset": "novelqa",
            "status": "error",
            "error": f"NovelQA.zip not found in snapshot at {snapshot_dir}",
        }

    bookmeta_src = snapshot_dir / "bookmeta.json"
    if bookmeta_src.exists():
        (target_dir / "bookmeta.json").write_bytes(bookmeta_src.read_bytes())

    novels_count = 0
    questions_total = 0
    all_questions: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Public-domain novel texts → full_texts/B*.txt
            if member.startswith("Books/PublicDomain/") and member.endswith(".txt"):
                novel_id = Path(member).stem
                with zf.open(member) as src:
                    (full_texts_dir / f"{novel_id}.txt").write_bytes(src.read())
                novels_count += 1

            # Public-domain question files → flatten into questions.jsonl
            if member.startswith("Data/PublicDomain/") and member.endswith(".json"):
                novel_id = Path(member).stem
                with zf.open(member) as src:
                    qmap = json.loads(src.read().decode("utf-8"))
                for question_id, q in qmap.items():
                    row = {"novel_id": novel_id, "question_id": question_id, **q}
                    all_questions.append(row)
                    questions_total += 1

    _atomic_write_jsonl(questions_path, all_questions)

    return {
        "dataset": "novelqa",
        "status": "downloaded",
        "path": str(target_dir),
        "novels_count": novels_count,
        "questions_count": questions_total,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

_DOWNLOADERS = {
    "qasper": download_qasper,
    "quality": download_quality,
    "novelqa": download_novelqa,
}


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(
        description="Acquire pilot datasets per pilot_setup_plan.md § 2."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_DOWNLOADERS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_project_data_root(),
        help="Root directory for downloaded datasets. Default: code/data/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files are already present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat NovelQA's manual_action_required as a hard failure.",
    )
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    selected = list(_DOWNLOADERS.keys()) if args.dataset == "all" else [args.dataset]

    results: list[dict[str, Any]] = []
    for name in selected:
        result = _DOWNLOADERS[name](args.data_root, force=args.force)
        results.append(result)
        print(json.dumps(result, indent=2))

    statuses = [r["status"] for r in results]
    if any(s == "error" for s in statuses):
        return 1
    if not args.strict and any(s == "manual_action_required" for s in statuses):
        return 2
    if args.strict and any(s == "manual_action_required" for s in statuses):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
