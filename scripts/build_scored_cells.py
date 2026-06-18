"""Foundation for the main-study analysis: one scored cell per (dataset, cluster,
question, architecture).

- QASPER: per-question Answer-F1, meaned over the 5 repeats (local gold on disk).
- NovelQA: per-question correctness (0/1) recovered from the Codabench scoring
  logs of the run_index-0 submissions. The 5 repeats are 100%/99.7% identical at
  T=0 (verified), so run_index 0 is the 5-repeat value; n_repeats=5 records that.

Cluster unit = paper_id (QASPER) / novel_id (NovelQA) for the clustered bootstrap.

Output: code/outputs/main_study/scored_cells.jsonl  (+ a headline summary on stdout).
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pilot.env import load_env
from pilot.codabench.extract_score import fetch_correctness_strings, _norm_title

# arch -> Codabench submission id. flat/graphrag verified; the other two are the
# two parallel ids {799673, 799674}, disambiguated below by their known accuracy
# (naive_rag 0.6624 > raptor 0.6604).
KNOWN = {"flat": 799662, "graphrag": 799676}
AMBIGUOUS = [799673, 799674]
ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]


def main() -> int:
    load_env()
    data = Path(__file__).resolve().parents[1] / "data"
    rd = glob.glob(str(Path(__file__).resolve().parents[1] / "outputs" / "runs" / "main-full-*"))[0]
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "main_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calibration novels are held out of the main-evaluation metric: they were
    # used to tune the pilot, so scoring on them would leak calibration data into
    # the test number. One of them (B48) is additionally degenerate in the run
    # (identical correctness across all four architectures), confirming it is not
    # an independent per-architecture measurement. Honour exclude_from_main_eval.
    excl = json.loads((data / "novelqa" / "calibration_novels.json").read_text(
        encoding="utf-8")).get("exclude_from_main_eval", [])
    EXCLUDE = set(excl)
    print(f"excluding calibration novels from main eval: {sorted(EXCLUDE)}")

    # our public-domain novels + titles, and per-novel QID order
    nids = set()
    qorder: dict[str, list[str]] = defaultdict(list)
    with (data / "novelqa" / "questions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            nids.add(q["novel_id"])
            qorder[q["novel_id"]].append(q["question_id"])
    meta = json.loads((data / "novelqa" / "bookmeta.json").read_text(encoding="utf-8"))
    norm2nid = {_norm_title(meta[n]["title"]): n for n in nids if n in meta and "title" in meta[n]}

    _cache: dict[int, dict] = {}

    def correct_for(sub_id: int) -> dict[tuple[str, str], int]:
        if sub_id in _cache:
            return _cache[sub_id]
        strings = fetch_correctness_strings(sub_id)  # {codabench_key: "TFTF..."}
        out: dict[tuple[str, str], int] = {}
        for norm, s in strings.items():
            # normalise the Codabench key through the same transliterating
            # _norm_title as norm2nid, so accented titles join (Les Mis\'erables).
            nid = norm2nid.get(_norm_title(norm))
            if not nid:
                continue
            qids = qorder.get(nid, [])
            for i, ch in enumerate(s):
                if i < len(qids):
                    out[(nid, qids[i])] = 1 if ch == "T" else 0
        _cache[sub_id] = out
        return out

    def acc(c: dict) -> float:
        return sum(c.values()) / len(c) if c else 0.0

    # disambiguate the two parallel submissions by accuracy
    amb_acc = {sid: acc(correct_for(sid)) for sid in AMBIGUOUS}
    hi = max(amb_acc, key=amb_acc.get)
    lo = min(amb_acc, key=amb_acc.get)
    sub = {**KNOWN, "naive_rag": hi, "raptor": lo}
    print(f"arch -> submission id: {sub}")
    print(f"  (disambiguated by acc: {amb_acc} -> naive_rag={hi}, raptor={lo})")

    nov = {a: correct_for(sid) for a, sid in sub.items()}

    # Surface any of our novels (outside the calibration hold-out) that got NO
    # recoverable Codabench gold, so the exclusion is explicit, not silent.
    matched = {nid for c in nov.values() for (nid, _qid) in c}
    no_gold = sorted(n for n in nids if n not in EXCLUDE and n not in matched)
    if no_gold:
        print(f"WARNING: no Codabench gold recovered for non-calibration novels {no_gold} "
              f"(answered but unscorable; excluded from the eval pool).")

    # QASPER per-(paper,qid)-arch F1 over repeats
    qasper: dict[tuple[str, str], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for a in ARCHS:
        for line in open(f"{rd}/{a}_predictions.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("dataset") != "qasper":
                continue
            f1 = r.get("answer_f1")
            if isinstance(f1, (int, float)):
                qasper[(r["paper_id"], r["question_id"])][a][r["run_index"]] = f1

    n = 0
    with (out_dir / "scored_cells.jsonl").open("w", encoding="utf-8") as fh:
        for (pid, qid), archs in qasper.items():
            for a, byri in archs.items():
                vals = list(byri.values())
                fh.write(json.dumps({
                    "dataset": "qasper", "cluster": pid, "qid": qid, "arch": a,
                    "metric": round(statistics.mean(vals), 6), "n_repeats": len(vals),
                }) + "\n")
                n += 1
        for a in sub:
            for (nid, qid), c in nov[a].items():
                if nid in EXCLUDE:
                    continue
                fh.write(json.dumps({
                    "dataset": "novelqa", "cluster": nid, "qid": qid, "arch": a,
                    "metric": c, "n_repeats": 5,
                }) + "\n")
                n += 1
    print(f"wrote {n} scored cells -> {out_dir / 'scored_cells.jsonl'}\n")

    def acc_eval(c):
        kept = {k: v for k, v in c.items() if k[0] not in EXCLUDE}
        return sum(kept.values()) / len(kept) if kept else 0.0

    print("headline (sanity):")
    for a in ARCHS:
        qf = [statistics.mean(qasper[k][a].values()) for k in qasper if a in qasper[k]]
        print(f"  {a:10s} QASPER mean F1 {statistics.mean(qf):.4f} (n={len(qf)})  |  "
              f"NovelQA acc {acc_eval(nov[a]):.4f} (n={sum(1 for k in nov[a] if k[0] not in EXCLUDE)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
