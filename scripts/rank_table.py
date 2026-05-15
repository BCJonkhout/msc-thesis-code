"""Build the rank table: cost (NovelQA full grid) + quality (QASPER F1).

NovelQA quality is leaderboard-held (Codabench) so we report it as
'pending'. QASPER F1 is computed locally and was recorded by prior
Phase F.1 sweeps; we pick the most-recent verdict per candidate that
has all four architectures scored.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/rank_table.py
"""
from __future__ import annotations

import json
from pathlib import Path


def _latest_qasper_verdict(model: str) -> dict | None:
    """Return the most-recent QASPER verdict json for ``model`` that has
    all four architectures scored, or None if no qualifying verdict
    exists."""
    sanity = Path("outputs/sanity")
    candidates = []
    for vp in sorted(sanity.glob("step_3_dry_run_*.json"), reverse=True):
        try:
            v = json.loads(vp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if v.get("answerer_model") != model:
            continue
        if "qasper" not in (v.get("datasets") or []):
            continue
        macro = v.get("per_arch_macro") or {}
        ok = all(
            isinstance(macro.get(a, {}).get("answer_f1"), (int, float))
            for a in ("flat", "naive_rag", "raptor", "graphrag")
        )
        candidates.append((ok, vp.name, v))
    # Prefer a 4-of-4 verdict; fall back to the most-recent any
    for ok, _, v in candidates:
        if ok:
            return v
    return candidates[0][2] if candidates else None


def main() -> int:
    novelqa_summary = json.loads(
        Path("outputs/sanity/novelqa_grid_summary.json").read_text(encoding="utf-8")
    )
    # Build cost lookup keyed by answerer model name (strip the
    # "g_novelqa-" prefix from the label).
    cost_by_model: dict[str, dict] = {}
    for row in novelqa_summary["rows"]:
        if "error" in row:
            continue
        cand = row["candidate"].replace("g_novelqa-", "")
        # Some Openrouter candidates carry the "deepseek/" slash that the
        # provider uses but the label flattens to a dash-less form. Try
        # both spellings against the verdict's answerer_model.
        cost_by_model[cand] = row
        cost_by_model[f"deepseek/{cand}"] = row  # for openrouter
    # Map candidate label to clean model name (full-grid only).
    full_grid = [
        ("gemini-flash-latest", "gemini-flash-latest"),
        ("deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
        ("grok-4.20-0309-non-reasoning", "grok-4.20-0309-non-reasoning"),
        ("grok-4-fast-reasoning", "grok-4-fast-reasoning"),
        ("grok-4.20-0309-reasoning", "grok-4.20-0309-reasoning"),
        ("grok-4.3", "grok-4.3"),
        # resumed; mark cost as partial-comparable
        ("deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
        ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview"),
        ("grok-4-1-fast-non-reasoning", "grok-4-1-fast-non-reasoning"),
    ]

    rows = []
    for cand_label, model_name in full_grid:
        cost = cost_by_model.get(cand_label) or cost_by_model.get(model_name)
        if cost is None:
            print(f"missing cost for {cand_label}")
            continue
        verdict = _latest_qasper_verdict(model_name)
        if verdict is None:
            macro = {}
        else:
            macro = verdict.get("per_arch_macro") or {}
        rows.append({
            "candidate": cand_label,
            "novelqa_total_usd": cost["total_usd"],
            "novelqa_preprocess_usd": cost["per_stage_usd"].get("preprocess", 0.0),
            "novelqa_generate_usd": cost["per_stage_usd"].get("generate", 0.0),
            "novelqa_arch_counts": cost["arch_counts"],
            "qasper_flat_f1": macro.get("flat", {}).get("answer_f1"),
            "qasper_naive_f1": macro.get("naive_rag", {}).get("answer_f1"),
            "qasper_raptor_f1": macro.get("raptor", {}).get("answer_f1"),
            "qasper_graphrag_f1": macro.get("graphrag", {}).get("answer_f1"),
            "qasper_macro_mean": (
                round(
                    sum(
                        macro.get(a, {}).get("answer_f1") or 0.0
                        for a in ("flat", "naive_rag", "raptor", "graphrag")
                    ) / 4, 4)
                if macro else None
            ),
        })

    # Sort by NovelQA total cost ascending.
    rows.sort(key=lambda r: r["novelqa_total_usd"])

    print(f"{'#':<2} {'Candidate':<35} {'NovelQA $':>9} {'QASPER macro F1 (flat / naive / raptor / graphrag) — mean'}")
    print("-" * 130)
    for i, r in enumerate(rows, 1):
        f1s = []
        for k in ("qasper_flat_f1", "qasper_naive_f1", "qasper_raptor_f1", "qasper_graphrag_f1"):
            v = r[k]
            f1s.append(f"{v:.3f}" if isinstance(v, (int, float)) else "  -  ")
        mean = r["qasper_macro_mean"]
        mean_s = f"{mean:.3f}" if isinstance(mean, (int, float)) else " - "
        print(
            f"{i:<2} {r['candidate']:<35} {r['novelqa_total_usd']:>9.2f}   "
            f"{f1s[0]} / {f1s[1]} / {f1s[2]} / {f1s[3]}   ->  mean {mean_s}"
        )
    print()
    print("NovelQA accuracy: leaderboard-held (Codabench); not yet computed.")
    print("QASPER F1 source: latest per-candidate Phase F.1 verdict in outputs/sanity/.")

    out_path = Path("outputs/sanity/rank_table.json")
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[rank_table] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
