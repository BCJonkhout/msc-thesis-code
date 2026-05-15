"""Does the architecture ranking stay stable across answerer models?

Computes Kendall's tau-b between every pair of candidates, where each
candidate's "ranking" is its four architectures ordered by NovelQA
local-consensus accuracy (best -> worst). High pairwise tau means
architecture rank is answerer-independent; low or negative tau means
the choice of architecture depends on which answerer you pair it with.

The pre-registered decision threshold from Phase F.1 (pilot_findings)
is tau >= 0.67. The thesis Section 6 caveat for the cross-model
robustness check is triggered when any pair falls below it.

Run from ``code/``::

    .venv/Scripts/python.exe scripts/kendall_arch_stability.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from pilot.cli.phase_f_kendall import _kendall_tau, _rank_by_f1


_THRESHOLD = 0.67


def main() -> int:
    data = json.loads(
        Path("outputs/sanity/novelqa_local_scores.json").read_text(encoding="utf-8")
    )
    per_cand = data["per_candidate"]

    # Build per-candidate (arch -> accuracy) for full-grid candidates.
    # A "full-grid" candidate has accuracy_vs_consensus for all 4 archs.
    archs = ["flat", "naive_rag", "raptor", "graphrag"]
    rankings: dict[str, list[str]] = {}
    score_table: dict[str, dict[str, float]] = {}
    for cand, by_arch in per_cand.items():
        scores = {}
        for a in archs:
            v = by_arch.get(a, {}).get("accuracy_vs_consensus")
            if isinstance(v, (int, float)):
                scores[a] = v
        if len(scores) < 4:
            continue  # need all 4 archs to rank meaningfully
        rankings[cand] = _rank_by_f1(scores)
        score_table[cand] = scores

    # Pairwise Kendall tau-b across all candidate pairs.
    pairs = list(combinations(sorted(rankings.keys()), 2))
    taus = []
    print()
    print("Per-candidate architecture ranking (best -> worst):")
    print()
    for cand in sorted(rankings.keys()):
        ranking = rankings[cand]
        scs = score_table[cand]
        marked = " > ".join(f"{a}({scs[a]:.3f})" for a in ranking)
        print(f"  {cand:<46} {marked}")
    print()
    print(f"Pairwise Kendall tau-b across {len(pairs)} candidate pairs")
    print(f"  (threshold for STABLE architecture rank: tau >= {_THRESHOLD})")
    print()
    print(f"{'Candidate A':<46} {'Candidate B':<46} {'tau':>7}  decision")
    print("-" * 116)
    n_stable = 0
    n_unstable = 0
    pair_results = []
    for a, b in pairs:
        tau, conc, disc = _kendall_tau(rankings[a], rankings[b])
        stable = tau >= _THRESHOLD
        if stable:
            n_stable += 1
        else:
            n_unstable += 1
        pair_results.append({
            "a": a, "b": b, "tau": round(tau, 4),
            "concordant": conc, "discordant": disc,
            "stable": stable,
        })
        print(
            f"{a:<46} {b:<46} {tau:>7.3f}  "
            f"{'STABLE' if stable else 'RANK_DEPENDS_ON_ANSWERER'}"
        )
    print()
    print(f"Summary: {n_stable} of {len(pairs)} pairs STABLE  "
          f"({100*n_stable/len(pairs):.0f}%)")
    print()
    if n_stable >= 0.8 * len(pairs):
        verdict = (
            "YES — architecture ranking is largely answerer-independent. "
            "The thesis can claim architecture matters more than answerer "
            "for ranking purposes."
        )
    elif n_stable <= 0.4 * len(pairs):
        verdict = (
            "NO — architecture ranking flips across answerers. The "
            "thesis must claim architecture and answerer choice "
            "INTERACT; you cannot recommend one architecture in isolation."
        )
    else:
        verdict = (
            "MIXED — about half the candidate pairs agree on architecture "
            "ranking, the other half do not. The thesis should report "
            "this as a 'partial rank stability' finding."
        )
    print(f"Verdict: {verdict}")

    # Dump JSON.
    out_path = Path("outputs/sanity/novelqa_kendall_arch_stability.json")
    out_path.write_text(json.dumps({
        "schema_version": 1,
        "threshold": _THRESHOLD,
        "rankings": rankings,
        "score_table": score_table,
        "pairs": pair_results,
        "summary": {
            "n_pairs": len(pairs),
            "n_stable": n_stable,
            "n_unstable": n_unstable,
            "stable_fraction": round(n_stable / len(pairs), 4),
            "verdict": verdict,
        },
    }, indent=2), encoding="utf-8")
    print(f"[kendall_arch] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
