"""Break-even re-derivation under the alternative Flat cost treatment.

The main analysis (breakeven_main_study.py) charges Flat a density-independent
per-query cost: the realized mean of its uncached first reads and cached
re-reads over the observed questions per document. The Threats section names
the alternative treatment -- charge the uncached first-read *premium* of each
document as a one-time build cost, so Flat gets its own amortizing curve --
and defers the re-derivation. This script performs it.

Treatment (per price card):
  C_off^flat(doc)  = T_doc * (p_uncached - p_cached)
      where T_doc is the document's full-context token count, measured as the
      max uncached_input_tokens over that document's run-index-0 generate rows
      (the first read carries the whole document uncached).
  C_on^flat        = realized C_on - sum_doc C_off^flat(doc)   (residual)
      i.e. every other cost stays per-query, including re-reads that missed
      the cache in the observed runs. The decomposition conserves the realized
      total exactly, so the two treatments coincide at the observed density by
      construction and the reported deployment costs are unchanged.

Break-even vs a structured architecture a then solves
  C_off^a/N + c_on^a  =  C_off^flat/N + c_on^flat_alt
  =>  N* = (C_off^a - C_off^flat) / (c_on^flat_alt - c_on^a)   [per document]

The arch-side terms are read from breakeven_by_dataset.json so both
treatments use identical attribution. The script also reports the realized
cache-hit structure of Flat's re-reads, because the alternative treatment's
premise -- one uncached read per document, cached re-reads afterwards -- is an
idealization the ledger can contradict (cache TTL expiry produces uncached
re-reads that no build term can absorb).

Output: outputs/main_study/breakeven_flat_build_treatment.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breakeven_main_study import iter_ledger  # noqa: E402

OUT = ROOT / "outputs" / "main_study"
DATASETS = ["qasper", "novelqa"]

# gemini-3.1-flash-lite-preview rates, USD/token (configs/price_card.yaml,
# provider:ai.google.dev 2026-04); the cards differ only in the cached rate.
P_UNCACHED = 0.10e-6
P_CACHED = {"base": 0.025e-6, "cache": 0.01e-6}


def main() -> int:
    # ── Included pool (same as the quality/cost denominator) ──────────────────
    incl_nov, q_keys, doc_keys = set(), set(), set()
    for line in open(OUT / "scored_cells.jsonl", encoding="utf-8"):
        o = json.loads(line)
        ds, cl, qid = o["dataset"], o["cluster"], o["qid"]
        q_keys.add((ds, cl, qid)); doc_keys.add((ds, cl))
        if ds == "novelqa":
            incl_nov.add(cl)
    N_Q_DS = Counter(ds for (ds, _cl, _qid) in q_keys)
    N_DOCS_DS = Counter(ds for (ds, _cl) in doc_keys)

    def included(ds: str, pid: str) -> bool:
        return not (ds == "novelqa" and pid not in incl_nov)

    rd = glob.glob(str(ROOT / "outputs" / "runs" / "main-full-*"))[0]
    flat_docs = [(o["dataset"], o["paper_id"])
                 for o in iter_ledger(f"{rd}/flat_predictions.jsonl")
                 if o.get("run_index") == 0]

    # ── Flat generate rows, attributed to documents (same order-walk) ─────────
    tok = {ds: [0, 0, 0] for ds in DATASETS}       # [uncached, cached, output]
    doc_tokens = defaultdict(int)                   # pid -> full-context tokens
    doc_ds = {}                                     # pid -> dataset
    doc_rows = Counter()                            # pid -> n queries (run 0)
    doc_first_unc = {}                              # pid -> first row's uncached
    genptr = 0
    for o in iter_ledger(f"{rd}/ledger.jsonl"):
        if o.get("run_index") != 0 or o.get("architecture") != "flat":
            continue
        if o.get("stage") != "generate":
            continue
        ds, pid = flat_docs[genptr]; genptr += 1
        if not included(ds, pid):
            continue
        u = o.get("uncached_input_tokens", 0) or 0
        c = o.get("cached_input_tokens", 0) or 0
        t = o.get("output_tokens", 0) or 0
        tok[ds][0] += u; tok[ds][1] += c; tok[ds][2] += t
        doc_tokens[pid] = max(doc_tokens[pid], u)
        doc_ds[pid] = ds
        doc_rows[pid] += 1
        doc_first_unc.setdefault(pid, u)

    # ── Realized cache structure: how far off is the idealized premise? ───────
    cache_report = {}
    for ds in DATASETS:
        pids = [p for p, d in doc_ds.items() if d == ds]
        first_reads = sum(doc_tokens[p] for p in pids)
        u, c, _t = tok[ds]
        rereads_unc = u - first_reads       # uncached tokens beyond one full read/doc
        # tokens an ideal cache would have served cached on re-reads
        ideal_rereads = sum(doc_tokens[p] * (doc_rows[p] - 1) for p in pids)
        cache_report[ds] = {
            "uncached_tokens": u, "cached_tokens": c,
            "first_read_tokens": first_reads,
            "rereads_uncached_tokens": rereads_unc,
            "ideal_reread_tokens": ideal_rereads,
            "reread_cache_hit_rate": round(c / ideal_rereads, 4) if ideal_rereads else None,
        }

    # ── Arch-side terms + realized Flat marginal from the published artifacts ──
    be = json.loads((OUT / "breakeven_by_dataset.json").read_text(encoding="utf-8"))
    cost_ds = json.loads((OUT / "cost_by_dataset.json").read_text(encoding="utf-8"))

    out = {}
    for card in ("base", "cache"):
        prem = P_UNCACHED - P_CACHED[card]
        for ds in DATASETS:
            nq, ndocs = N_Q_DS[ds], N_DOCS_DS[ds]
            build_total = sum(doc_tokens[p] * prem for p, d in doc_ds.items() if d == ds)
            coff_flat_doc = build_total / ndocs
            flat_on_total = cost_ds[f"{card}|flat|{ds}"]["c_on_total"]
            flat_onq_real = cost_ds[f"{card}|flat|{ds}"]["c_on_per_query"]
            flat_onq_alt = (flat_on_total - build_total) / nq
            out[f"{card}|flat|{ds}"] = {
                "dataset": ds, "treatment": "first-read premium as build",
                "c_off_per_doc": round(coff_flat_doc, 6),
                "c_on_per_query_realized": flat_onq_real,
                "c_on_per_query_alt": round(flat_onq_alt, 6),
                "premium_usd_per_tok": prem,
                "cache_structure": cache_report[ds],
            }
            for a in ("naive_rag", "raptor", "graphrag"):
                b = be[f"{card}|{a}|{ds}"]
                num = b["c_off_per_doc"] - coff_flat_doc
                den = flat_onq_alt - b["c_on_per_query"]
                if den > 0 and num > 0:
                    nstar = num / den
                    verdict = f"break-even at N*={nstar:.1f} q/doc"
                elif den > 0:
                    nstar = 0.0
                    verdict = "cheaper than flat at any density (lower build AND lower per-query)"
                else:
                    nstar = None
                    verdict = "never cheaper than flat (per-query cost exceeds flat's marginal)"
                out[f"{card}|{a}|{ds}"] = {
                    "dataset": ds,
                    "c_off_per_doc": b["c_off_per_doc"],
                    "flat_c_off_per_doc": round(coff_flat_doc, 6),
                    "c_on_per_query": b["c_on_per_query"],
                    "flat_c_on_per_query_alt": round(flat_onq_alt, 6),
                    "n_star_alt": nstar,
                    "n_star_realized_treatment": b["n_star"],
                    "density": b["density"],
                    "verdict": verdict,
                }

    # conservation guard: build + alt-marginal * nq == realized flat C_on
    for card in ("base", "cache"):
        for ds in DATASETS:
            r = out[f"{card}|flat|{ds}"]
            lhs = r["c_off_per_doc"] * N_DOCS_DS[ds] + r["c_on_per_query_alt"] * N_Q_DS[ds]
            rhs = cost_ds[f"{card}|flat|{ds}"]["c_on_total"]
            assert abs(lhs - rhs) < 0.01, (card, ds, lhs, rhs)

    print("FLAT CACHE STRUCTURE (run-index-0 generate rows, included pool):")
    for ds in DATASETS:
        cr = cache_report[ds]
        print(f"  {ds}: uncached {cr['uncached_tokens']/1e6:.1f}M  cached {cr['cached_tokens']/1e6:.1f}M  "
              f"first-reads {cr['first_read_tokens']/1e6:.1f}M  "
              f"uncached-rereads {cr['rereads_uncached_tokens']/1e6:.1f}M  "
              f"re-read cache-hit rate {cr['reread_cache_hit_rate']}")

    for card in ("base", "cache"):
        print(f"\nALTERNATIVE-TREATMENT BREAK-EVEN ({card} card):")
        for ds in DATASETS:
            f = out[f"{card}|flat|{ds}"]
            print(f"  {ds}: flat C_off/doc {f['c_off_per_doc']*1000:.3f}m  "
                  f"C_on/q {f['c_on_per_query_realized']*1000:.3f}m -> {f['c_on_per_query_alt']*1000:.3f}m")
            for a in ("naive_rag", "raptor", "graphrag"):
                r = out[f"{card}|{a}|{ds}"]
                old = r["n_star_realized_treatment"]
                olds = f"{old:.1f}" if old is not None else "never"
                news = f"{r['n_star_alt']:.1f}" if r["n_star_alt"] is not None else "never"
                print(f"      {a:10s} N* {olds} -> {news}   ({r['verdict']})")

    (OUT / "breakeven_flat_build_treatment.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'breakeven_flat_build_treatment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
