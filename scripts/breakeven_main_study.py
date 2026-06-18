"""Main-study cost, break-even, and study-wide amortized cost, reconciled to the
SAME post-exclusion evaluation pool the quality numbers use.

Why reconciliation: the raw run ledger covers every answered cell (2,449 q / 309
docs incl. 60 NovelQA novels). Quality, however, is scored on the post-exclusion
test pool (2,303 q / 303 docs incl. 54 novels): the calibration / held-out novels
are dropped so calibration data does not leak into the metric. Cost is therefore
attributed per document/cell and restricted to the SAME pool, so the cost and
quality sections agree. The excluded novels carried disproportionate build cost
(one long novel, B42, alone accounted for ~$5.6 of GraphRAG's extraction), so the
reconciliation is material, not cosmetic.

Attribution: the ledger has no per-row document id, but predictions are written
in execution order, so the i-th run-index-0 generate row of an architecture maps
to the i-th run-index-0 prediction (validated: counts match, token magnitudes
match QASPER vs novel). Builds are interleaved per document (a `preprocess` burst
precedes that document's queries), so each preprocess row is charged to the
document of the next generate row. The included/excluded split comes from
scored_cells.jsonl (the quality pool); QASPER has no exclusions.

Cost components: C_off = "preprocess" (one-time build); C_on = "retrieval" +
"generate" (per query). Gemini priced via the price card (both standard and
cache-discount cards); bge-m3 embedding billed at the local RTX 4070 GPU rate
using the locked idle-server rate below. Storage C_store is the per-architecture
persistent-artifact footprint over the locked study horizon; the study-wide
amortized cost C_study folds it in.

Outputs: code/outputs/main_study/cost_per_arch.json, breakeven.json.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot.price_card import load_price_card, _row_cost_usd
from pilot.ledger import CallRecord

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
OUT = ROOT / "outputs" / "main_study"
MS = OUT
OUT.mkdir(parents=True, exist_ok=True)

# bge-m3 idle-Ollama embed rate, measured once on the experiment host (RTX 4070,
# single-flight idle server) and locked; see the run's embedding_cost_calibration
# and the previously published cost_per_arch.embed_rate. Reused here for a
# deterministic recompute that does not depend on a live Ollama server.
EMBED_FIXED_S_PER_CALL = 0.17121772848101266
EMBED_MARGINAL_S_PER_CHAR = 3.46486075949368e-06

# Main-study preprocess-cache config signatures (to pick the right variant dir
# when a document was also built under a different pilot config).
MAIN_CONFIG = {
    "graphrag": lambda ck: ck.get("top_k_entities") == 10 and ck.get("community_prop") == 0.15,
    "raptor": lambda ck: ck.get("max_layers") == 5 and ck.get("chunk_size_tokens") == 100,
}

_dec = json.JSONDecoder()


def iter_ledger(path: str):
    """Yield dict rows from a JSONL/concatenated-JSON ledger robustly."""
    for raw in open(path, encoding="utf-8"):
        s = raw.strip(); pos = 0
        while pos < len(s):
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                break
            try:
                o, end = _dec.raw_decode(s, pos)
            except json.JSONDecodeError:
                break
            pos = end
            if isinstance(o, dict):
                yield o


def _rec(o: dict) -> CallRecord | None:
    try:
        return CallRecord(**{k: o.get(k) for k in CallRecord.__dataclass_fields__})
    except TypeError:
        return None


def artifact_bytes(arch: str, included_docs: set[str]) -> tuple[int, dict]:
    """Sum the main-config artifact.pkl bytes over included docs; also return
    per-document sizes split by dataset prefix (novel ids start with 'B')."""
    base = ROOT / "outputs" / "preprocess_cache" / arch
    match = MAIN_CONFIG[arch]
    total = 0
    per_doc = {"novelqa": [], "qasper": []}
    if not base.exists():
        return 0, per_doc
    for doc_dir in base.iterdir():
        if not doc_dir.is_dir() or doc_dir.name not in included_docs:
            continue
        for var in doc_dir.iterdir():
            meta = var / "build_meta.json"
            art = var / "artifact.pkl"
            if not (meta.exists() and art.exists()):
                continue
            try:
                ck = json.loads(meta.read_text(encoding="utf-8")).get("cache_key_inputs", {})
            except json.JSONDecodeError:
                continue
            if match(ck):
                sz = art.stat().st_size
                total += sz
                ds = "novelqa" if ck.get("dataset") == "novelqa" else "qasper"
                per_doc[ds].append(sz)
                break
    return total, per_doc


def main() -> int:
    pc_base = load_price_card(ROOT / "configs" / "price_card.yaml")
    cache_path = ROOT / "configs" / "price_card_cache_discount.yaml"
    pc_cache = load_price_card(cache_path) if cache_path.exists() else pc_base
    gpu_rate = (pc_base.get("gpu") or {}).get("usd_per_second") or 0.0
    store = (pc_base.get("storage") or {})
    store_rate = (store.get("rate_usd_per_gib_month") or {}).get("value") or 0.0
    horizon_days = (store.get("study_horizon_days") or {}).get("value") or 0.0
    store_factor = store_rate * (horizon_days / 30.0)  # USD per GiB over the horizon
    GIB = 1024 ** 3

    # ── Evaluation (quality) pool from scored_cells: the cost denominator ──────
    incl_nov, q_keys, doc_keys = set(), set(), set()
    for line in open(MS / "scored_cells.jsonl", encoding="utf-8"):
        o = json.loads(line)
        ds, cl, qid = o["dataset"], o["cluster"], o["qid"]
        q_keys.add((ds, cl, qid)); doc_keys.add((ds, cl))
        if ds == "novelqa":
            incl_nov.add(cl)
    N_Q = len(q_keys)
    N_DOCS = len(doc_keys)

    def included(ds: str, pid: str) -> bool:
        return not (ds == "novelqa" and pid not in incl_nov)

    rd = glob.glob(str(ROOT / "outputs" / "runs" / "main-full-*"))[0]
    # per-arch ordered (dataset, doc) for run-index-0 generate rows
    docs = {}
    for a in ARCHS:
        docs[a] = [(o["dataset"], o["paper_id"])
                   for o in iter_ledger(f"{rd}/{a}_predictions.jsonl")
                   if o.get("run_index") == 0]

    # ── Walk the ledger once, attributing every run-index-0 row to a document ──
    gem = {"base": defaultdict(float), "cache": defaultdict(float)}  # (arch, off|on) -> usd, included only
    emb = defaultdict(lambda: [0, 0])  # (arch, off|on) -> [calls, chars], included only
    billed_gem = 0.0  # all gemini rows, all repeats, full grid (billing reconciliation)
    doc_tokens = defaultdict(int)  # included doc id -> full-context tokens (flat first-query uncached)
    genptr = Counter()
    pend_off_gem = {a: [] for a in ARCHS}
    pend_off_emb = {a: [0, 0] for a in ARCHS}
    pend_on_emb = {a: [0, 0] for a in ARCHS}

    for o in iter_ledger(f"{rd}/ledger.jsonl"):
        m = str(o.get("model", "") or "")
        if m.startswith("gemini"):
            r = _rec(o)
            if r is not None:
                billed_gem += _row_cost_usd(r, pc_base)
        if o.get("run_index") != 0:
            continue
        a, st = o.get("architecture"), o.get("stage")
        if a not in docs:
            continue
        isg = m.startswith("gemini")
        isbge = m == "bge-m3"
        if st == "generate":
            ds, pid = docs[a][genptr[a]]; genptr[a] += 1
            inc = included(ds, pid)
            # flush pending build (off) and query-embed (on) to this document
            if inc:
                for o2 in pend_off_gem[a]:
                    r = _rec(o2)
                    if r is not None:
                        gem["base"][(a, "off")] += _row_cost_usd(r, pc_base)
                        gem["cache"][(a, "off")] += _row_cost_usd(r, pc_cache)
                emb[(a, "off")][0] += pend_off_emb[a][0]; emb[(a, "off")][1] += pend_off_emb[a][1]
                emb[(a, "on")][0] += pend_on_emb[a][0]; emb[(a, "on")][1] += pend_on_emb[a][1]
            pend_off_gem[a] = []; pend_off_emb[a] = [0, 0]; pend_on_emb[a] = [0, 0]
            if isg and inc:
                r = _rec(o)
                if r is not None:
                    gem["base"][(a, "on")] += _row_cost_usd(r, pc_base)
                    gem["cache"][(a, "on")] += _row_cost_usd(r, pc_cache)
            if a == "flat" and inc:
                doc_tokens[pid] = max(doc_tokens[pid], o.get("uncached_input_tokens", 0) or 0)
        elif st == "preprocess":
            if isg:
                pend_off_gem[a].append(o)
            elif isbge:
                pend_off_emb[a][0] += 1
                pend_off_emb[a][1] += (o.get("uncached_input_tokens", 0) or 0) * 4
        elif st == "retrieval":
            if isbge:
                pend_on_emb[a][0] += 1
                pend_on_emb[a][1] += (o.get("uncached_input_tokens", 0) or 0) * 4

    def embed_usd(a: str, bucket: str) -> float:
        calls, chars = emb[(a, bucket)]
        return (calls * EMBED_FIXED_S_PER_CALL + chars * EMBED_MARGINAL_S_PER_CHAR) * gpu_rate

    # ── Persistent-artifact storage per architecture (over the included pool) ──
    raw_text_bytes = sum(t * 4 for t in doc_tokens.values())          # flat: source text
    naive_bytes = raw_text_bytes + sum((t / 384) * 1024 * 4 for t in doc_tokens.values())  # chunks + bge-m3 vectors
    # preprocess-cache dirs name QASPER docs with underscores (1503_00841), not dots
    incl_cache = incl_nov | {c.replace(".", "_") for d, c in doc_keys if d == "qasper"}
    gr_bytes, gr_doc = artifact_bytes("graphrag", incl_cache)
    rp_bytes, rp_doc = artifact_bytes("raptor", incl_cache)
    store_bytes = {"flat": raw_text_bytes, "naive_rag": int(naive_bytes), "raptor": rp_bytes, "graphrag": gr_bytes}
    c_store = {a: store_bytes[a] / GIB * store_factor for a in ARCHS}

    # ── Per-architecture cost, both cards ─────────────────────────────────────
    per_arch = {}
    for card in ("base", "cache"):
        for a in ARCHS:
            g_off, g_on = gem[card][(a, "off")], gem[card][(a, "on")]
            e_off, e_on = embed_usd(a, "off"), embed_usd(a, "on")
            c_off, c_on = g_off + e_off, g_on + e_on
            per_arch[(card, a)] = {
                "c_off_total": round(c_off, 4), "c_on_total": round(c_on, 4),
                "total": round(c_off + c_on, 4),
                "gemini_off": round(g_off, 4), "embed_off": round(e_off, 4),
                "gemini_on": round(g_on, 4), "embed_on": round(e_on, 4),
                "c_on_per_query": round(c_on / N_Q, 6),
                "c_store_total": round(c_store[a], 6),
                # study-wide amortized cost per query (build + storage + online) / N_Q
                "c_study_per_query": round((c_off + c_store[a] + c_on) / N_Q, 6),
            }

    n_nov = sum(1 for d, _c in doc_keys if d == "novelqa")
    n_pap = sum(1 for d, _c in doc_keys if d == "qasper")
    print(f"reconciled pool: {N_Q} questions / {N_DOCS} docs ({n_nov} novels + {n_pap} papers)")
    print("PER-ARCH (base card):")
    for a in ARCHS:
        d = per_arch[("base", a)]
        print(f"  {a:10s} C_off ${d['c_off_total']:7.3f} (gem {d['gemini_off']:.2f}+emb {d['embed_off']:.2f})  "
              f"C_on ${d['c_on_total']:7.3f}  TOTAL ${d['total']:7.3f}  "
              f"C_on/q {d['c_on_per_query']*1000:.3f}m  C_store ${d['c_store_total']:.4f}  "
              f"C_study/q {d['c_study_per_query']*1000:.3f}m")

    # ── Break-even N vs flat (per-document build amortized over N queries) ─────
    breakeven = {}
    for card in ("base", "cache"):
        flat_onq = per_arch[(card, "flat")]["c_on_per_query"]
        for a in ARCHS:
            if a == "flat":
                continue
            coff_doc = per_arch[(card, a)]["c_off_total"] / N_DOCS
            onq = per_arch[(card, a)]["c_on_per_query"]
            curve = {N: round(coff_doc / N + onq, 6) for N in (1, 2, 5, 10, 25)}
            if flat_onq > onq:
                nstar = coff_doc / (flat_onq - onq)
                verdict = f"break-even at N*={nstar:.1f} q/doc (cheaper than flat above it)"
            else:
                nstar = None
                verdict = "never cheaper than flat (flat lower per-query too)"
            breakeven[(card, a)] = {"c_off_per_doc": round(coff_doc, 6), "c_on_per_query": onq,
                                    "flat_c_on_per_query": flat_onq, "n_star": nstar,
                                    "amortized_per_query_by_N": curve, "verdict": verdict}

    print(f"\nBREAK-EVEN vs flat (flat C_on/q ${per_arch[('base','flat')]['c_on_per_query']*1000:.3f}m), base:")
    for a in ("naive_rag", "raptor", "graphrag"):
        b = breakeven[("base", a)]
        print(f"  {a:10s} C_off/doc ${b['c_off_per_doc']*1000:.3f}m -> {b['verdict']}")

    deploy_total = {c: round(sum(per_arch[(c, a)]["total"] for a in ARCHS), 2) for c in ("base", "cache")}
    print(f"\ndeployment total: base ${deploy_total['base']} cache ${deploy_total['cache']} | billed gemini ${billed_gem:.2f}")

    def mean_mb(sizes):
        return round(sum(sizes) / len(sizes) / 1e6, 2) if sizes else None
    storage_report = {
        "flat": {"per_doc_mb_mean": round(raw_text_bytes / max(1, len(doc_tokens)) / 1e6, 3), "kind": "raw source text"},
        "naive_rag": {"per_doc_mb_mean": round(naive_bytes / max(1, len(doc_tokens)) / 1e6, 2), "kind": "chunks + bge-m3 vectors (estimated)"},
        "raptor": {"per_doc_mb_mean_novel": mean_mb(rp_doc["novelqa"]), "per_doc_mb_mean_paper": mean_mb(rp_doc["qasper"]), "kind": "tree (leaves+summaries+embeddings)"},
        "graphrag": {"per_doc_mb_mean_novel": mean_mb(gr_doc["novelqa"]), "per_doc_mb_mean_paper": mean_mb(gr_doc["qasper"]), "kind": "entity graph + community reports + embeddings"},
    }
    print("\nstorage (measured artifacts):")
    for a in ARCHS:
        print(f"  {a:10s} total ${c_store[a]:.4f} over {horizon_days:.0f}d  | {storage_report[a]}")

    def keyed(d):
        return {f"{c}|{a}": v for (c, a), v in d.items()}
    (OUT / "cost_per_arch.json").write_text(json.dumps({
        "embed_rate": {"fixed_s_per_call": EMBED_FIXED_S_PER_CALL, "marginal_s_per_char": EMBED_MARGINAL_S_PER_CHAR,
                       "gpu_usd_per_second": gpu_rate},
        "storage_rate": {"usd_per_gib_month": store_rate, "horizon_days": horizon_days,
                         "store_bytes_per_arch": store_bytes, "report": storage_report},
        "per_arch": keyed(per_arch), "deployment_total": deploy_total,
        "study_total": round(sum(per_arch[("base", a)]["total"] + per_arch[("base", a)]["c_store_total"] for a in ARCHS), 4),
        "billed_gemini": round(billed_gem, 2), "n_questions": N_Q, "n_docs": N_DOCS,
    }, indent=2), encoding="utf-8")
    (OUT / "breakeven.json").write_text(json.dumps(keyed(breakeven), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'cost_per_arch.json'} + {OUT/'breakeven.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
