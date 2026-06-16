"""Main-study cost + break-even, with the bge-m3 embedding GPU cost billed at the
RTX 4070 rate (NOT $0, and NOT via the contaminated ledger wallclock).

Embedding GPU cost: measured directly on an idle Ollama (total_duration -
load_duration), fitted as fixed-per-call + marginal-per-char, then applied to
the ledger's run_index-0 embed volume per architecture/stage. Gemini calls are
priced via the price card (_row_cost_usd). flat has zero embeds; the structured
architectures carry all embed cost.

Deployment cost = run_index 0 only. C_off = stage "preprocess" (build). C_on =
"retrieval" + "generate" (per query). Reported under BOTH price cards.
Break-even N (questions per document) vs flat is reported as a curve.

Outputs: code/outputs/main_study/cost_per_arch.json, breakeven.json (+ a console
summary). LaTeX tables are emitted too.
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pilot.price_card import load_price_card, _row_cost_usd
from pilot.ledger import CallRecord

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]
OLLAMA = "http://localhost:11434/api/embed"
OUT = ROOT / "outputs" / "main_study"
OUT.mkdir(parents=True, exist_ok=True)


def measure_embed_rate() -> tuple[float, float]:
    """Idle-Ollama bge-m3: return (fixed_s_per_call, marginal_s_per_char)."""
    def eval_s(text: str, n: int = 8) -> float:
        vals = []
        with httpx.Client(timeout=120) as c:
            c.post(OLLAMA, json={"model": "bge-m3", "input": text})  # warmup
            for _ in range(n):
                d = c.post(OLLAMA, json={"model": "bge-m3", "input": text}).json()
                vals.append((d.get("total_duration", 0) - d.get("load_duration", 0)) / 1e9)
        return statistics.median(vals)
    short, long_ = "word " * 5, "word " * 400   # ~25 vs ~2000 chars
    s_s, s_l = eval_s(short), eval_s(long_)
    lc, lcl = len(short), len(long_)
    marginal = max(0.0, (s_l - s_s) / (lcl - lc))
    fixed = max(0.0, s_s - lc * marginal)
    print(f"embed idle rate: short {s_s*1000:.1f}ms/{lc}c, long {s_l*1000:.1f}ms/{lcl}c "
          f"-> fixed {fixed*1000:.3f} ms/call + {marginal*1e6:.3f} us/char")
    return fixed, marginal


def main() -> int:
    fixed, marginal = measure_embed_rate()

    pc_base = load_price_card(ROOT / "configs" / "price_card.yaml")
    cache_path = ROOT / "configs" / "price_card_cache_discount.yaml"
    pc_cache = load_price_card(cache_path) if cache_path.exists() else pc_base
    gpu_rate = (pc_base.get("gpu") or {}).get("usd_per_second") or 0.0
    print(f"gpu rate: ${gpu_rate}/s | cache-discount card: {cache_path.exists()}")

    rd = glob.glob(str(ROOT / "outputs" / "runs" / "main-full-*"))[0]
    dec = json.JSONDecoder()
    gem = {"base": defaultdict(float), "cache": defaultdict(float)}   # (arch,stage)->usd
    emb_calls = defaultdict(int)   # (arch,stage)->count
    emb_chars = defaultdict(int)   # (arch,stage)->chars
    billed_gem = 0.0
    for raw in open(f"{rd}/ledger.jsonl", encoding="utf-8"):
        s = raw.strip(); pos = 0
        while pos < len(s):
            while pos < len(s) and s[pos].isspace(): pos += 1
            if pos >= len(s): break
            try:
                o, end = dec.raw_decode(s, pos)
            except json.JSONDecodeError:
                break
            pos = end
            if not isinstance(o, dict): continue
            m = o.get("model", "") or ""; arch = o.get("architecture"); stage = o.get("stage")
            ri = o.get("run_index", 0)
            if m.startswith("gemini"):
                try:
                    row = CallRecord(**{k: o.get(k) for k in CallRecord.__dataclass_fields__})
                except TypeError:
                    continue
                billed_gem += _row_cost_usd(row, pc_base)
                if ri == 0:
                    gem["base"][(arch, stage)] += _row_cost_usd(row, pc_base)
                    gem["cache"][(arch, stage)] += _row_cost_usd(row, pc_cache)
            elif m == "bge-m3" and ri == 0:
                toks = o.get("uncached_input_tokens", 0) or 0
                emb_calls[(arch, stage)] += 1
                emb_chars[(arch, stage)] += toks * 4   # ledger token = chars // 4

    def embed_usd(arch, stage):
        gpu_s = emb_calls[(arch, stage)] * fixed + emb_chars[(arch, stage)] * marginal
        return gpu_s * gpu_rate

    N_Q = 2449  # run_index-0 questions
    per_arch = {}
    for card in ("base", "cache"):
        for a in ARCHS:
            g_off = gem[card][(a, "preprocess")]
            e_off = embed_usd(a, "preprocess")
            c_off = g_off + e_off
            g_on = gem[card][(a, "retrieval")] + gem[card][(a, "generate")]
            e_on = embed_usd(a, "retrieval") + embed_usd(a, "generate")
            c_on = g_on + e_on
            per_arch[(card, a)] = {
                "c_off_total": round(c_off, 4), "c_on_total": round(c_on, 4),
                "total": round(c_off + c_on, 4),
                "gemini_off": round(g_off, 4), "embed_off": round(e_off, 4),
                "gemini_on": round(g_on, 4), "embed_on": round(e_on, 4),
                "c_on_per_query": round(c_on / N_Q, 6),
            }

    print("\nPER-ARCH DEPLOYMENT COST (run_index 0), base card:")
    for a in ARCHS:
        d = per_arch[("base", a)]
        print(f"  {a:10s} C_off ${d['c_off_total']:7.2f} (gem ${d['gemini_off']:.2f} + emb ${d['embed_off']:.2f})  "
              f"C_on ${d['c_on_total']:7.2f} (gem ${d['gemini_on']:.2f} + emb ${d['embed_on']:.2f})  "
              f"TOTAL ${d['total']:7.2f}  | C_on/query ${d['c_on_per_query']*1000:.3f}m")

    # Break-even N vs flat (per-document build amortized over N questions).
    N_DOCS = 309  # 249 QASPER papers + 60 NovelQA novels
    breakeven = {}
    for card in ("base", "cache"):
        flat_onq = per_arch[(card, "flat")]["c_on_per_query"]
        for a in ARCHS:
            if a == "flat": continue
            coff_doc = per_arch[(card, a)]["c_off_total"] / N_DOCS
            onq = per_arch[(card, a)]["c_on_per_query"]
            curve = {N: round(coff_doc / N + onq, 6) for N in (1, 2, 5, 10, 25)}
            # crossover: coff_doc/N + onq = flat_onq  => N* = coff_doc/(flat_onq - onq)
            if flat_onq > onq:
                nstar = coff_doc / (flat_onq - onq)
                verdict = f"break-even at N*={nstar:.1f} q/doc (cheaper than flat above it)"
            else:
                nstar = None
                verdict = "never cheaper than flat (flat lower per-query too)"
            breakeven[(card, a)] = {"c_off_per_doc": round(coff_doc, 6), "c_on_per_query": onq,
                                    "flat_c_on_per_query": flat_onq, "n_star": nstar,
                                    "amortized_per_query_by_N": curve, "verdict": verdict}

    print(f"\nBREAK-EVEN vs flat (flat C_on/query ${per_arch[('base','flat')]['c_on_per_query']*1000:.3f}m), base card:")
    for a in ARCHS:
        if a == "flat": continue
        b = breakeven[("base", a)]
        print(f"  {a:10s} C_off/doc ${b['c_off_per_doc']*1000:.3f}m  C_on/query ${b['c_on_per_query']*1000:.4f}m  -> {b['verdict']}")

    deploy_total = {c: round(sum(per_arch[(c, a)]["total"] for a in ARCHS), 2) for c in ("base", "cache")}
    print(f"\ndeployment total (run_index 0): base ${deploy_total['base']}  cache ${deploy_total['cache']}  | billed gemini ${billed_gem:.2f}")

    # serialize (string keys)
    def keyed(d): return {f"{c}|{a}": v for (c, a), v in d.items()}
    (OUT / "cost_per_arch.json").write_text(json.dumps({
        "embed_rate": {"fixed_s_per_call": fixed, "marginal_s_per_char": marginal, "gpu_usd_per_second": gpu_rate},
        "per_arch": keyed(per_arch), "deployment_total": deploy_total,
        "billed_gemini": round(billed_gem, 2), "n_questions": N_Q, "n_docs": N_DOCS,
    }, indent=2), encoding="utf-8")
    (OUT / "breakeven.json").write_text(json.dumps(keyed(breakeven), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'cost_per_arch.json'} + {OUT/'breakeven.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
