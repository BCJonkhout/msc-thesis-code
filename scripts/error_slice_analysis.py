"""Deterministic RQ2 error-slice analysis: where, and why, the summary-based
architectures lose to full-context. NO model is in the loop -- every number is
a census, a string match, or a fixed-seed clustered bootstrap over the released
predictions and the datasets' OWN labels, so the result is identical regardless
of who runs it.

Inputs (all released):
  outputs/main_study/scored_cells.jsonl              objective per-question scores
  outputs/runs/main-full-*/{arch}_predictions.jsonl  raw answers (run_index 0)
  data/novelqa/questions.jsonl                       NovelQA Aspect/Complexity/Options
  data/qasper/{test,dev,train}.jsonl                 QASPER gold (all annotators) + types

Outputs into outputs/main_study/:
  error_slices.jsonl          full census: every losing case with all fields
  error_slices_review.csv     human-review sheet (author_confirms column)
  error_slices_summary.json   rates + provenance manifest + definitions
Outputs into outputs/main_study/export/ (promoted to thesis-msc/generated/):
  mainstudy_error_macros.tex     \\newcommand inline numbers used in prose
  mainstudy_error_examples.tex   verbatim worked examples (tab:results-error-examples)
  mainstudy_error_appendix.tex   NovelQA per-aspect per-method accuracy with per-slice n
                                 and 95% clustered-bootstrap CIs (tab:error-novelqa),
                                 plus the same table over NovelQA's own Complexity labels
                                 (sh/mh/dtl) as a grouping-free check
                                 (tab:error-novelqa-complexity)
  mainstudy_error_qasper.tex     QASPER per-type per-method F1 with per-slice n
                                 and 95% clustered-bootstrap CIs (tab:error-qasper)

Uncertainty: percentile bootstrap, BOOT_RESAMPLES paired cluster resamples
(cluster = novel for NovelQA, paper for QASPER), fixed per-slice seeds derived
from BOOT_SEED so the output is byte-stable across runs and slice orderings.

Losing slice (objective): flat correct AND raptor wrong AND graphrag wrong, with
NovelQA correct=accuracy==1/wrong==0 and QASPER correct=F1>=0.5/wrong=F1<=0.15.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "outputs" / "main_study"
DATA = ROOT / "data"
EXPORT = MS / "export"
EXPORT.mkdir(parents=True, exist_ok=True)

ARCH = ["flat", "naive_rag", "raptor", "graphrag"]
LABEL = {"flat": "Flat", "naive_rag": "Naive RAG", "raptor": "RAPTOR", "graphrag": "GraphRAG"}
SUMMARY_ARCHS = ["raptor", "graphrag"]
GRAN_BINS = [("gist", ["plot", "relat"]), ("mid", ["character", "settg"]),
             ("detail", ["meaning", "times", "span"])]
BOOT_RESAMPLES = 10000
BOOT_SEED = 20240613

HEDGE_PHRASES = ["does not specify", "does not mention", "not specified", "not mentioned",
                 "provided text does not", "no information", "cannot be determined",
                 "unable to determine", "does not provide", "not stated", "does not state",
                 "text does not", "is not specified", "unknown"]


def is_correct(ds, v): return (v == 1) if ds == "novelqa" else (v >= 0.5)
def is_wrong(ds, v): return (v == 0) if ds == "novelqa" else (v <= 0.15)
def is_hedge(t): return bool(t) and any(h in t.lower() for h in HEDGE_PHRASES)
def sha12(p: Path): return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def latex_escape(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s


def trunc(s, n=70):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def find_run():
    cands = sorted(p for p in (MS.parent / "runs").glob("main-full-*")
                   if (p / "flat_predictions.jsonl").exists())
    if not cands:
        raise SystemExit("no main-full-* run dir with predictions found")
    return cands[-1]


def absent_letters(options):
    out = set()
    for L, txt in (options or {}).items():
        tl = str(txt).strip().lower()
        if tl.startswith("no") or "never" in tl or "does not appear" in tl:
            out.add(L)
    return out


def qasper_type(answer_block):
    a = answer_block.get("answer", answer_block)
    if a.get("unanswerable"):
        return "unanswerable", "(unanswerable)"
    if a.get("yes_no") is not None:
        return "yes_no", "Yes" if a["yes_no"] else "No"
    if a.get("extractive_spans"):
        return "extractive", " | ".join(a["extractive_spans"])
    if a.get("free_form_answer"):
        return "abstractive", a["free_form_answer"]
    return "other", ""


def main() -> None:
    run = find_run()
    manifest = {
        "scored_cells": MS / "scored_cells.jsonl",
        "novelqa_questions": DATA / "novelqa" / "questions.jsonl",
        "flat_predictions": run / "flat_predictions.jsonl",
        "raptor_predictions": run / "raptor_predictions.jsonl",
        "graphrag_predictions": run / "graphrag_predictions.jsonl",
        "naive_rag_predictions": run / "naive_rag_predictions.jsonl",
    }

    metric = defaultdict(dict)
    for line in manifest["scored_cells"].open(encoding="utf-8"):
        d = json.loads(line)
        metric[(d["dataset"], d["cluster"], d["qid"])][d["arch"]] = d["metric"]

    nq = {}
    for line in manifest["novelqa_questions"].open(encoding="utf-8"):
        d = json.loads(line)
        opts = d.get("Options")
        if isinstance(opts, str):
            try:
                opts = ast.literal_eval(opts)
            except Exception:
                opts = {}
        nq[(d["novel_id"], d["question_id"])] = (d.get("Aspect"), d.get("Complexity"), opts or {})

    # QASPER: collect ALL annotator gold answers (max-F1 scoring is multi-gold)
    qgold = defaultdict(list)   # qid -> [(type, text), ...]
    qtype = {}                  # qid -> first answer type (coarse class)
    for fn in ["test.jsonl", "dev.jsonl", "train.jsonl"]:
        fp = DATA / "qasper" / fn
        if not fp.exists():
            continue
        for line in fp.open(encoding="utf-8"):
            d = json.loads(line)
            qas = d.get("qas")
            if qas is None:
                continue
            if isinstance(qas, str):
                qas = ast.literal_eval(qas)
            for q in qas:
                anss = q.get("answers", [])
                golds = [qasper_type(a) for a in anss] or [("other", "")]
                qgold[q["question_id"]] = golds
                qtype[q["question_id"]] = golds[0][0]

    pred = {}
    for arch in ARCH:
        for line in (run / f"{arch}_predictions.jsonl").open(encoding="utf-8"):
            d = json.loads(line)
            if d["run_index"] == 0:
                pred[(d["dataset"], d["paper_id"], d["question_id"], arch)] = d

    def ans(ds, c, q, a):
        r = pred.get((ds, c, q, a), {})
        return r.get("predicted_letter") if ds == "novelqa" else r.get("predicted_answer")

    eval_keys = sorted(k for k in metric if all(a in metric[k] for a in ARCH))

    # per-method accuracy by aspect (NovelQA) / type (QASPER) and by granularity bin;
    # values are kept as (cluster, metric) pairs so the bootstrap can resample clusters
    acc_aspect = defaultdict(lambda: defaultdict(list))
    acc_qtype = defaultdict(lambda: defaultdict(list))
    acc_bin = defaultdict(lambda: defaultdict(list))
    acc_cx = defaultdict(lambda: defaultdict(list))
    asp_to_bin = {a: b for b, members in GRAN_BINS for a in members}
    for k in eval_keys:
        ds, c, q = k
        if ds == "novelqa":
            a, cx, _ = nq.get((c, q), (None, None, {}))
            for arch in ARCH:
                acc_aspect[a][arch].append((c, metric[k][arch]))
                if a in asp_to_bin:
                    acc_bin[asp_to_bin[a]][arch].append((c, metric[k][arch]))
                if cx is not None:
                    acc_cx[cx][arch].append((c, metric[k][arch]))
        else:
            t = qtype.get(q, "unknown")
            for arch in ARCH:
                acc_qtype[t][arch].append((c, metric[k][arch]))

    def mean_by(d):
        return {key: {a: sum(x[1] for x in v[a]) / len(v[a]) for a in ARCH}
                for key, v in d.items() if v["flat"]}

    def n_by(d):
        return {key: len(v["flat"]) for key, v in d.items() if v["flat"]}

    def boot_ci(slice_rows, tag):
        """95% percentile CI per method from a paired clustered bootstrap: resample
        clusters (novels/papers) with replacement, same resample for all methods,
        per-slice seed derived from BOOT_SEED + tag so output is order-independent."""
        clusters = sorted({c for c, _ in slice_rows["flat"]})
        cix = {c: i for i, c in enumerate(clusters)}
        counts = np.zeros(len(clusters))
        for c, _ in slice_rows["flat"]:
            counts[cix[c]] += 1
        rng = np.random.default_rng(
            [BOOT_SEED, int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")])
        idx = rng.integers(0, len(clusters), size=(BOOT_RESAMPLES, len(clusters)))
        den = counts[idx].sum(axis=1)
        out = {}
        for a in ARCH:
            sums = np.zeros(len(clusters))
            for c, v in slice_rows[a]:
                sums[cix[c]] += v
            lo, hi = np.percentile(sums[idx].sum(axis=1) / den, [2.5, 97.5])
            out[a] = (float(lo), float(hi))
        return out

    def ci_by(d, tag_prefix):
        return {key: boot_ci(v, f"{tag_prefix}:{key}") for key, v in d.items() if v["flat"]}

    def boot_gap_diff(rows_a, rows_b, tag):
        """95% percentile CI for gap(bucket a) - gap(bucket b), gap = flat mean minus
        the worst (min) of the other three method means. Same paired clustered
        machinery as boot_ci: ONE shared novel-cluster resample (union of the two
        buckets' clusters) reused across both buckets and all four methods, and the
        per-bucket gap is recomputed inside every resample before differencing."""
        clusters = sorted({c for c, _ in rows_a["flat"]} | {c for c, _ in rows_b["flat"]})
        cix = {c: i for i, c in enumerate(clusters)}
        rng = np.random.default_rng(
            [BOOT_SEED, int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")])
        idx = rng.integers(0, len(clusters), size=(BOOT_RESAMPLES, len(clusters)))

        def bucket_means(rows):
            counts = np.zeros(len(clusters))
            for c, _ in rows["flat"]:
                counts[cix[c]] += 1
            den = counts[idx].sum(axis=1)
            assert (den > 0).all(), f"empty bucket in a bootstrap resample ({tag})"
            means = {}
            for a in ARCH:
                sums = np.zeros(len(clusters))
                for c, v in rows[a]:
                    sums[cix[c]] += v
                means[a] = sums[idx].sum(axis=1) / den
            return means

        ma, mb = bucket_means(rows_a), bucket_means(rows_b)
        ga = ma["flat"] - np.minimum.reduce([ma[a] for a in ["naive_rag", "raptor", "graphrag"]])
        gb = mb["flat"] - np.minimum.reduce([mb[a] for a in ["naive_rag", "raptor", "graphrag"]])
        lo, hi = np.percentile(ga - gb, [2.5, 97.5])
        return float(lo), float(hi)

    am, qm, bm = mean_by(acc_aspect), mean_by(acc_qtype), mean_by(acc_bin)
    an, qn, bn = n_by(acc_aspect), n_by(acc_qtype), n_by(acc_bin)
    aci, qci = ci_by(acc_aspect, "novelqa"), ci_by(acc_qtype, "qasper")
    cxm, cxn = mean_by(acc_cx), n_by(acc_cx)
    cxci = ci_by(acc_cx, "novelqa-complexity")

    def gap(row):
        return row["flat"] - min(row[a] for a in ["naive_rag", "raptor", "graphrag"])

    # gap-difference point estimates + paired clustered-bootstrap CIs
    gapdiff = {
        "complexity_dtl_minus_sh": (gap(cxm["dtl"]) - gap(cxm["sh"]),
                                    boot_gap_diff(acc_cx["dtl"], acc_cx["sh"],
                                                  "gapdiff:complexity:dtl-sh")),
        "granularity_detail_minus_gist": (gap(bm["detail"]) - gap(bm["gist"]),
                                          boot_gap_diff(acc_bin["detail"], acc_bin["gist"],
                                                        "gapdiff:gran:detail-gist")),
        "granularity_mid_minus_gist": (gap(bm["mid"]) - gap(bm["gist"]),
                                       boot_gap_diff(acc_bin["mid"], acc_bin["gist"],
                                                     "gapdiff:gran:mid-gist")),
        "granularity_detail_minus_mid": (gap(bm["detail"]) - gap(bm["mid"]),
                                         boot_gap_diff(acc_bin["detail"], acc_bin["mid"],
                                                       "gapdiff:gran:detail-mid")),
    }

    # census of the losing slice
    census, totals, losing, abstr_specific = [], Counter(), Counter(), Counter()
    for k in eval_keys:
        ds, c, q = k
        v = metric[k]
        totals[ds] += 1
        if not (is_correct(ds, v["flat"]) and is_wrong(ds, v["raptor"]) and is_wrong(ds, v["graphrag"])):
            continue
        losing[ds] += 1
        naive_ok = is_correct(ds, v["naive_rag"])
        abstr_specific[ds] += int(naive_ok)
        row = {"id": f"{c}/{q}", "dataset": ds, "cluster": c, "qid": q,
               "question": pred.get((ds, c, q, "flat"), {}).get("question"),
               "metric": {a: v[a] for a in ARCH},
               "answer": {a: ans(ds, c, q, a) for a in ARCH},
               "naive_rag_also_correct": naive_ok}
        if ds == "novelqa":
            asp, cx, opts = nq.get((c, q), (None, None, {}))
            ab = absent_letters(opts)
            gold = ans(ds, c, q, "flat")
            row.update({"aspect": asp, "complexity": cx, "options": opts,
                        "gold_letter": gold, "gold_text": opts.get(gold),
                        "absent_option_present": bool(ab),
                        "abstractors_say_absent": bool(ab) and all(ans(ds, c, q, a) in ab for a in SUMMARY_ARCHS)})
        else:
            row.update({"qtype": qtype.get(q, "unknown"),
                        "golds": qgold.get(q, []),
                        "abstractors_hedge": sum(is_hedge(ans(ds, c, q, a)) for a in SUMMARY_ARCHS)})
        census.append(row)

    # Tell 1: QASPER hedge on questions Flat answers concretely+correctly
    flat_good = [k for k in eval_keys if k[0] == "qasper"
                 and not is_hedge(pred.get((k[0], k[1], k[2], "flat"), {}).get("predicted_answer"))
                 and (pred.get((k[0], k[1], k[2], "flat"), {}).get("answer_f1") or 0) >= 0.5]
    hedge_flatgood = {a: sum(is_hedge(pred.get((k[0], k[1], k[2], a), {}).get("predicted_answer")) for k in flat_good) for a in ARCH}

    # Tell 2: NovelQA presence/count false-absent
    times_losing = [r for r in census if r["dataset"] == "novelqa" and r.get("aspect") == "times"]
    times_abs = [r for r in times_losing if r["absent_option_present"]]
    false_absent = {a: sum(1 for r in times_abs if r["answer"][a] in absent_letters(r["options"])) for a in ARCH}
    false_absent_both = sum(1 for r in times_abs
                            if all(r["answer"][a] in absent_letters(r["options"]) for a in SUMMARY_ARCHS))

    # Tell 2, full denominator: ALL scored `times` questions offering an absent
    # option whose gold answer is a present option and Flat answers correctly
    # (Flat correct implies its letter IS the gold letter) -- not restricted to
    # the losing slice. Counts how often each method picks an absent option.
    fa_all_den, fa_all = 0, Counter()
    for k in eval_keys:
        ds, c, q = k
        if ds != "novelqa":
            continue
        asp, _, opts = nq.get((c, q), (None, None, {}))
        if asp != "times":
            continue
        ab = absent_letters(opts)
        if not ab or not is_correct(ds, metric[k]["flat"]):
            continue
        if ans(ds, c, q, "flat") in ab:   # gold must be a present option
            continue
        fa_all_den += 1
        for a in ARCH:
            fa_all[a] += int(ans(ds, c, q, a) in ab)

    # deterministic example selection: abstraction-specific dissociation
    # (Flat correct AND Naive RAG correct AND both summary methods wrong),
    # so the table shows the loss is the tree/graph abstraction, not retrieval.
    nq_ex = sorted(r["id"] for r in census if r["dataset"] == "novelqa"
                   and r.get("naive_rag_also_correct"))[:3]
    qa_ex = sorted(r["id"] for r in census if r["dataset"] == "qasper"
                   and r.get("naive_rag_also_correct")
                   and r.get("golds") and all(t != "unanswerable" for t, _ in r["golds"]))[:1]

    # ---- census jsonl + review csv ----
    with (MS / "error_slices.jsonl").open("w", encoding="utf-8") as fh:
        for r in census:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (MS / "error_slices_review.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dataset", "aspect_or_qtype", "complexity", "question", "gold",
                    "flat", "naive_rag", "raptor", "graphrag", "derived_flag", "author_confirms"])
        for r in census:
            if r["dataset"] == "novelqa":
                flag = "abstractors_say_absent" if r.get("abstractors_say_absent") else ("abstraction_specific" if r["naive_rag_also_correct"] else "")
                gold = f'{r.get("gold_letter")}: {r.get("gold_text")}'
                a = {x: f'{r["answer"][x]}: {(r.get("options") or {}).get(r["answer"][x], "")}' for x in ARCH}
                cat = r.get("aspect")
            else:
                flag = f'abstractors_hedge={r.get("abstractors_hedge")}'
                gold = " | ".join(t for _, t in r.get("golds", []))
                a = {x: r["answer"][x] for x in ARCH}
                cat = r.get("qtype")
            w.writerow([r["id"], r["dataset"], cat, r.get("complexity", ""), r["question"],
                        gold, a["flat"], a["naive_rag"], a["raptor"], a["graphrag"], flag, ""])

    # ---- summary json ----
    summary = {
        "generated_by": "code/scripts/error_slice_analysis.py (deterministic; no model in the loop)",
        "input_manifest": {n: {"path": str(p.relative_to(ROOT)), "sha256_12": sha12(p),
                               "lines": sum(1 for _ in p.open(encoding="utf-8"))} for n, p in manifest.items()},
        "definitions": {"losing_slice": "flat correct AND raptor wrong AND graphrag wrong",
                        "correct": {"novelqa": "accuracy==1", "qasper": "answer_f1>=0.5"},
                        "wrong": {"novelqa": "accuracy==0", "qasper": "answer_f1<=0.15"},
                        "hedge_phrases": HEDGE_PHRASES,
                        "granularity_bins": {b: mem for b, mem in GRAN_BINS},
                        "complexity_labels": {"sh": "single-hop", "mh": "multi-hop",
                                              "dtl": "detail",
                                              "source": "data/novelqa/questions.jsonl Complexity field, "
                                                        "joined per question like Aspect"},
                        "gap": "flat mean minus min(naive_rag, raptor, graphrag) mean on the slice",
                        "gap_difference_bootstrap": "gap(bucket a) - gap(bucket b): ONE shared paired "
                                                    "novel-cluster resample (union of both buckets' "
                                                    "clusters, same resamples/seed policy as the "
                                                    "per-slice CIs) reused across buckets and methods; "
                                                    "the per-bucket gap is recomputed inside every "
                                                    "resample before differencing",
                        "false_absent_full_pool": "all scored NovelQA `times` questions offering an "
                                                  "absent option (absent_letters rule) whose gold "
                                                  "answer is a present option and Flat is correct; "
                                                  "NOT restricted to the losing slice",
                        "bootstrap": {"resamples": BOOT_RESAMPLES, "seed": BOOT_SEED,
                                      "interval": "95% percentile, paired cluster resample",
                                      "cluster": {"novelqa": "novel", "qasper": "paper"}}},
        "totals_eval": dict(totals), "losing_slice_size": dict(losing),
        "abstraction_specific": dict(abstr_specific),
        "granularity_bin_means": bm,
        "granularity_bin_n": bn,
        "granularity_bin_gaps": {b: round(gap(bm[b]), 4) for b in bm},
        "novelqa_accuracy_by_aspect": am,
        "novelqa_aspect_n": an,
        "novelqa_accuracy_by_aspect_ci95": {k: {a: [round(lo, 4), round(hi, 4)] for a, (lo, hi) in v.items()}
                                            for k, v in aci.items()},
        "novelqa_accuracy_by_complexity": cxm,
        "novelqa_complexity_n": cxn,
        "novelqa_accuracy_by_complexity_ci95": {k: {a: [round(lo, 4), round(hi, 4)] for a, (lo, hi) in v.items()}
                                                for k, v in cxci.items()},
        "novelqa_complexity_gaps": {k: round(gap(cxm[k]), 4) for k in cxm},
        "gap_difference_ci95": {name: {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}
                                for name, (d, (lo, hi)) in gapdiff.items()},
        "qasper_f1_by_type": qm,
        "qasper_type_n": qn,
        "qasper_f1_by_type_ci95": {k: {a: [round(lo, 4), round(hi, 4)] for a, (lo, hi) in v.items()}
                                   for k, v in qci.items()},
        "tell1_hedge_on_flat_correct_concrete": {"base_n": len(flat_good), "hedges": hedge_flatgood},
        "tell2_false_absent": {"times_with_absent_n": len(times_abs), "picks_absent": false_absent,
                               "picks_absent_both": false_absent_both},
        "tell2_false_absent_full_pool": {"den": fa_all_den,
                                         "picks_absent": {a: fa_all[a] for a in ARCH}},
        "selected_examples": {"novelqa": nq_ex, "qasper": qa_ex},
    }
    (MS / "error_slices_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- export: macros ----
    gist, mid, det = gap(bm["gist"]), gap(bm["mid"]), gap(bm["detail"])
    fa = false_absent["raptor"]
    qmax = max(gap(qm[t]) for t in qm)
    macros = [
        "% Auto-generated by code/scripts/error_slice_analysis.py -- do not edit by hand.",
        f"\\newcommand{{\\errorGapGist}}{{{gist:.2f}}}",
        f"\\newcommand{{\\errorGapMid}}{{{mid:.2f}}}",
        f"\\newcommand{{\\errorGapDetail}}{{{det:.2f}}}",
        f"\\newcommand{{\\errorFalseAbsentNum}}{{{fa}}}",
        f"\\newcommand{{\\errorFalseAbsentDen}}{{{len(times_abs)}}}",
        f"\\newcommand{{\\errorFalseAbsentPct}}{{{round(100*fa/max(1,len(times_abs)))}}}",
        f"\\newcommand{{\\errorFalseAbsentRaptor}}{{{false_absent['raptor']}}}",
        f"\\newcommand{{\\errorFalseAbsentGraph}}{{{false_absent['graphrag']}}}",
        f"\\newcommand{{\\errorFalseAbsentBoth}}{{{false_absent_both}}}",
        f"\\newcommand{{\\errorSliceNGist}}{{{bn['gist']}}}",
        f"\\newcommand{{\\errorSliceNMid}}{{{bn['mid']}}}",
        f"\\newcommand{{\\errorSliceNDetail}}{{{bn['detail']}}}",
        f"\\newcommand{{\\errorHedgeGraph}}{{{hedge_flatgood['graphrag']}}}",
        f"\\newcommand{{\\errorHedgeBase}}{{{len(flat_good)}}}",
        f"\\newcommand{{\\errorNovelLosing}}{{{losing['novelqa']}}}",
        f"\\newcommand{{\\errorNovelTotal}}{{{totals['novelqa']}}}",
        f"\\newcommand{{\\errorAbstractionSpecific}}{{{abstr_specific['novelqa']}}}",
        # 3dp: the value sits near 0.10, where 2dp rounding erases the margin.
        f"\\newcommand{{\\errorQasperMaxGap}}{{{qmax:.3f}}}",
        f"\\newcommand{{\\errorSliceNSh}}{{{cxn['sh']}}}",
        f"\\newcommand{{\\errorSliceNMh}}{{{cxn['mh']}}}",
        f"\\newcommand{{\\errorSliceNDtl}}{{{cxn['dtl']}}}",
        f"\\newcommand{{\\errorGapSh}}{{{gap(cxm['sh']):.2f}}}",
        f"\\newcommand{{\\errorGapMh}}{{{gap(cxm['mh']):.2f}}}",
        f"\\newcommand{{\\errorGapDtl}}{{{gap(cxm['dtl']):.2f}}}",
        f"\\newcommand{{\\errorGapDtlShDiff}}{{{gapdiff['complexity_dtl_minus_sh'][0]:+.2f}}}",
        f"\\newcommand{{\\errorGapDtlShCI}}{{[{gapdiff['complexity_dtl_minus_sh'][1][0]:+.2f}, "
        f"{gapdiff['complexity_dtl_minus_sh'][1][1]:+.2f}]}}",
        f"\\newcommand{{\\errorGapDetailGistDiff}}{{{gapdiff['granularity_detail_minus_gist'][0]:+.2f}}}",
        f"\\newcommand{{\\errorGapDetailGistCI}}{{[{gapdiff['granularity_detail_minus_gist'][1][0]:+.2f}, "
        f"{gapdiff['granularity_detail_minus_gist'][1][1]:+.2f}]}}",
        f"\\newcommand{{\\errorGapMidGistDiff}}{{{gapdiff['granularity_mid_minus_gist'][0]:+.2f}}}",
        f"\\newcommand{{\\errorGapMidGistCI}}{{[{gapdiff['granularity_mid_minus_gist'][1][0]:+.2f}, "
        f"{gapdiff['granularity_mid_minus_gist'][1][1]:+.2f}]}}",
        f"\\newcommand{{\\errorGapDetailMidDiff}}{{{gapdiff['granularity_detail_minus_mid'][0]:+.2f}}}",
        f"\\newcommand{{\\errorGapDetailMidCI}}{{[{gapdiff['granularity_detail_minus_mid'][1][0]:+.2f}, "
        f"{gapdiff['granularity_detail_minus_mid'][1][1]:+.2f}]}}",
        f"\\newcommand{{\\errorFalseAbsentAllDen}}{{{fa_all_den}}}",
        f"\\newcommand{{\\errorFalseAbsentAllFlat}}{{{fa_all['flat']}}}",
        f"\\newcommand{{\\errorFalseAbsentAllNaive}}{{{fa_all['naive_rag']}}}",
        f"\\newcommand{{\\errorFalseAbsentAllRaptor}}{{{fa_all['raptor']}}}",
        f"\\newcommand{{\\errorFalseAbsentAllGraph}}{{{fa_all['graphrag']}}}",
        f"\\newcommand{{\\errorHedgeNaive}}{{{hedge_flatgood['naive_rag']}}}",
        f"\\newcommand{{\\errorHedgeRaptor}}{{{hedge_flatgood['raptor']}}}",
    ]
    # \errorFalseAbsentNum predates the per-method split and is pinned to the
    # RAPTOR count; guard it so a refactor cannot silently repoint the macro.
    legacy = next(m for m in macros if m.startswith("\\newcommand{\\errorFalseAbsentNum}"))
    assert legacy == f"\\newcommand{{\\errorFalseAbsentNum}}{{{false_absent['raptor']}}}", \
        "legacy \\errorFalseAbsentNum must equal the RAPTOR false-absent count"
    if false_absent["raptor"] != false_absent["graphrag"]:
        print("WARNING: false-absent counts diverge: raptor "
              f"{false_absent['raptor']} != graphrag {false_absent['graphrag']}. "
              "Any prose citing one shared count (\\errorFalseAbsentNum) is now wrong; "
              "use \\errorFalseAbsentRaptor / \\errorFalseAbsentGraph instead.",
              file=sys.stderr)
    (EXPORT / "mainstudy_error_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    # ---- export: worked examples table ----
    ex_lines = ["% Auto-generated by code/scripts/error_slice_analysis.py -- do not edit by hand.",
                "\\begin{table}[t]", "\\centering",
                "\\caption{Example questions on which Flat and Naive RAG answer correctly while "
                "RAPTOR and GraphRAG do not. Answers are the verbatim model outputs from the first "
                "of the five repeats (run~0); for QASPER the gold column lists each annotator's "
                "reference answer. Question and gold text are quoted verbatim from the "
                "datasets, including any typographical errors.}"
                "\\label{tab:results-error-examples}",
                "{\\footnotesize", "\\begin{tabularx}{\\linewidth}{p{0.34\\linewidth}p{0.10\\linewidth}X}",
                "\\toprule", "Question & Gold & Flat / Naive RAG / RAPTOR / GraphRAG \\\\", "\\midrule"]
    cen = {r["id"]: r for r in census}
    for cid in nq_ex:
        r = cen[cid]
        opts = r.get("options") or {}
        def opt(letter): return f"{letter}: {trunc(opts.get(letter, ''), 28)}"
        ex_lines.append(f"\\emph{{NovelQA}}: {latex_escape(trunc(r['question'], 90))} & "
                        f"{latex_escape(opt(r['gold_letter']))} & "
                        f"{latex_escape(opt(r['answer']['flat']))} / {latex_escape(opt(r['answer']['naive_rag']))} / "
                        f"{latex_escape(opt(r['answer']['raptor']))} / "
                        f"{latex_escape(opt(r['answer']['graphrag']))} \\\\[2pt]")
    for cid in qa_ex:
        r = cen[cid]
        gold = " / ".join(t for _, t in r.get("golds", []) if t)
        ex_lines.append(f"\\emph{{QASPER}}: {latex_escape(trunc(r['question'], 90))} & "
                        f"{latex_escape(trunc(gold, 28))} & "
                        f"{latex_escape(trunc(r['answer']['flat'], 18))} / "
                        f"{latex_escape(trunc(r['answer']['naive_rag'], 18))} / "
                        f"{latex_escape(trunc(r['answer']['raptor'], 30))} / "
                        f"{latex_escape(trunc(r['answer']['graphrag'], 30))} \\\\")
    ex_lines += ["\\bottomrule", "\\end{tabularx}", "}", "\\end{table}", ""]
    (EXPORT / "mainstudy_error_examples.tex").write_text("\n".join(ex_lines), encoding="utf-8")

    # ---- export: appendix per-method tables ----
    def acc_table(rows_means, rows_n, rows_ci, order, name_map, caption, label, comment=True):
        out = (["% Auto-generated by code/scripts/error_slice_analysis.py -- do not edit by hand."]
               if comment else [])
        out += ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}",
               "{\\footnotesize", "\\begin{tabular}{lrrrrr}", "\\toprule",
               "subset & $n$ & Flat & Naive RAG & RAPTOR & GraphRAG \\\\", "\\midrule"]
        for key in order:
            if key not in rows_means:
                continue
            r, ci = rows_means[key], rows_ci[key]
            out.append(f"{name_map.get(key, key)} & {rows_n[key]} & "
                       + " & ".join(f"{r[a]:.2f}" for a in ARCH) + " \\\\")
            out.append("& & " + " & ".join(f"{{\\scriptsize[{ci[a][0]:.2f}, {ci[a][1]:.2f}]}}"
                                           for a in ARCH) + " \\\\[2pt]")
        out += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""]
        return "\n".join(out)

    boot_n_tex = f"{BOOT_RESAMPLES:,}".replace(",", "\\,")
    ci_note = ("Brackets give 95\\% clustered-bootstrap confidence intervals "
               f"(percentile method, {boot_n_tex} resamples, resampling {{cl}} with "
               "replacement); $n$ is the number of questions in the subset.")
    aspect_tex = acc_table(
        am, an, aci, ["times", "meaning", "span", "character", "settg", "relat", "plot"],
        {"times": "Counting (\\texttt{times})", "meaning": "Paraphrase (\\texttt{meaning})",
         "span": "Span (\\texttt{span})", "character": "Character", "settg": "Setting",
         "relat": "Relational (\\texttt{relat})", "plot": "Plot"},
        "NovelQA per-method accuracy by question aspect (full evaluation pool). "
        + ci_note.format(cl="novels"),
        "tab:error-novelqa")
    complexity_tex = acc_table(
        cxm, cxn, cxci, ["sh", "mh", "dtl"],
        {"sh": "Single-hop (\\texttt{sh})", "mh": "Multi-hop (\\texttt{mh})",
         "dtl": "Detail (\\texttt{dtl})"},
        "NovelQA per-method accuracy by the dataset's own complexity labels "
        "(single-hop, multi-hop, detail). Same CI method as "
        "Table~\\ref{tab:error-novelqa}; $n$ is the number of questions in the subset.",
        "tab:error-novelqa-complexity", comment=False)
    (EXPORT / "mainstudy_error_appendix.tex").write_text(
        aspect_tex + "\n" + complexity_tex, encoding="utf-8")
    (EXPORT / "mainstudy_error_qasper.tex").write_text(acc_table(
        qm, qn, qci, ["yes_no", "extractive", "abstractive", "unanswerable"],
        {"yes_no": "yes/no", "extractive": "extractive", "abstractive": "abstractive",
         "unanswerable": "unanswerable"},
        "QASPER per-method Answer-F1 by answer type (full evaluation pool). "
        + ci_note.format(cl="papers"),
        "tab:error-qasper"), encoding="utf-8")

    print(f"run: {run.name}")
    print(f"losing: novelqa {losing['novelqa']}/{totals['novelqa']}, qasper {losing['qasper']}/{totals['qasper']}; "
          f"abstraction-specific novelqa {abstr_specific['novelqa']}")
    print(f"gran gaps: gist {gist:.2f}, mid {mid:.2f}, detail {det:.2f}; qasper max gap {qmax:.3f}")
    print(f"gran bin n: gist {bn['gist']}, mid {bn['mid']}, detail {bn['detail']}")
    print("complexity means: " + "; ".join(
        f"{k} (n={cxn[k]}) " + " ".join(f"{LABEL[a]} {cxm[k][a]:.3f}" for a in ARCH)
        for k in ["sh", "mh", "dtl"]))
    print("complexity gaps: " + ", ".join(f"{k} {gap(cxm[k]):.2f}" for k in ["sh", "mh", "dtl"]))
    print("gap diffs: " + "; ".join(
        f"{name} {d:+.2f} [{lo:+.2f}, {hi:+.2f}]" for name, (d, (lo, hi)) in gapdiff.items()))
    print(f"false-absent (of {len(times_abs)}): raptor {false_absent['raptor']}, "
          f"graphrag {false_absent['graphrag']}, both {false_absent_both}")
    print(f"false-absent full pool (of {fa_all_den}): " +
          ", ".join(f"{a} {fa_all[a]}" for a in ARCH))
    print(f"hedge on {len(flat_good)} flat-good: " + ", ".join(f"{a} {hedge_flatgood[a]}" for a in ARCH))
    print(f"examples: novelqa {nq_ex} qasper {qa_ex}")
    print("wrote census/csv/summary + export macros/examples/appendix/qasper")


if __name__ == "__main__":
    main()
