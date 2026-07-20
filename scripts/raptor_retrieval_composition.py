"""Leaf-vs-summary composition of RAPTOR's retrieved context, per question.

Replays the main study's collapsed-tree retrieval (top_k=20, 2000-token
budget, the exact ``_RAPTOR_DEFAULTS`` the runner used) over the cached
main-study trees and records, for every evaluation question, how the
retrieved 2000-token context splits between leaf nodes (layer 0, verbatim
document chunks) and summary nodes (layer >= 1, LLM-written cluster
summaries). Backs the Discussion 5.1 interpretation that collapsed-tree
search lets summary nodes outrank the leaf carrying a question's exact
wording; the tested prediction is that the summary-token fraction of the
retrieved context is higher for questions in finer-grained NovelQA slices
(gist -> mid -> detail; single-hop -> detail on the dataset's own
Complexity labels). QASPER's shallow trees are the comparison control.

NO model is in the loop and nothing is rebuilt or re-embedded:

  - Trees come from the main study's preprocess cache
    (outputs/preprocess_cache/raptor/<doc>/<key>/artifact.pkl), selected by
    the main-study cache-key lineage (clustering seed, summary model,
    encoder) so they are the EXACT artefacts behind the released answers.
  - Query embeddings are served from the content-addressed embed cache the
    main run wrote (outputs/embed_cache, keyed by (model, text)); a live
    Ollama server is contacted only if a query is missing from the cache,
    and the script aborts up front if neither source can serve every query.
  - Retrieval goes through the vendored ``TreeRetriever`` itself (the same
    code path ``run_raptor`` uses), so node selection, similarity ordering,
    token budgeting, and the tokenizer (tiktoken cl100k_base) are identical
    to the run. Retrieval is deterministic given tree + query embedding, and
    the five answer repeats share one retrieval, so one retrieval per
    question covers the whole grid.

Inputs (all released):
  outputs/preprocess_cache/raptor/**/artifact.pkl   main-study RAPTOR trees
  outputs/embed_cache/bge-m3/**                     cached query embeddings
  data/novelqa/questions.jsonl                      NovelQA Aspect/Complexity
  data/{qasper,novelqa}/...                         eval pools via the run's
                                                    own loaders (step_3_dry_run)

Output:
  outputs/analysis/raptor_retrieval_composition.json
    per-question rows (qid, doc, dataset, nodes/tokens by layer,
    summary_token_fraction) + per-dataset / per-slice aggregates with 95%
    document-clustered bootstrap CIs and slice-difference CIs.

Uncertainty: percentile bootstrap, BOOT_RESAMPLES paired cluster resamples
(cluster = novel for NovelQA, paper for QASPER), fixed per-slice seeds
derived from BOOT_SEED -- the same conventions as error_slice_analysis.py,
so the output is byte-stable across runs and slice orderings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The vendored RAPTOR modules call logging.basicConfig(level=INFO) at import
# time; installing a WARNING-level root handler first makes those calls
# no-ops so per-question retriever chatter stays out of the log.
logging.basicConfig(level=logging.WARNING)

# Importing the runner module pins OMP/NUMBA threads and puts
# code/third_party on sys.path (needed both for `import raptor` and for
# unpickling _RaptorState / Tree objects).
from pilot.architectures.raptor import (  # noqa: E402
    _RAPTOR_DEFAULTS,
    CLUSTERING_SEED,
)
from pilot.cli.step_3_dry_run import (  # noqa: E402
    load_novelqa_full,
    load_qasper_full,
)
from pilot.encoders import OllamaEmbedder  # noqa: E402
from raptor import BaseEmbeddingModel  # noqa: E402
from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig  # noqa: E402
from raptor.tree_structures import Tree  # noqa: E402

DATA = ROOT / "data"
TREE_CACHE = ROOT / "outputs" / "preprocess_cache" / "raptor"
EMBED_CACHE = ROOT / "outputs" / "embed_cache"
OUT_PATH = ROOT / "outputs" / "analysis" / "raptor_retrieval_composition.json"
LOG_PATH = ROOT / "runs" / "raptor_retrieval_composition.log"

# Main-study configuration the cache entries must match (run_manifest.json of
# main-full-gemini-3-1-flash-lite-preview-novelqa-qasper-literature-836965318c7c).
EMBED_MODEL = "bge-m3"
SUMMARY_MODEL = "gemini-3.1-flash-lite-preview"
# code_version_hash recorded in the main full-pass cache entries. Every
# evaluation document has exactly one tree at this hash; the constant is used
# only to disambiguate the three documents that ALSO carry a rehearsal-slice
# tree from an earlier code state (1503.00841, 1601.02403, B42).
MAIN_STUDY_CODE_HASH = "f6c5e2f9e9cf0d1c4095ec31623d4ec1f3a87781"

# Granularity binning + bootstrap conventions mirrored from
# error_slice_analysis.py so the two analyses slice identically.
GRAN_BINS = [("gist", ["plot", "relat"]), ("mid", ["character", "settg"]),
             ("detail", ["meaning", "times", "span"])]
COMPLEXITY_ORDER = ["sh", "mh", "dtl"]
BOOT_RESAMPLES = 10000
BOOT_SEED = 20240613


def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _safe_paper(paper_id: str) -> str:
    """Mirror of pilot.preprocess_cache._entry_dir's paper-id sanitisation."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in paper_id)


class _CacheBackedEmbedding(BaseEmbeddingModel):
    """Query-embedding adapter for the replayed TreeRetriever.

    Wraps ``OllamaEmbedder`` with the run's content-addressed embed cache;
    every main-study query embedding is already on disk, so replay normally
    never opens a connection. A cache miss falls through to the live server
    (connectivity is verified before any retrieval starts).
    """

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.embedder = embedder

    def create_embedding(self, text):
        return self.embedder.embed([text]).embeddings[0]


def find_tree_entry(dataset: str, paper_id: str) -> tuple[Path, dict] | None:
    """Locate the main-study cache entry for one document.

    Matches on the build-relevant key inputs (clustering seed, summary
    model, encoder, dataset) rather than recomputing the key hash, because
    the hash also binds git HEAD at build time. When a document carries a
    second (rehearsal-slice) entry, the full-pass entry is selected by its
    recorded code_version_hash.
    """
    doc_dir = TREE_CACHE / _safe_paper(paper_id)
    if not doc_dir.is_dir():
        return None
    candidates: list[tuple[Path, dict]] = []
    for meta_path in sorted(doc_dir.glob("*/build_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        k = meta.get("cache_key_inputs") or {}
        if (
            k.get("architecture") == "raptor"
            and k.get("paper_id") == paper_id
            and k.get("dataset") == dataset
            and k.get("seed") == CLUSTERING_SEED
            and k.get("summary_model") == SUMMARY_MODEL
            and k.get("encoder_model") == EMBED_MODEL
            and (meta_path.parent / "artifact.pkl").exists()
        ):
            candidates.append((meta_path.parent / "artifact.pkl", meta))
    if not candidates:
        return None
    if len(candidates) > 1:
        preferred = [
            c for c in candidates
            if (c[1].get("cache_key_inputs") or {}).get("code_version_hash")
            == MAIN_STUDY_CODE_HASH
        ]
        if preferred:
            return preferred[0]
    return candidates[0]


def load_tree(pkl_path: Path) -> Tree | None:
    """Unpickle a cached _RaptorState and surface its pure-data Tree."""
    with open(pkl_path, "rb") as fh:
        state = pickle.load(fh)
    tree = getattr(state, "_restored_tree", None)
    if tree is None:
        ra = getattr(state, "ra", None)
        tree = getattr(ra, "tree", None)
    return tree if isinstance(tree, Tree) else None


def ollama_alive(base_url: str = "http://localhost:11434") -> bool:
    import httpx
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=3.0).status_code == 200
    except Exception:
        return False


# ── clustered bootstrap (error_slice_analysis.py conventions) ──────────────

def _resample_indices(clusters: list[str], tag: str) -> np.ndarray:
    rng = np.random.default_rng(
        [BOOT_SEED, int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big")])
    return rng.integers(0, len(clusters), size=(BOOT_RESAMPLES, len(clusters)))


def boot_mean_ci(rows: list[tuple[str, float]], tag: str) -> tuple[float, float]:
    """95% percentile CI for the mean of ``rows`` values, resampling
    clusters (documents) with replacement; per-slice seed derived from
    BOOT_SEED + tag so output is order-independent."""
    clusters = sorted({c for c, _ in rows})
    cix = {c: i for i, c in enumerate(clusters)}
    counts = np.zeros(len(clusters))
    sums = np.zeros(len(clusters))
    for c, v in rows:
        counts[cix[c]] += 1
        sums[cix[c]] += v
    idx = _resample_indices(clusters, tag)
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff_ci(rows_a: list[tuple[str, float]], rows_b: list[tuple[str, float]],
                 tag: str) -> tuple[float, float]:
    """95% percentile CI for mean(a) - mean(b): ONE shared document-cluster
    resample (union of the two slices' clusters) reused across both slices,
    with each slice mean recomputed inside every resample before
    differencing -- the paired machinery error_slice_analysis.py uses for
    its gap differences."""
    clusters = sorted({c for c, _ in rows_a} | {c for c, _ in rows_b})
    cix = {c: i for i, c in enumerate(clusters)}
    idx = _resample_indices(clusters, tag)

    def slice_means(rows: list[tuple[str, float]]) -> np.ndarray:
        counts = np.zeros(len(clusters))
        sums = np.zeros(len(clusters))
        for c, v in rows:
            counts[cix[c]] += 1
            sums[cix[c]] += v
        den = counts[idx].sum(axis=1)
        assert (den > 0).all(), f"empty slice in a bootstrap resample ({tag})"
        return sums[idx].sum(axis=1) / den

    diffs = slice_means(rows_a) - slice_means(rows_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def mean(rows: list[tuple[str, float]]) -> float:
    return sum(v for _, v in rows) / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--log", type=Path, default=LOG_PATH)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = args.log.open("w", encoding="utf-8")
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log("[raptor-comp] loading evaluation pools (main-study loaders, full split)")
    items = load_qasper_full(DATA) + load_novelqa_full(DATA)
    items.sort(key=lambda it: (it["dataset"], it["paper_id"], it["question_id"]))
    log(f"[raptor-comp] eval questions: {len(items)}")

    # NovelQA aspect/complexity labels, joined per question exactly as in
    # error_slice_analysis.py.
    nq_labels: dict[tuple[str, str], tuple[str | None, str | None]] = {}
    nq_questions_path = DATA / "novelqa" / "questions.jsonl"
    for line in nq_questions_path.open(encoding="utf-8"):
        d = json.loads(line)
        nq_labels[(d["novel_id"], d["question_id"])] = (d.get("Aspect"), d.get("Complexity"))
    asp_to_bin = {a: b for b, members in GRAN_BINS for a in members}

    # Query-embedding availability gate: every question must be resolvable
    # from the embed cache or a live Ollama server BEFORE retrieval starts,
    # otherwise the embedder's retry/backoff would stall mid-run.
    embedder = OllamaEmbedder(model=EMBED_MODEL, cache_dir=EMBED_CACHE)
    uncached = [it for it in items if embedder._cache_get(it["question"]) is None]
    log(f"[raptor-comp] query embeddings not in cache: {len(uncached)}")
    if uncached:
        if not ollama_alive(embedder.base_url):
            examples = [(it["dataset"], it["paper_id"], it["question_id"])
                        for it in uncached[:5]]
            log(f"[raptor-comp] ABORT: {len(uncached)} query embeddings missing "
                f"from {EMBED_CACHE} and Ollama at {embedder.base_url} is not "
                f"reachable. Examples: {examples}")
            return 2
        log(f"[raptor-comp] Ollama reachable at {embedder.base_url}; "
            f"misses will be embedded live and cached")

    # Group questions by document; one tree load + one retriever per doc.
    by_doc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        by_doc[(it["dataset"], it["paper_id"])].append(it)
    doc_keys = sorted(by_doc)
    log(f"[raptor-comp] documents: {len(doc_keys)}")

    adapter = _CacheBackedEmbedding(embedder)
    tr_cfg = TreeRetrieverConfig(
        embedding_model=adapter,
        context_embedding_model="EMB",
        top_k=_RAPTOR_DEFAULTS["tr_top_k"],
        threshold=_RAPTOR_DEFAULTS["tr_threshold"],
        selection_mode=_RAPTOR_DEFAULTS["tr_selection_mode"],
    )

    per_question: list[dict] = []
    documents: list[dict] = []
    missing_docs: list[dict] = []
    skipped_questions = 0

    for i, (dataset, paper_id) in enumerate(doc_keys, 1):
        qs = by_doc[(dataset, paper_id)]
        entry = find_tree_entry(dataset, paper_id)
        if entry is None:
            missing_docs.append({"dataset": dataset, "doc_id": paper_id,
                                 "n_questions": len(qs)})
            skipped_questions += len(qs)
            log(f"[raptor-comp] [{i}/{len(doc_keys)}] {dataset}/{paper_id} "
                f"NO CACHED TREE -- skipping {len(qs)} questions")
            continue
        pkl_path, meta = entry
        tree = load_tree(pkl_path)
        if tree is None:
            missing_docs.append({"dataset": dataset, "doc_id": paper_id,
                                 "n_questions": len(qs),
                                 "reason": "artifact unpickled but carries no Tree"})
            skipped_questions += len(qs)
            log(f"[raptor-comp] [{i}/{len(doc_keys)}] {dataset}/{paper_id} "
                f"UNLOADABLE TREE -- skipping {len(qs)} questions")
            continue

        retriever = TreeRetriever(tr_cfg, tree)
        tok = retriever.tokenizer  # tiktoken cl100k_base, the run's budget tokenizer

        for it in qs:
            _context, layer_info = retriever.retrieve(
                it["question"],
                collapse_tree=_RAPTOR_DEFAULTS["collapse_tree"],
                top_k=_RAPTOR_DEFAULTS["tr_top_k"],
                max_tokens=_RAPTOR_DEFAULTS["retrieval_max_tokens"],
                return_layer_information=True,
            )
            nodes_by_layer: dict[int, int] = defaultdict(int)
            tokens_by_layer: dict[int, int] = defaultdict(int)
            for li in layer_info:
                node = tree.all_nodes[li["node_index"]]
                layer = li["layer_number"]
                nodes_by_layer[layer] += 1
                tokens_by_layer[layer] += len(tok.encode(node.text))
            total_tokens = sum(tokens_by_layer.values())
            total_nodes = sum(nodes_by_layer.values())
            summary_tokens = sum(v for l, v in tokens_by_layer.items() if l >= 1)
            summary_nodes = sum(v for l, v in nodes_by_layer.items() if l >= 1)
            row = {
                "dataset": dataset,
                "doc_id": paper_id,
                "qid": it["question_id"],
                "n_nodes": total_nodes,
                "n_nodes_by_layer": {str(l): nodes_by_layer[l] for l in sorted(nodes_by_layer)},
                "total_tokens": total_tokens,
                "tokens_by_layer": {str(l): tokens_by_layer[l] for l in sorted(tokens_by_layer)},
                "summary_token_fraction": (summary_tokens / total_tokens
                                           if total_tokens else None),
                "summary_node_fraction": (summary_nodes / total_nodes
                                          if total_nodes else None),
            }
            if dataset == "novelqa":
                aspect, cx = nq_labels.get((paper_id, it["question_id"]), (None, None))
                row["aspect"] = aspect
                row["complexity"] = cx
                row["granularity_bin"] = asp_to_bin.get(aspect)
            per_question.append(row)

        layer_sizes = {str(l): len(ns) for l, ns in sorted(tree.layer_to_nodes.items())}
        documents.append({
            "dataset": dataset,
            "doc_id": paper_id,
            "n_questions": len(qs),
            "tree_num_layers": tree.num_layers,
            "tree_nodes_by_layer": layer_sizes,
            "cache_entry": pkl_path.parent.name,
            "cache_code_version_hash": (meta.get("cache_key_inputs") or {}).get("code_version_hash"),
            "cache_build_run_id": meta.get("build_run_id"),
        })
        log(f"[raptor-comp] [{i}/{len(doc_keys)}] {dataset}/{paper_id} "
            f"layers={tree.num_layers + 1} nodes={len(tree.all_nodes)} "
            f"questions={len(qs)} elapsed={time.monotonic() - t0:.0f}s")

    # ── aggregates ──────────────────────────────────────────────────────
    ds_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    gran_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    cx_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    node_frac_rows: dict[str, list[tuple[str, float]]] = defaultdict(list)
    ds_tokens_by_layer: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in per_question:
        if r["summary_token_fraction"] is None:
            continue
        ds_rows[r["dataset"]].append((r["doc_id"], r["summary_token_fraction"]))
        node_frac_rows[r["dataset"]].append((r["doc_id"], r["summary_node_fraction"]))
        for l, v in r["tokens_by_layer"].items():
            ds_tokens_by_layer[r["dataset"]][l] += v
        if r["dataset"] == "novelqa":
            if r.get("granularity_bin"):
                gran_rows[r["granularity_bin"]].append((r["doc_id"], r["summary_token_fraction"]))
            if r.get("complexity"):
                cx_rows[r["complexity"]].append((r["doc_id"], r["summary_token_fraction"]))

    def slice_stats(rows: list[tuple[str, float]], tag: str) -> dict:
        lo, hi = boot_mean_ci(rows, tag)
        return {"n": len(rows), "n_docs": len({c for c, _ in rows}),
                "mean_summary_token_fraction": round(mean(rows), 4),
                "ci95": [round(lo, 4), round(hi, 4)]}

    per_dataset = {}
    for ds in sorted(ds_rows):
        s = slice_stats(ds_rows[ds], f"raptorcomp:ds:{ds}")
        s["mean_summary_node_fraction"] = round(mean(node_frac_rows[ds]), 4)
        totals = ds_tokens_by_layer[ds]
        grand = sum(totals.values())
        s["retrieved_token_share_by_layer"] = {
            l: round(v / grand, 4) for l, v in sorted(totals.items())}
        qrows = [r for r in per_question if r["dataset"] == ds
                 and r["summary_token_fraction"] is not None]
        s["mean_retrieved_tokens"] = round(
            sum(r["total_tokens"] for r in qrows) / len(qrows), 1)
        s["mean_retrieved_nodes"] = round(
            sum(r["n_nodes"] for r in qrows) / len(qrows), 2)
        s["questions_with_any_summary_tokens"] = sum(
            1 for r in qrows if r["summary_token_fraction"] > 0)
        per_dataset[ds] = s

    by_granularity = {b: slice_stats(gran_rows[b], f"raptorcomp:gran:{b}")
                      for b, _ in GRAN_BINS if b in gran_rows}
    by_complexity = {c: slice_stats(cx_rows[c], f"raptorcomp:cx:{c}")
                     for c in COMPLEXITY_ORDER if c in cx_rows}

    def diff_entry(rows_a, rows_b, tag) -> dict:
        d = mean(rows_a) - mean(rows_b)
        lo, hi = boot_diff_ci(rows_a, rows_b, tag)
        return {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}

    differences = {
        "granularity_detail_minus_gist": diff_entry(
            gran_rows["detail"], gran_rows["gist"], "raptorcomp:diff:gran:detail-gist"),
        "granularity_mid_minus_gist": diff_entry(
            gran_rows["mid"], gran_rows["gist"], "raptorcomp:diff:gran:mid-gist"),
        "granularity_detail_minus_mid": diff_entry(
            gran_rows["detail"], gran_rows["mid"], "raptorcomp:diff:gran:detail-mid"),
        "complexity_dtl_minus_sh": diff_entry(
            cx_rows["dtl"], cx_rows["sh"], "raptorcomp:diff:cx:dtl-sh"),
    }

    # Tree shape per dataset (context for the QASPER control).
    tree_stats = {}
    for ds in sorted({d["dataset"] for d in documents}):
        docs = [d for d in documents if d["dataset"] == ds]
        depth_hist: dict[str, int] = defaultdict(int)
        for d in docs:
            depth_hist[str(d["tree_num_layers"] + 1)] += 1
        tree_stats[ds] = {
            "n_docs": len(docs),
            "tree_depth_histogram_layers": dict(sorted(depth_hist.items())),
            "mean_leaf_nodes": round(sum(
                int(d["tree_nodes_by_layer"].get("0", 0)) for d in docs) / len(docs), 1),
            "mean_summary_nodes": round(sum(
                sum(v for l, v in d["tree_nodes_by_layer"].items() if l != "0")
                for d in docs) / len(docs), 1),
        }

    out = {
        "generated_by": "code/scripts/raptor_retrieval_composition.py "
                        "(deterministic; no model in the loop; retrieval "
                        "replayed from cached trees + cached query embeddings)",
        "input_manifest": {
            "novelqa_questions": {"path": str(nq_questions_path.relative_to(ROOT)),
                                  "sha256_12": sha12(nq_questions_path)},
            "tree_cache_root": str(TREE_CACHE.relative_to(ROOT)),
            "embed_cache_root": str(EMBED_CACHE.relative_to(ROOT)),
            "main_study_code_version_hash": MAIN_STUDY_CODE_HASH,
        },
        "retrieval_params": {
            "collapse_tree": _RAPTOR_DEFAULTS["collapse_tree"],
            "top_k": _RAPTOR_DEFAULTS["tr_top_k"],
            "max_tokens": _RAPTOR_DEFAULTS["retrieval_max_tokens"],
            "selection_mode": _RAPTOR_DEFAULTS["tr_selection_mode"],
            "clustering_seed": CLUSTERING_SEED,
            "embedder_model": EMBED_MODEL,
            "summary_model": SUMMARY_MODEL,
            "tokenizer": "tiktoken cl100k_base (TreeRetriever default; the "
                         "run's own budget tokenizer)",
        },
        "definitions": {
            "leaf": "tree node at layer 0 (verbatim ~100-token document chunk)",
            "summary": "tree node at layer >= 1 (LLM-written cluster summary)",
            "summary_token_fraction": "tokens of retrieved summary nodes / "
                                      "tokens of all retrieved nodes, per "
                                      "question (cl100k_base on raw node text, "
                                      "the same count the 2000-token retrieval "
                                      "budget uses)",
            "granularity_bins": {b: mem for b, mem in GRAN_BINS},
            "complexity_labels": {"sh": "single-hop", "mh": "multi-hop",
                                  "dtl": "detail",
                                  "source": "data/novelqa/questions.jsonl "
                                            "Complexity field"},
            "one_retrieval_per_question": "retrieval is deterministic given "
                                          "tree + query embedding; the five "
                                          "answer repeats share it",
            "bootstrap": {"resamples": BOOT_RESAMPLES, "seed": BOOT_SEED,
                          "interval": "95% percentile, document-clustered "
                                      "(paired shared resample for differences)",
                          "cluster": {"novelqa": "novel", "qasper": "paper"}},
        },
        "coverage": {
            "eval_questions": len(items),
            "questions_analyzed": len(per_question),
            "questions_skipped_no_tree": skipped_questions,
            "docs_total": len(doc_keys),
            "docs_analyzed": len(documents),
            "docs_missing_tree": missing_docs,
            "query_embeddings_missing_from_cache_at_start": len(uncached),
        },
        "per_dataset": per_dataset,
        "novelqa_by_granularity": by_granularity,
        "novelqa_by_complexity": by_complexity,
        "differences_summary_token_fraction": differences,
        "tree_stats": tree_stats,
        "documents": documents,
        "per_question": per_question,
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"[raptor-comp] per-dataset mean summary-token fraction: "
        + ", ".join(f"{ds} {per_dataset[ds]['mean_summary_token_fraction']:.4f} "
                    f"(n={per_dataset[ds]['n']})" for ds in sorted(per_dataset)))
    log("[raptor-comp] novelqa granularity: "
        + ", ".join(f"{b} {by_granularity[b]['mean_summary_token_fraction']:.4f} "
                    f"[{by_granularity[b]['ci95'][0]:.4f}, {by_granularity[b]['ci95'][1]:.4f}] "
                    f"(n={by_granularity[b]['n']})" for b, _ in GRAN_BINS if b in by_granularity))
    log("[raptor-comp] novelqa complexity: "
        + ", ".join(f"{c} {by_complexity[c]['mean_summary_token_fraction']:.4f} "
                    f"[{by_complexity[c]['ci95'][0]:.4f}, {by_complexity[c]['ci95'][1]:.4f}] "
                    f"(n={by_complexity[c]['n']})" for c in COMPLEXITY_ORDER if c in by_complexity))
    log("[raptor-comp] differences: "
        + "; ".join(f"{name} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                    for name, d in differences.items()))
    log(f"[raptor-comp] wrote {args.out.relative_to(ROOT)} "
        f"({len(per_question)} question rows, {len(documents)} docs) "
        f"in {time.monotonic() - t0:.0f}s")
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
