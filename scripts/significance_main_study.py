"""Headline statistical significance for the four-architecture benchmark.

Per dataset (QASPER, NovelQA) this computes:

  1. Per-architecture point estimate (mean over questions): QASPER mean
     Answer-F1, NovelQA accuracy.
  2. 95% CI per architecture via a CLUSTERED bootstrap. The resampling unit is
     the cluster (paper_id for QASPER, novel_id for NovelQA), not the individual
     question, because questions within a paper/novel are not independent. Each
     of the 10,000 resamples draws clusters with replacement and recomputes the
     mean over every question in the drawn clusters; the CI is the 2.5th/97.5th
     percentile of that bootstrap distribution.
  3. All pairwise PAIRED comparisons of interest (flat vs each retrieval
     architecture, plus naive_rag vs raptor). For each question present under
     both architectures we take the paired difference A - B on the metric, then
     clustered-bootstrap the mean paired difference. A two-sided bootstrap
     p-value is 2 * min(frac > 0, frac < 0), clamped to [0, 1].

The five per-question repeats are deterministic at T=0, so each stored metric is
already the repeat value; the only meaningful uncertainty is the question-level
clustered variance, which is exactly what the cluster bootstrap captures.

Deterministic and non-interactive: a single fixed seed drives every resample and
is recorded in the JSON output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEED = 20260616
N_RESAMPLES = 10_000

ARCHS = ["flat", "naive_rag", "raptor", "graphrag"]

# Pairwise comparisons to report: flat vs every retrieval architecture, plus the
# near-tied naive_rag vs raptor contrast that motivates a formal test.
PAIRS = [
    ("flat", "naive_rag"),
    ("flat", "raptor"),
    ("flat", "graphrag"),
    ("naive_rag", "raptor"),
]

DATASETS = ["qasper", "novelqa"]

CODE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = CODE_DIR / "outputs" / "main_study" / "scored_cells.jsonl"
OUT_DIR = CODE_DIR / "outputs" / "main_study"
JSON_PATH = OUT_DIR / "significance.json"
TEX_PATH = OUT_DIR / "significance_table.tex"

# Expected point estimates used as a guard rail (QASPER mean F1 / NovelQA acc).
# NovelQA values are over the held-out test pool (55 novels): the 4 calibration
# novels are excluded per data/novelqa/calibration_novels.json, the 2.5M-token
# outlier B48 is never built, and Frankenstein (B30) is dropped for lack of
# recoverable Codabench gold. Les Mis\'erables (B42) is included after the
# accent-aware title-join fix (extract_score._norm_title).
EXPECTED = {
    "qasper": {
        "flat": 0.4573,
        "naive_rag": 0.4442,
        "raptor": 0.4130,
        "graphrag": 0.4025,
    },
    "novelqa": {
        "flat": 0.7574,
        "naive_rag": 0.6539,
        "raptor": 0.6459,
        "graphrag": 0.5996,
    },
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset_arrays(rows, dataset):
    """Return aligned per-question arrays for one dataset.

    Output:
      qids      : list of question ids in a stable order
      cluster_id: int array (len n_questions) mapping question -> cluster index
      clusters  : list of cluster labels (index == cluster_id value)
      metrics   : dict arch -> float array (len n_questions); NaN where the arch
                  is missing for that question.
    """
    sub = [r for r in rows if r["dataset"] == dataset]

    # Stable question ordering (first appearance) and its cluster.
    qid_order = []
    qid_cluster = {}
    for r in sub:
        qid = r["qid"]
        if qid not in qid_cluster:
            qid_cluster[qid] = r["cluster"]
            qid_order.append(qid)

    qid_index = {q: i for i, q in enumerate(qid_order)}
    n_q = len(qid_order)

    # Cluster labels -> contiguous integer ids (first appearance order).
    clusters = []
    cluster_index = {}
    for q in qid_order:
        c = qid_cluster[q]
        if c not in cluster_index:
            cluster_index[c] = len(clusters)
            clusters.append(c)
    cluster_id = np.array([cluster_index[qid_cluster[q]] for q in qid_order], dtype=np.int64)

    metrics = {a: np.full(n_q, np.nan, dtype=np.float64) for a in ARCHS}
    for r in sub:
        a = r["arch"]
        if a in metrics:
            metrics[a][qid_index[r["qid"]]] = float(r["metric"])

    return {
        "qids": qid_order,
        "cluster_id": cluster_id,
        "clusters": clusters,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Clustered bootstrap machinery
# --------------------------------------------------------------------------- #


def make_cluster_resamples(cluster_id, n_clusters, n_resamples, rng):
    """Precompute, for each resample, the per-question multiplicity weights.

    We draw `n_clusters` clusters with replacement. A question's weight in a
    resample is the number of times its cluster was drawn. Working with integer
    weights lets every per-arch / per-pair statistic reuse the same draw, so the
    bootstrap is a handful of vectorised matmuls rather than a Python loop over
    10k iterations.

    Returns weights with shape (n_resamples, n_questions), dtype float64.
    """
    # draws[r, k] = index of the k-th cluster drawn in resample r.
    draws = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    # Count cluster multiplicities per resample: counts[r, c].
    counts = np.zeros((n_resamples, n_clusters), dtype=np.int32)
    # np.add.at over a flattened (resample, cluster) index space.
    rep_rows = np.repeat(np.arange(n_resamples), n_clusters)
    np.add.at(counts, (rep_rows, draws.ravel()), 1)

    # Map cluster counts onto questions via the fixed question->cluster index.
    weights = counts[:, cluster_id].astype(np.float64)  # (n_resamples, n_q)
    return weights


def weighted_means(values, weights, mask):
    """Bootstrap distribution of the weighted mean of `values`.

    values  : (n_q,) per-question metric (may contain NaN where masked out)
    weights : (n_resamples, n_q) cluster-draw multiplicities
    mask    : (n_q,) bool, True for questions that contribute
    Returns (n_resamples,) array of resampled means.
    """
    v = np.where(mask, values, 0.0)
    w = weights * mask  # zero-out masked questions
    num = w @ v                       # (n_resamples,)
    den = w.sum(axis=1)               # (n_resamples,)
    # den is effectively always > 0 (whole clusters of valid questions), but be
    # defensive against an all-empty draw.
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def percentile_ci(samples, low=2.5, high=97.5):
    return float(np.percentile(samples, low)), float(np.percentile(samples, high))


# --------------------------------------------------------------------------- #
# Per-dataset computation
# --------------------------------------------------------------------------- #


def compute_dataset(data, dataset):
    cluster_id = data["cluster_id"]
    n_clusters = len(data["clusters"])
    metrics = data["metrics"]

    # One fixed RNG per dataset, derived deterministically from the global seed.
    rng = np.random.default_rng(SEED + DATASETS.index(dataset))
    weights = make_cluster_resamples(cluster_id, n_clusters, N_RESAMPLES, rng)

    # ---- per-arch point estimates + CIs -------------------------------------
    per_arch = {}
    for a in ARCHS:
        v = metrics[a]
        mask = ~np.isnan(v)
        point = float(np.mean(v[mask]))
        boot = weighted_means(v, weights, mask)
        ci_low, ci_high = percentile_ci(boot)
        # number of distinct clusters contributing valid questions for this arch
        n_clu = int(np.unique(cluster_id[mask]).size)
        per_arch[a] = {
            "mean": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_questions": int(mask.sum()),
            "n_clusters": n_clu,
        }

    # ---- pairwise paired comparisons ----------------------------------------
    pairwise = []
    for a, b in PAIRS:
        va, vb = metrics[a], metrics[b]
        mask = ~np.isnan(va) & ~np.isnan(vb)  # questions present for both
        diff = np.where(mask, va - vb, 0.0)
        point = float(diff[mask].mean())

        boot = weighted_means(diff, weights, mask)
        ci_low, ci_high = percentile_ci(boot)

        frac_gt = float(np.mean(boot > 0.0))
        frac_lt = float(np.mean(boot < 0.0))
        p_value = min(1.0, 2.0 * min(frac_gt, frac_lt))

        pairwise.append({
            "a": a,
            "b": b,
            "mean_diff": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
        })

    return {"per_arch": per_arch, "pairwise": pairwise}


# --------------------------------------------------------------------------- #
# Verification guard rail
# --------------------------------------------------------------------------- #


def verify_point_estimates(results):
    problems = []
    for ds in DATASETS:
        for a in ARCHS:
            got = results[ds]["per_arch"][a]["mean"]
            exp = EXPECTED[ds][a]
            if abs(got - exp) > 5e-4:
                problems.append(f"{ds}/{a}: got {got:.6f}, expected {exp:.4f}")
    if problems:
        raise SystemExit("Point-estimate verification FAILED:\n  " + "\n  ".join(problems))
    print("Point-estimate verification PASSED (all archs match within 5e-4).")


# --------------------------------------------------------------------------- #
# LaTeX table
# --------------------------------------------------------------------------- #

ARCH_LABEL = {
    "flat": "Flat",
    "naive_rag": "Naive RAG",
    "raptor": "RAPTOR",
    "graphrag": "GraphRAG",
}


def sig_marker(results, dataset, arch):
    """Significance marker for `arch` vs flat (the reference) on `dataset`."""
    if arch == "flat":
        return ""
    for p in results[dataset]["pairwise"]:
        if p["a"] == "flat" and p["b"] == arch:
            return r"$^{*}$" if p["significant"] else ""
    return ""


def fmt_cell(entry, marker):
    return (
        f"{entry['mean']:.4f}{marker} "
        f"\\textsubscript{{[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]}}"
    )


def write_latex(results):
    lines = []
    lines.append("% Auto-generated by code/scripts/significance_main_study.py -- do not edit by hand.")
    lines.append("% Per-architecture point estimate with 95% clustered-bootstrap CI.")
    lines.append("% QASPER column: mean Answer-F1. NovelQA column: accuracy.")
    lines.append(f"% Seed={SEED}, resamples={N_RESAMPLES}. $^{{*}}$ marks p<0.05 vs Flat (paired clustered bootstrap).")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\hline")
    lines.append(r"Architecture & QASPER F1 & NovelQA Acc. \\")
    lines.append(r" & {\footnotesize mean [95\% CI]} & {\footnotesize mean [95\% CI]} \\")
    lines.append(r"\hline")
    for a in ARCHS:
        q = results["qasper"]["per_arch"][a]
        n = results["novelqa"]["per_arch"][a]
        qcell = fmt_cell(q, sig_marker(results, "qasper", a))
        ncell = fmt_cell(n, sig_marker(results, "novelqa", a))
        lines.append(f"{ARCH_LABEL[a]} & {qcell} & {ncell} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    rows = load_rows(INPUT_PATH)
    results = {}
    for ds in DATASETS:
        data = build_dataset_arrays(rows, ds)
        results[ds] = compute_dataset(data, ds)

    verify_point_estimates(results)

    payload = {
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "ci_method": "clustered_bootstrap_percentile_95",
        "cluster_unit": {"qasper": "paper_id", "novelqa": "novel_id"},
        "metric": {"qasper": "answer_f1", "novelqa": "accuracy"},
        "datasets": results,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_latex(results)

    # Console summary.
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        for a in ARCHS:
            e = results[ds]["per_arch"][a]
            print(f"  {a:10s} mean={e['mean']:.4f}  "
                  f"95% CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]  "
                  f"n_q={e['n_questions']} n_clusters={e['n_clusters']}")
        print("  pairwise:")
        for p in results[ds]["pairwise"]:
            tag = "SIG" if p["significant"] else "ns "
            print(f"    [{tag}] {p['a']:10s} - {p['b']:10s} "
                  f"diff={p['mean_diff']:+.4f}  "
                  f"CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  "
                  f"p={p['p_value']:.4f}")

    print(f"\nWrote {JSON_PATH}")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
