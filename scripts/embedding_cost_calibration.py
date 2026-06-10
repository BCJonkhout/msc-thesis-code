"""Estimate the embedding GPU cost = ledger token volume x measured eval rate.

The ledger's embed WALLCLOCK cannot be used as GPU time. RAPTOR embeds run
in a thread pool whose calls all queue behind the single embed semaphore
(the B41 saturation fix), so each row's wallclock includes its queue-wait,
and summing those overlapping waits overcounts massively (observed ~6 s per
bge-m3 embed in a real run -- ~50-100x too high; the non-threaded naive_rag
path, by contrast, looks sane). So we split the estimate:

  - token VOLUME per architecture comes from the ledger (reliable), and
  - the per-token GPU EVAL rate is measured on a small sample re-embedded
    through an IDLE Ollama (Ollama's native total_duration - load_duration),
    at each architecture's representative chunk size.

  embedding GPU seconds = token_volume x eval_rate ; cost = x the GPU $/s rate.

The ledger wallclock sum is reported only as a contaminated upper bound. The
Gemini rows (answerer / summary / extraction) carry real API token counts
and are summed unchanged.

Run AFTER the study, with --ollama-check, against an IDLE Ollama:

    python scripts/embedding_cost_calibration.py --run-dir <run> --ollama-check

Without --ollama-check it only reads the ledger (safe anytime): it reports
the per-architecture token volume and the wallclock upper bound, and tells
you to re-run with --ollama-check on an idle Ollama for the real number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.ledger import CallRecord  # noqa: E402
from pilot.price_card import load_price_card, _row_cost_usd  # noqa: E402

DEFAULT_EMBEDDER = "bge-m3"
# Representative embed token size per architecture (dominant embed workload).
ARCH_BUCKET = {"raptor": 100, "naive_rag": 384, "graphrag": 600}


def gpu_rate_per_s(pc: dict[str, Any]) -> float:
    g = pc.get("gpu") or {}
    rate = g.get("usd_per_second") or g.get("h100_usd_per_second")
    if rate is None:
        per_hour = g.get("usd_per_hour") or g.get("h100_usd_per_hour")
        if isinstance(per_hour, dict):
            per_hour = per_hour.get("value")
        rate = (per_hour / 3600.0) if per_hour else 0.0
    return float(rate)


def load_ledger_rows(ledger_path: Path) -> tuple[list[dict], int]:
    """Parse a JSONL ledger robustly into (rows, n_malformed_fragments).

    The ledger is appended by multiple threads (RAPTOR's leaf-embed pool),
    whose writes can interleave or concatenate on a single line. We decode
    each line with raw_decode in a loop -- recovering cleanly-concatenated
    objects -- and count fragments we cannot recover rather than crashing.
    """
    dec = json.JSONDecoder()
    rows: list[dict] = []
    malformed = 0
    for raw in ledger_path.read_text("utf-8").splitlines():
        s = raw.strip()
        pos, n = 0, len(s)
        while pos < n:
            while pos < n and s[pos].isspace():
                pos += 1
            if pos >= n:
                break
            try:
                obj, end = dec.raw_decode(s, pos)
            except json.JSONDecodeError:
                malformed += 1
                break  # unrecoverable interleaved fragment; skip rest of line
            if isinstance(obj, dict):
                rows.append(obj)
            pos = end
    return rows, malformed


def token_volume_from_ledger(
    rows: list[dict], embedder_model: str
) -> dict[str, dict[str, float]]:
    """Per-architecture embed token volume (char/4) + a contaminated
    wallclock sum (upper bound only)."""
    per_arch: dict[str, dict[str, float]] = defaultdict(
        lambda: {"char4_tokens": 0.0, "embed_rows": 0, "wallclock_sum_s": 0.0}
    )
    for d in rows:
        if d.get("run_index", 0) != 0 or d.get("model") != embedder_model:
            continue
        a = per_arch[d.get("architecture", "?")]
        a["char4_tokens"] += float(d.get("uncached_input_tokens", 0) or 0)
        a["embed_rows"] += 1
        a["wallclock_sum_s"] += float(d.get("wallclock_s", 0.0) or 0.0)
    return dict(per_arch)


def non_embedding_cost(rows: list[dict], embedder_model: str, pc: dict[str, Any]) -> float:
    total = 0.0
    for d in rows:
        if d.get("run_index", 0) != 0 or d.get("model") == embedder_model:
            continue
        row = CallRecord(**{k: d.get(k) for k in CallRecord.__dataclass_fields__})
        total += _row_cost_usd(row, pc)
    return total


def _sample_texts(data_root: Path, target_tokens: int, n: int) -> list[str]:
    """~n texts of ~target_tokens tokens, windowed from the real corpus."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    sources: list[str] = []
    qdev = data_root / "qasper" / "dev.jsonl"
    if qdev.exists():
        for line in qdev.read_text("utf-8").splitlines()[:30]:
            if not line.strip():
                continue
            p = json.loads(line)
            parts = [p.get("title") or "", p.get("abstract") or ""]
            for sec in p.get("full_text", []) or []:
                for para in sec.get("paragraphs", []) or []:
                    if isinstance(para, str) and para.strip():
                        parts.append(para)
            sources.append("\n\n".join(parts))
    ft = data_root / "novelqa" / "full_texts"
    if ft.exists():
        for f in sorted(ft.glob("*.txt"))[:3]:
            sources.append(f.read_text("utf-8", errors="ignore"))
    texts: list[str] = []
    for src in sources:
        toks = enc.encode(src)
        for i in range(0, len(toks), target_tokens):
            if len(texts) >= n:
                return texts
            c = enc.decode(toks[i : i + target_tokens]).strip()
            if c:
                texts.append(c)
    return texts


def measure_eval_rates(
    ollama_url: str, model: str, data_root: Path, n_per_size: int
) -> dict[int, dict[str, float]]:
    """Measure GPU eval-seconds per char/4 token at each chunk size on an
    IDLE Ollama, via Ollama's native total_duration - load_duration."""
    import httpx

    out: dict[int, dict[str, float]] = {}
    with httpx.Client(base_url=ollama_url, timeout=120.0) as client:
        for size in sorted(set(ARCH_BUCKET.values())):
            texts = _sample_texts(data_root, size, n_per_size)
            sum_eval = sum_wall = sum_char4 = 0.0
            ok = 0
            for t in texts:
                t0 = time.perf_counter()
                try:
                    r = client.post("/api/embed", json={"model": model, "input": t})
                    wall = time.perf_counter() - t0
                    if r.status_code != 200:
                        continue
                    dd = r.json()
                    eval_s = max(
                        0.0,
                        (int(dd.get("total_duration", 0)) - int(dd.get("load_duration", 0))) / 1e9,
                    )
                except Exception:
                    continue
                sum_eval += eval_s
                sum_wall += wall
                sum_char4 += max(1, len(t) // 4)
                ok += 1
                time.sleep(0.02)
            out[size] = {
                "samples": ok,
                "eval_s_per_char4": (sum_eval / sum_char4) if sum_char4 else 0.0,
                "mean_eval_ms": (1000 * sum_eval / ok) if ok else 0.0,
                "mean_wallclock_ms": (1000 * sum_wall / ok) if ok else 0.0,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--embedder-model", default=DEFAULT_EMBEDDER)
    ap.add_argument("--ollama-check", action="store_true",
                    help="measure the real per-token GPU eval rate on an IDLE Ollama (required for the real cost).")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--data-root", type=Path, default=CODE_ROOT / "data")
    ap.add_argument("--sample-per-size", type=int, default=150)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ledger_path = args.run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        print(f"no ledger at {ledger_path}", file=sys.stderr)
        return 1

    pc = load_price_card(CODE_ROOT / "configs" / "price_card.yaml")
    rate = gpu_rate_per_s(pc)

    rows, malformed = load_ledger_rows(ledger_path)
    vols = token_volume_from_ledger(rows, args.embedder_model)
    nonembed = non_embedding_cost(rows, args.embedder_model, pc)

    wallclock_upper_s = sum(v["wallclock_sum_s"] for v in vols.values())

    report: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "embedder_model": args.embedder_model,
        "gpu_usd_per_second": rate,
        "malformed_ledger_fragments": malformed,
        "non_embedding_cost_usd": round(nonembed, 6),
        "per_architecture_token_volume": {
            a: {"embed_rows": int(v["embed_rows"]), "char4_tokens": round(v["char4_tokens"])}
            for a, v in sorted(vols.items())
        },
        "ledger_wallclock_upper_bound": {
            "seconds": round(wallclock_upper_s, 1),
            "usd": round(wallclock_upper_s * rate, 6),
            "note": "CONTAMINATED by the embed-semaphore queue-wait in threaded (raptor/graphrag) embeds; upper bound only, NOT the real GPU cost.",
        },
    }

    if args.ollama_check:
        rates = measure_eval_rates(args.ollama_url, args.embedder_model, args.data_root, args.sample_per_size)
        per_arch = {}
        corr_total_s = 0.0
        for a, v in sorted(vols.items()):
            size = ARCH_BUCKET.get(a, 384)
            r = rates.get(size, {}).get("eval_s_per_char4", 0.0)
            gpu_s = v["char4_tokens"] * r
            corr_total_s += gpu_s
            per_arch[a] = {
                "char4_tokens": round(v["char4_tokens"]),
                "eval_s_per_char4": round(r, 8),
                "gpu_seconds": round(gpu_s, 2),
                "gpu_usd": round(gpu_s * rate, 6),
            }
        report["measured_eval_rates"] = rates
        report["embedding_gpu_cost"] = {
            "per_architecture": per_arch,
            "total_gpu_seconds": round(corr_total_s, 2),
            "total_gpu_usd": round(corr_total_s * rate, 6),
        }
        report["total_cost_usd"] = round(nonembed + corr_total_s * rate, 6)
    else:
        report["next_step"] = (
            "Re-run with --ollama-check on an IDLE Ollama to measure the real "
            "per-token GPU eval rate and compute the embedding GPU cost. The "
            "ledger wallclock above is only a contaminated upper bound."
        )

    out = args.out or (args.run_dir / "embedding_cost_calibration.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "measured_eval_rates"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
