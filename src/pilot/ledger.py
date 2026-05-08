"""Cost ledger: per-call JSONL append-only writer with a context manager.

Per pilot plan § 4.1 and § 9.3, every LLM/embedding call writes one
JSONL row. The schema captures cost fields, provenance fields for
replay, and reproducibility fields (prompt_hash, response_hash,
provider_request_id, model_id, seed, temperature).

Cost-attribution rule (§ 5.8 row #13): only `run_index == 0` rows
contribute to deployment cost. Runs 2..N feed variance estimation only.
The ledger writes all rows; the price_card resolver does the filtering.

Storage attribution is intentionally simplified: no per-call
`storage_delta_bytes` field (per pilot plan § 4.1). C_store is
computed as a back-of-envelope per-architecture footprint estimate
× study_horizon_days × storage_rate. See price_card.compute().
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator


class Stage(str, Enum):
    """Logical stage of an LLM call."""

    PREPROCESS = "preprocess"   # RAPTOR summarisation, GraphRAG entity extraction
    RETRIEVAL = "retrieval"     # encoder embedding, similarity search
    GENERATE = "generate"       # final answerer call


@dataclass
class CallRecord:
    """One row of the cost ledger.

    Caller fills the cost fields and provider_request_id during the
    `log_call` block; the ledger fills timestamp, run_index, and
    wallclock_s automatically.
    """

    timestamp: str = ""
    architecture: str = ""
    stage: str = ""
    model: str = ""
    run_index: int = 0
    uncached_input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    wallclock_s: float = 0.0
    gpu_s_estimate: float = 0.0
    provider_request_id: str | None = None
    prompt_hash: str = ""
    response_hash: str = ""
    seed: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    provider_region: str | None = None
    failed: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None or k in {"provider_request_id"}}


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CostLedger:
    """Append-only JSONL ledger for one experiment run.

    One CostLedger instance per (architecture, run_id) pair. Multiple
    architectures share the same run_id but write to separate files
    via separate ledger instances; or one ledger with `architecture=`
    on each call.

    Files land at `<root>/<run_id>/ledger.jsonl`. Created on first
    write; never overwritten.
    """

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.root = root if root is not None else Path("outputs/runs")
        self.run_dir = self.root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "ledger.jsonl"

    @contextmanager
    def log_call(
        self,
        *,
        architecture: str,
        stage: Stage | str,
        model: str,
        prompt: str,
        run_index: int = 0,
        seed: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> Iterator[CallRecord]:
        """Context manager yielding a CallRecord to populate during the call.

        Usage:
            with ledger.log_call(architecture="flat", stage=Stage.GENERATE,
                                 model="claude-sonnet-4-6-20260217",
                                 prompt=rendered) as rec:
                response = client.messages.create(...)
                rec.uncached_input_tokens = response.usage.input_tokens
                rec.cached_input_tokens = response.usage.cache_read_input_tokens or 0
                rec.output_tokens = response.usage.output_tokens
                rec.provider_request_id = response.id
                rec.response_hash = sha256_hex(response.content[0].text)

        On exit, timestamp + wallclock_s are filled and the row is
        flushed as JSONL. Exceptions inside the block mark the row
        as `failed=true` with the exception's repr; the row is still
        written.
        """
        rec = CallRecord(
            architecture=architecture,
            stage=stage.value if isinstance(stage, Stage) else stage,
            model=model,
            run_index=run_index,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_hash=sha256_hex(prompt),
        )
        start = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec.failed = True
            rec.failure_reason = repr(exc)
            self._write_row(rec, start)
            raise
        self._write_row(rec, start)

    def _write_row(self, rec: CallRecord, start: float) -> None:
        rec.timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        rec.wallclock_s = round(time.perf_counter() - start, 6)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), separators=(",", ":")))
            f.write("\n")
            f.flush()
            # fsync so a power-loss after the LLM call (which we
            # already paid for) keeps the cost row durable on disk.
            # Cost: ~1ms per row; rows happen at LLM-call cadence
            # which is much slower, so the wall-clock impact is
            # negligible. Swallow OSError on platforms without fsync
            # so the ledger stays usable.
            try:
                import os as _os
                _os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass

    def read(self) -> list[CallRecord]:
        """Read all rows back as CallRecord instances. Used for tests + replay."""
        if not self.path.exists():
            return []
        rows: list[CallRecord] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append(CallRecord(**{k: d.get(k) for k in CallRecord.__dataclass_fields__}))
        return rows


def new_run_id() -> str:
    """Generate a new run id: `<utc-iso-date>-<short-uuid>`."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    short = uuid.uuid4().hex[:8]
    return f"{today}-{short}"
