"""Ollama HTTP-API embedding wrapper.

The pilot's default encoder (BGE-M3 per pilot plan § 5.8 row #9) is
served locally via Ollama at the standard `http://localhost:11434`
endpoint. This wrapper exposes a small uniform `Embedder` surface so
that other code (the Recall@k experiment, the Naive RAG runtime,
RAPTOR's leaf embedding pass) can swap encoders by changing one
line in the config.

Concurrency note
----------------
Ollama serves embeddings single-threaded by default (and the pilot
pins ``OMP_NUM_THREADS=1``). RAPTOR builds leaf embeddings through an
*unbounded* ``ThreadPoolExecutor`` and several build processes can run
in parallel across provider lanes; without a cap this saturates Ollama
and it returns sustained HTTP 500s (the deterministic "B41" build
failure — the root cause was embed-load saturation, not chunk content).
Two defences live here: a process-global semaphore that serialises this
process's embed POSTs (``OLLAMA_MAX_CONCURRENCY``, default 1), and a
retry policy wide enough to ride out transient saturation. Cross-process
load is additionally bounded by setting ``OLLAMA_NUM_PARALLEL=1`` on the
server and running builds sequentially.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx


# Bound concurrent embed POSTs from THIS process into Ollama. RAPTOR's
# leaf-embed thread pool and Naive RAG/GraphRAG all funnel through here,
# so a single shared semaphore serialises every embedding request the
# process makes regardless of how many threads issue them.
_EMBED_CONCURRENCY = max(1, int(os.environ.get("OLLAMA_MAX_CONCURRENCY", "1")))
_EMBED_SEMAPHORE = threading.Semaphore(_EMBED_CONCURRENCY)

# Retry budget for transient failures (sustained-load 500s, transport
# disconnects, read timeouts). Wider than a couple of attempts because
# under saturation the server can 500 for several seconds before draining.
_MAX_ATTEMPTS = max(1, int(os.environ.get("OLLAMA_EMBED_MAX_ATTEMPTS", "8")))

# Optional content-addressed embedding cache. When OLLAMA_EMBED_CACHE_DIR
# is set (or a cache_dir is passed to the embedder), each (model, text)
# embedding is stored under <dir>/<model>/<h[:2]>/<h>.json so a crash
# mid-build re-embeds only the unfinished tail instead of re-embedding the
# whole document. Keyed by content hash, so a hit is exact and the store
# is concurrency-safe across parallel build lanes (atomic write,
# last-writer-wins on byte-identical content).
_EMBED_CACHE_DIR = os.environ.get("OLLAMA_EMBED_CACHE_DIR")


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"


def _backoff_sleep(attempt: int) -> None:
    """Capped exponential backoff with jitter."""
    time.sleep(min(2 ** attempt, 30) * 0.5 + random.uniform(0, 0.5))


@dataclass(frozen=True)
class EmbeddingResult:
    """One model + one batch of embeddings, plus the model id used."""
    model: str
    embeddings: list[list[float]]


class OllamaEmbedder:
    """Embedding client for an Ollama server's `/api/embed` endpoint.

    The endpoint accepts either a single string or a list of strings
    in the `input` field; this wrapper always passes lists for batch
    efficiency. The server returns `embeddings` as a list of vectors
    in the same order.

    The server URL defaults to the canonical Ollama localhost port.
    A custom URL can be provided for remote-Ollama or self-hosted
    setups via the `base_url` kwarg or the `OLLAMA_HOST` env var.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        url = base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        # Allow OLLAMA_HOST to be a bare host:port (Ollama's own convention)
        # or a fully-qualified URL.
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        self.base_url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        # Content-addressed embedding cache (None = disabled). Explicit
        # cache_dir wins; otherwise fall back to OLLAMA_EMBED_CACHE_DIR.
        _cache = cache_dir if cache_dir is not None else _EMBED_CACHE_DIR
        self._cache_dir = Path(_cache) if _cache else None
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def _request_embed(self, texts: list[str]) -> list[list[float]]:
        """POST one batch with retry on transient 5xx / transport errors.

        Retries cover Ollama's sustained-load 500s and httpx transport /
        timeout blips with capped exponential backoff + jitter. A 404
        (model not pulled) and any non-5xx 4xx fail fast. The embed POST is
        guarded by a process-global semaphore so concurrent RAPTOR leaf
        threads cannot saturate a single-threaded Ollama.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with _EMBED_SEMAPHORE:
                    response = self._client.post(
                        "/api/embed",
                        json={"model": self.model, "input": texts},
                    )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                # Connection resets / read timeouts under load are transient.
                last_exc = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    _backoff_sleep(attempt)
                    continue
                raise
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model {self.model!r}. "
                    f"Run `ollama pull {self.model}` and retry."
                )
            if 500 <= response.status_code < 600:
                last_exc = httpx.HTTPStatusError(
                    f"server {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt < _MAX_ATTEMPTS - 1:
                    _backoff_sleep(attempt)
                    continue
                raise last_exc
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Unexpected /api/embed response shape: keys={list(data.keys())}"
                )
            return [list(map(float, vec)) for vec in embeddings]
        # Loop exhausted without returning (all attempts were retryable).
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("embed retry loop exhausted without a response")

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, isolating a genuinely-bad item via bisection.

        If a multi-item batch fails after the full retry budget (e.g. one
        pathological chunk the server rejects deterministically), split and
        retry the halves so a single bad item cannot sink an entire batch.
        A persistently-failing *single* item re-raises so a true outage
        still surfaces rather than masquerading as a hang.
        """
        if not texts:
            return []
        try:
            return self._request_embed(texts)
        except Exception:
            if len(texts) == 1:
                raise
            mid = len(texts) // 2
            left = self._post_embed(texts[:mid])
            right = self._post_embed(texts[mid:])
            return left + right

    # ── content-addressed embedding cache ──────────────────────────────
    def _cache_file(self, text: str) -> Path:
        h = hashlib.sha256(
            f"{self.model}\x00{text}".encode("utf-8")
        ).hexdigest()
        return self._cache_dir / _model_slug(self.model) / h[:2] / f"{h}.json"

    def _cache_get(self, text: str) -> list[float] | None:
        path = self._cache_file(text)
        if not path.exists():
            return None
        try:
            vec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # treat a corrupt entry as a miss
        if isinstance(vec, list):
            return [float(x) for x in vec]
        return None

    def _cache_put(self, text: str, vec: list[float]) -> None:
        path = self._cache_file(text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic, concurrency-safe: a unique tmp avoids collisions
            # across parallel lanes; os.replace is atomic; identical
            # content makes last-writer-wins harmless.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(vec), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # a cache-write failure must never break the embed

    def embed(self, texts: Iterable[str]) -> EmbeddingResult:
        """Embed a sequence of strings; returns vectors in input order.

        When a cache dir is configured, already-embedded (model, text)
        pairs are served from disk and only the misses are POSTed, so a
        crash mid-build re-embeds only the unfinished tail.
        """
        items = list(texts)
        vectors: list[list[float] | None] = [None] * len(items)

        # Resolve cache hits; collect misses preserving input positions.
        miss_positions: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(items):
            cached = self._cache_get(text) if self._cache_dir is not None else None
            if cached is not None:
                vectors[i] = cached
            else:
                miss_positions.append(i)
                miss_texts.append(text)

        # Embed the misses in batches, then backfill + persist.
        embedded: list[list[float]] = []
        for offset in range(0, len(miss_texts), self.batch_size):
            embedded.extend(self._post_embed(miss_texts[offset : offset + self.batch_size]))
        for k, pos in enumerate(miss_positions):
            vectors[pos] = embedded[k]
            if self._cache_dir is not None:
                self._cache_put(miss_texts[k], embedded[k])

        return EmbeddingResult(
            model=self.model, embeddings=[v for v in vectors if v is not None]
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaEmbedder":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
