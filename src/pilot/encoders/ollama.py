"""Ollama HTTP-API embedding wrapper.

The pilot's default encoder (BGE-M3 per pilot plan § 5.8 row #9) is
served locally via Ollama at the standard `http://localhost:11434`
endpoint. This wrapper exposes a small uniform `Embedder` surface so
that other code (the Recall@k experiment, the Naive RAG runtime,
RAPTOR's leaf embedding pass) can swap encoders by changing one
line in the config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import httpx


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
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def _post_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.post(
            "/api/embed",
            json={"model": self.model, "input": texts},
        )
        if response.status_code == 404:
            # Most common cause: model not pulled yet. Surface a
            # specific error so the user knows to run `ollama pull`.
            raise RuntimeError(
                f"Ollama returned 404 for model {self.model!r}. "
                f"Run `ollama pull {self.model}` and retry."
            )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Unexpected /api/embed response shape: keys={list(data.keys())}"
            )
        return [list(map(float, vec)) for vec in embeddings]

    def embed(self, texts: Iterable[str]) -> EmbeddingResult:
        """Embed a sequence of strings; returns vectors in input order."""
        items = list(texts)
        all_vectors: list[list[float]] = []
        for offset in range(0, len(items), self.batch_size):
            batch = items[offset : offset + self.batch_size]
            all_vectors.extend(self._post_embed(batch))
        return EmbeddingResult(model=self.model, embeddings=all_vectors)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaEmbedder":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
