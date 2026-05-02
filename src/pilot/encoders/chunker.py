"""Sentence-boundary-aware chunker.

Wraps `langchain-text-splitters`'s `RecursiveCharacterTextSplitter`
configured for tokenizer-aware sizing via `tiktoken`. The chunking
spec follows `configs/methods.yaml#naive_rag`:

  - chunk_size_tokens: 384  (lit:lewis2020retrieval)
  - overlap: 0              (lit:sarthi2024raptor)
  - chunker: sentence_boundary_aware  (lit:robertson2009probabilistic)

The recursive splitter walks a list of separators in priority order
(paragraph break → sentence-terminator → word → char) and splits at
the highest-priority boundary that keeps each chunk under
``chunk_size_tokens``. Token counts are measured by tiktoken's
``cl100k_base`` encoder; this is *not* the BGE-M3 tokenizer
exactly, but the discrepancy is well under 10% for English academic
text, and using a single fast deterministic tokenizer keeps the
chunker decoupled from the choice of downstream embedder.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text suitable for retrieval.

    The ``index`` field is the chunk's position in the source's
    chunk sequence; useful for downstream debugging and for stable
    deduplication when the same chunk appears in multiple top-k
    lists at different k.
    """
    text: str
    index: int


# Priority-ordered separators. The recursive splitter tries each
# separator in turn; once a chunk fits the size cap, it's emitted
# at that boundary. This is the canonical "paragraph → sentence →
# word" hierarchy used by every modern RAG splitter.
_SEPARATORS = [
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # sentence terminator
    "? ",
    "! ",
    "; ",     # clause break
    ", ",     # phrase break (last resort before word/char)
    " ",
    "",
]


class SentenceBoundaryChunker:
    """Sentence-boundary-aware chunker honouring methods.yaml#naive_rag."""

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 384,
        overlap_tokens: int = 0,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size_tokens,
            chunk_overlap=overlap_tokens,
            separators=_SEPARATORS,
        )

    def chunk(self, text: str) -> list[TextChunk]:
        """Return TextChunk records covering ``text`` in source order."""
        if not text or not text.strip():
            return []
        pieces = self._splitter.split_text(text)
        return [
            TextChunk(text=piece, index=i)
            for i, piece in enumerate(pieces)
            if piece.strip()
        ]
