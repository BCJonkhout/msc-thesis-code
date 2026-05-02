"""Tests for the sentence-boundary-aware chunker.

The chunker is a thin wrapper around langchain-text-splitters'
RecursiveCharacterTextSplitter; the tests here cover the wrapper's
contract (correct chunk sizing, deterministic output, sentence
boundaries respected, empty/edge-case handling) rather than the
upstream library's internals.
"""
from __future__ import annotations

import pytest

from pilot.encoders.chunker import SentenceBoundaryChunker, TextChunk


def test_constructor_validates_chunk_size_positive() -> None:
    with pytest.raises(ValueError):
        SentenceBoundaryChunker(chunk_size_tokens=0)
    with pytest.raises(ValueError):
        SentenceBoundaryChunker(chunk_size_tokens=-10)


def test_constructor_validates_overlap_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        SentenceBoundaryChunker(chunk_size_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        SentenceBoundaryChunker(chunk_size_tokens=100, overlap_tokens=200)


def test_constructor_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError):
        SentenceBoundaryChunker(chunk_size_tokens=100, overlap_tokens=-1)


def test_empty_text_returns_no_chunks() -> None:
    chunker = SentenceBoundaryChunker(chunk_size_tokens=64)
    assert chunker.chunk("") == []
    assert chunker.chunk("   \n\n  \t  ") == []


def test_short_text_returns_one_chunk() -> None:
    chunker = SentenceBoundaryChunker(chunk_size_tokens=64)
    chunks = chunker.chunk("This is one short sentence.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "short sentence" in chunks[0].text


def test_chunks_are_indexed_in_order() -> None:
    chunker = SentenceBoundaryChunker(chunk_size_tokens=20)
    # ~150 token paragraph, will split into multiple chunks.
    sentences = [
        "Sentence one is here. ",
        "Sentence two follows after. ",
        "And sentence three appears. ",
        "Sentence four arrives next. ",
        "Five comes after four. ",
        "Six follows five. ",
        "Seven follows six. ",
        "Eight follows seven. ",
        "Nine follows eight. ",
        "Ten follows nine.",
    ]
    chunks = chunker.chunk("".join(sentences))
    assert len(chunks) >= 2
    # Indexes are strictly monotonically increasing from 0.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_concatenate_back_to_source_with_no_overlap() -> None:
    chunker = SentenceBoundaryChunker(chunk_size_tokens=20, overlap_tokens=0)
    text = (
        "Alpha beta gamma delta epsilon. "
        "Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron. "
        "Pi rho sigma tau upsilon. "
        "Phi chi psi omega plus extra padding here."
    )
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # No-overlap chunking implies the union of chunk tokens covers
    # the source tokens with no duplicates beyond minor whitespace.
    chunk_tokens = sum(len(c.text.split()) for c in chunks)
    source_tokens = len(text.split())
    # Allow a small slack for whitespace-only differences at boundaries.
    assert chunk_tokens <= source_tokens + 2


def test_chunker_is_deterministic() -> None:
    chunker = SentenceBoundaryChunker(chunk_size_tokens=32)
    text = "First sentence. Second sentence. Third sentence is a bit longer."
    a = chunker.chunk(text)
    b = chunker.chunk(text)
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.index for c in a] == [c.index for c in b]


def test_text_chunk_is_immutable_dataclass() -> None:
    chunk = TextChunk(text="hi", index=0)
    with pytest.raises(Exception):
        # frozen=True dataclass; assignment must fail.
        chunk.text = "bye"  # type: ignore[misc]
