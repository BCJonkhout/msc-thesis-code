"""Tests for the content-addressed embedding cache in OllamaEmbedder.

The cache lets a crash mid-build re-embed only the unfinished tail.
These tests never touch a live Ollama: the HTTP layer (_post_embed) is
stubbed so we can assert exactly which texts were (re-)embedded.
"""
from __future__ import annotations

from pathlib import Path

from pilot.encoders.ollama import OllamaEmbedder


def _stub(emb: OllamaEmbedder, sink: list[str]):
    """Replace the HTTP embed with a deterministic recorder."""
    def fake_post_embed(batch):
        sink.extend(batch)
        return [[float(len(t)), 1.0] for t in batch]
    emb._post_embed = fake_post_embed  # type: ignore[assignment]


def test_cache_serves_hits_and_embeds_only_misses(tmp_path: Path):
    sink: list[str] = []
    emb = OllamaEmbedder(model="m", cache_dir=tmp_path)
    _stub(emb, sink)

    r1 = emb.embed(["aa", "b"])
    assert r1.embeddings == [[2.0, 1.0], [1.0, 1.0]]
    assert sink == ["aa", "b"]

    # Second call: aa + b are cached on disk; only ccc is embedded.
    r2 = emb.embed(["aa", "b", "ccc"])
    assert sink == ["aa", "b", "ccc"]  # only ccc newly embedded
    # Order preserved with interleaved hits/misses.
    assert r2.embeddings == [[2.0, 1.0], [1.0, 1.0], [3.0, 1.0]]


def test_cache_persists_across_embedder_instances(tmp_path: Path):
    """A fresh embedder (e.g. after a crash/restart) reuses the cache."""
    sink1: list[str] = []
    emb1 = OllamaEmbedder(model="m", cache_dir=tmp_path)
    _stub(emb1, sink1)
    emb1.embed(["x", "y"])
    assert sink1 == ["x", "y"]

    sink2: list[str] = []
    emb2 = OllamaEmbedder(model="m", cache_dir=tmp_path)
    _stub(emb2, sink2)
    emb2.embed(["x", "y", "z"])
    assert sink2 == ["z"]  # x, y came from the persisted cache


def test_cache_disabled_by_default(tmp_path: Path):
    """With no cache_dir and no env var, every text is embedded."""
    sink: list[str] = []
    emb = OllamaEmbedder(model="m")  # no cache
    _stub(emb, sink)
    emb.embed(["a", "b"])
    emb.embed(["a", "b"])
    assert sink == ["a", "b", "a", "b"]


def test_cache_key_is_model_specific(tmp_path: Path):
    """The same text under a different model is a cache miss."""
    sink_a: list[str] = []
    a = OllamaEmbedder(model="model-a", cache_dir=tmp_path)
    _stub(a, sink_a)
    a.embed(["t"])

    sink_b: list[str] = []
    b = OllamaEmbedder(model="model-b", cache_dir=tmp_path)
    _stub(b, sink_b)
    b.embed(["t"])
    assert sink_b == ["t"]  # different model -> not served from model-a's entry


def test_corrupt_cache_entry_is_a_miss(tmp_path: Path):
    sink: list[str] = []
    emb = OllamaEmbedder(model="m", cache_dir=tmp_path)
    _stub(emb, sink)
    emb.embed(["t"])  # populates the cache
    # Corrupt the stored entry.
    cache_file = emb._cache_file("t")
    cache_file.write_text("{not json", encoding="utf-8")
    emb.embed(["t"])  # corrupt -> miss -> re-embed
    assert sink == ["t", "t"]
