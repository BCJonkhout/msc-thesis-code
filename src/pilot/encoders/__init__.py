"""Embedding-encoder wrappers + chunkers for the pilot's encoder
selection sub-experiment (decision-matrix row #9).

The default encoder is BGE-M3 served via Ollama (`bge-m3`). Escalation
candidates listed in `configs/embedding.yaml` are activated only on
empirical failure of the default per pilot plan § 5.8 row #9.
"""
from pilot.encoders.chunker import SentenceBoundaryChunker, TextChunk
from pilot.encoders.ollama import OllamaEmbedder

__all__ = ["OllamaEmbedder", "SentenceBoundaryChunker", "TextChunk"]
