"""Library plumbing behind the four-architecture long-context QA benchmark.

This package powers a completed study comparing flat full-context, naive
RAG, RAPTOR, and GraphRAG on QASPER (local Answer-F1) and NovelQA (held-out
Codabench multiple-choice gold), with a single answerer at T=0 and a shared
BGE-M3 encoder. The package name ``pilot`` is historical: the project began
as a calibration pilot and the name was kept once the work grew into the
main study. See code/docs/CODEMAP.md for the producer-to-paper provenance map
and ../thesis-msc/notes/pilot_setup_plan.md for the original setup plan.
"""

__version__ = "0.1.0"
