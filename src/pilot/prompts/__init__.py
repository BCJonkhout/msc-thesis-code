"""Prompt template harness.

Two templates live as plain text files under this package:
- qa_freeform.txt        QASPER (free-form answers)
- qa_multiplechoice.txt  NovelQA, QuALITY (multiple choice)

Templates are byte-identical across architectures except for the
`{context}` payload, which differs by retrieval strategy. The loader
returns a callable; rendering with the same inputs always produces
byte-identical output (no implicit randomness).

See pilot plan § 4.2.
"""

from pilot.prompts.loader import PromptTemplate, load_template

__all__ = ["PromptTemplate", "load_template"]
