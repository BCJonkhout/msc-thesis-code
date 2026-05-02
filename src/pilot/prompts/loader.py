"""Prompt template loader.

`load_template("qa_freeform")` returns a `PromptTemplate` whose
`render(**slots)` method produces the rendered prompt. The same inputs
always produce byte-identical output.

Per pilot plan § 4.2:
- Templates are byte-identical across architectures (except for the
  `{context}` payload that the retriever fills).
- Slots are explicit named keys: `context`, `query`, `options`.
- The loader rejects rendering with unfilled or extra slots.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent
_SLOT_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    slots: frozenset[str]

    def render(self, **values: str) -> str:
        provided = set(values.keys())
        if provided != self.slots:
            missing = self.slots - provided
            extra = provided - self.slots
            raise ValueError(
                f"template {self.name!r} expected slots {sorted(self.slots)}, "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return self.text.format(**values)


def load_template(name: str) -> PromptTemplate:
    """Load a template by name (no extension).

    Looks for `<name>.txt` next to this loader.
    """
    path = _TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"template {name!r} not found at {path}")
    text = path.read_text(encoding="utf-8")
    slots = frozenset(_SLOT_RE.findall(text))
    return PromptTemplate(name=name, text=text, slots=slots)
