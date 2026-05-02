"""Prompt template loader: byte-identical rendering, slot enforcement."""
from __future__ import annotations

import pytest

from pilot.prompts import load_template


def test_freeform_template_renders_deterministically() -> None:
    """Same inputs → byte-identical output across two invocations."""
    tpl = load_template("qa_freeform")
    a = tpl.render(context="The cat sat on the mat.", query="What did the cat do?")
    b = tpl.render(context="The cat sat on the mat.", query="What did the cat do?")
    assert a == b


def test_freeform_template_slots_enforced() -> None:
    """Missing or extra slots raise ValueError."""
    tpl = load_template("qa_freeform")
    with pytest.raises(ValueError, match="missing"):
        tpl.render(context="just context")
    with pytest.raises(ValueError, match="extra"):
        tpl.render(context="c", query="q", options="not allowed here")


def test_multiplechoice_template_renders_deterministically() -> None:
    """MC template also renders deterministically."""
    tpl = load_template("qa_multiplechoice")
    a = tpl.render(
        context="Paris is the capital of France.",
        query="What is the capital?",
        options="A. Paris\nB. London\nC. Berlin",
    )
    b = tpl.render(
        context="Paris is the capital of France.",
        query="What is the capital?",
        options="A. Paris\nB. London\nC. Berlin",
    )
    assert a == b


def test_multiplechoice_template_slots() -> None:
    """MC template requires context, query, options."""
    tpl = load_template("qa_multiplechoice")
    assert tpl.slots == frozenset({"context", "query", "options"})
