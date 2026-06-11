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


# ── Prompt-style routing: the cross-architecture fairness contract ──
# All four architectures answer through base._render_prompt, so the prompt
# regime is one global choice (prompt_style), not a per-architecture default.
# These lock in the contract after the prompt-routing fix that unified the
# four answer prompts (the QASPER-F1 / NovelQA-abstention confound).

from pilot.architectures.base import _render_prompt  # noqa: E402


def test_multiplechoice_literature_has_no_abstention_clause() -> None:
    lit = load_template("qa_multiplechoice_literature")
    pilot = load_template("qa_multiplechoice")
    assert "i do not know" not in lit.text.lower()
    assert "i do not know" in pilot.text.lower()
    assert lit.slots == frozenset({"context", "query", "options"})


def test_render_prompt_freeform_routes_by_style() -> None:
    pilot = _render_prompt(context="c", query="q", options=None, prompt_style="pilot")
    lit = _render_prompt(context="c", query="q", options=None, prompt_style="literature")
    assert "i do not know" in pilot.lower()    # pilot free-form abstains
    assert "i do not know" not in lit.lower()  # literature free-form does not
    assert "concise" in lit.lower()


def test_render_prompt_mc_routes_by_style() -> None:
    opts = {"A": "Paris", "B": "London"}
    pilot = _render_prompt(context="c", query="q", options=opts, prompt_style="pilot")
    lit = _render_prompt(context="c", query="q", options=opts, prompt_style="literature")
    assert "i do not know" in pilot.lower()    # pilot MC abstains
    assert "i do not know" not in lit.lower()  # literature MC forces a letter
    assert "letter" in lit.lower()
