"""Multiple-choice answer post-processor.

Per pilot plan § 5.8 row #3, this parser must achieve 100% on a
synthetic test set covering common answer formats:
- "A"                  bare letter
- "(A)"                parenthesised letter
- "Option A"           "Option" + letter
- full option text     model echoed the answer text
- "A. The Eiffel Tower" letter + period + text

Returns the option letter (uppercase) on match, or None if no match.

The parser is deterministic and pure. Test set lives in
tests/test_mc_postprocessor.py.
"""
from __future__ import annotations

import re

# Patterns are tried in this order; first match wins.
# Each pattern captures the option letter in group 1.
_PATTERNS: tuple[re.Pattern, ...] = (
    # "(A)" or "( A )" with optional surrounding whitespace
    re.compile(r"^\s*\(\s*([A-Z])\s*\)\s*[\.\:\)]?\s*$"),
    # "A. text" or "A: text" or "A) text"
    re.compile(r"^\s*([A-Z])\s*[\.\:\)]\s+\S"),
    # bare "A" with optional trailing punctuation/whitespace
    re.compile(r"^\s*([A-Z])\s*[\.\!\?\:\)]?\s*$"),
    # "A\n\nExplanation..." — reasoning-tuned models emit the letter on its own
    # line followed by a justification. Must run before the keyword pattern so
    # it wins against any incidental "option"/"answer"/"choice" tokens in the
    # explanation body.
    re.compile(r"^\s*([A-Z])\s*\n+"),
    # "Option A" / "option a" / "Answer: A" / "The answer is A".
    # The trailing \b on the keyword group prevents partial matches against
    # plurals like "options", which under re.IGNORECASE would otherwise let
    # the [A-Z] capture pick up the 's' and yield a spurious 'S'.
    re.compile(r"\b(?:option|answer|choice)\b(?:\s*(?:is|:))?\s*([A-Z])\b", re.IGNORECASE),
    # "(option A)" loose
    re.compile(r"\(\s*(?:option\s+)?([A-Z])\s*\)", re.IGNORECASE),
)


def parse_mc_answer(raw: str, options: list[str] | None = None) -> str | None:
    """Parse the option letter from a model's free-form output.

    Args:
        raw:     Raw model output string.
        options: Optional list of option text strings (in order A, B, C, ...).
                 If provided, full-text matches are tried as a final fallback.

    Returns:
        The matched option letter in uppercase ('A', 'B', ...), or
        None if no rule matched.
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    # Strip a leading "Answer:" prefix that some models emit.
    text = re.sub(r"^\s*answer\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return None

    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper()

    # Fallback: full-option-text match (case-insensitive substring).
    if options:
        text_lc = text.lower()
        # Prefer exact-equal match; fall back to word-boundary match.
        for i, opt in enumerate(options):
            if opt.strip().lower() == text_lc:
                return chr(ord("A") + i)
        # Word-boundary search avoids short option strings like "No" matching
        # inside unrelated words such as "not" on yes/no questions, while
        # still permitting multi-word option text like "Spanish necklace".
        for i, opt in enumerate(options):
            opt_clean = opt.strip()
            if opt_clean and re.search(
                rf"\b{re.escape(opt_clean)}\b", text, re.IGNORECASE
            ):
                return chr(ord("A") + i)

    return None
