"""Single-letter ratio — proportion of tokens that are lone characters."""

from __future__ import annotations

import re

# Common single-letter words that are legitimate
_LEGIT_SINGLES = {"a", "i", "I"}


def single_letter_ratio(text: str) -> float:
    """Return the fraction of whitespace-separated tokens that are a single letter (0.0–1.0).

    Excludes common legitimate singles like 'a', 'I'.
    Excludes pure digits and punctuation.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    alpha_tokens = [t for t in tokens if re.match(r"^[A-Za-z]$", t)]
    suspicious = [t for t in alpha_tokens if t not in _LEGIT_SINGLES]
    return len(suspicious) / len(tokens)
