"""Typo rate — fraction of words that fail a frequency-based spellcheck."""

from __future__ import annotations

import re
from wordfreq import word_frequency

# Words with frequency >= this threshold in the English corpus are considered valid.
_FREQ_THRESHOLD = 1e-8

# Patterns we skip (numbers, emails, URLs, etc.)
_SKIP = re.compile(
    r"^("
    r"\d[\d.,:%/]*"           # numbers / dates / percentages
    r"|https?://\S+"          # URLs
    r"|\S+@\S+\.\S+"         # emails
    r"|[A-Z]{1,5}"           # short acronyms (NASA, OCR, …)
    r")$"
)


def _is_valid_word(word: str) -> bool:
    """Return True if the word is likely a real English word."""
    if _SKIP.match(word):
        return True
    # wordfreq: check lowercase; frequency 0 means unknown
    freq = word_frequency(word.lower(), "en")
    return freq >= _FREQ_THRESHOLD


def typo_rate(text: str) -> float:
    """Return the fraction of words that appear to be misspelled (0.0–1.0)."""
    words = re.findall(r"[A-Za-z']+(?:-[A-Za-z']+)*", text)
    if not words:
        return 0.0
    bad = sum(1 for w in words if not _is_valid_word(w))
    return bad / len(words)
