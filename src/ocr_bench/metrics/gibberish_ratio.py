"""Gibberish ratio — fraction of words that look like random character soup."""

from __future__ import annotations

import re
from wordfreq import word_frequency

# A word is "gibberish" if it:
#  1. Has no dictionary frequency at all
#  2. Contains unusual character patterns (excessive consonant clusters, etc.)

_CONSONANT_CLUSTER = re.compile(r"[bcdfghjklmnpqrstvwxyz]{5,}", re.IGNORECASE)
_DIGIT_MIX = re.compile(r"(?:[a-zA-Z]+\d+|\d+[a-zA-Z]+)")  # mixed alpha+digit like "h3ll0"
_SKIP = re.compile(r"^(\d[\d.,]*|https?://\S+|\S+@\S+\.\S+|[A-Z]{1,5})$")


def _is_gibberish(word: str) -> bool:
    if _SKIP.match(word):
        return False
    if len(word) <= 2:
        return False
    lw = word.lower()
    # If word has any dictionary frequency, it's not gibberish
    if word_frequency(lw, "en") > 0:
        return False
    # Heuristics for gibberish
    if _CONSONANT_CLUSTER.search(lw):
        return True
    if _DIGIT_MIX.match(word):
        return True
    # Long unknown word with low vowel ratio
    vowels = sum(1 for c in lw if c in "aeiou")
    if len(lw) >= 5 and vowels / len(lw) < 0.15:
        return True
    return False


def gibberish_ratio(text: str) -> float:
    """Return the fraction of words that look like gibberish (0.0–1.0)."""
    words = re.findall(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*", text)
    if not words:
        return 0.0
    bad = sum(1 for w in words if _is_gibberish(w))
    return bad / len(words)
