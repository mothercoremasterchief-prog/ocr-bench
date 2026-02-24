"""Char-separated detection — count words with spaces between every letter
and mid-word space insertion (e.g. 'Ope nOC R')."""

from __future__ import annotations

import re

# Classic: single letters separated by spaces, 3+ letters — "H e l l o"
_CHAR_SEP_PATTERN = re.compile(r"(?<!\S)([A-Za-z] ){2,}[A-Za-z](?!\S)")

# Mid-word splits: 3+ consecutive short fragments (1-3 chars each) that look
# like one word got chopped up — "Ope nOC R", "Te st ing"
# Requires at least 3 fragments, each 1-3 alpha chars, separated by single spaces.
_MID_WORD_PATTERN = re.compile(
    r"(?<!\S)"
    r"[A-Za-z]{1,3}"          # first fragment
    r"(?: [A-Za-z]{1,3}){2,}" # 2+ more short fragments
    r"(?!\S)"
)

# Common short words to avoid false positives on sequences like "I am a"
_COMMON_SHORT = frozenset(
    "i a an am is it in on at to or of if no so do up us we he me my by"
    " be go ok oh hi ha ah the and but for not you all can had her was one our".split()
)


def char_separated_count(text: str) -> int:
    """Return the number of char-separated sequences found.

    Detects both classic patterns ('H e l l o') and mid-word space insertion
    ('Ope nOC R', 'Te st ing').
    """
    classic = len(_CHAR_SEP_PATTERN.findall(text))

    # Mid-word detection: find candidate sequences, filter out common-word runs
    midword = 0
    for m in _MID_WORD_PATTERN.finditer(text):
        frags = m.group().lower().split()
        # If most fragments are common English words, skip (not a split)
        common_count = sum(1 for f in frags if f in _COMMON_SHORT)
        if common_count / len(frags) >= 0.6:
            continue
        midword += 1

    return classic + midword
