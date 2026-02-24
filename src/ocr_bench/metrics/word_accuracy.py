"""Word accuracy metric — detects abnormal word fragmentation and merging.

Good OCR preserves word boundaries. Bad OCR either:
- Splits words: "document" → "docu ment" (inflated word count)
- Merges words: "the cat" → "thecat" (deflated word count)

This metric measures the ratio of "suspicious" tokens — words that are
unusually long (likely merges) or sequences of unusually short tokens
(likely splits). Unlike char_separated which catches "H e l l o" patterns,
this catches subtler fragmentation like "docu ment ation" or merges like
"theentire".
"""

from __future__ import annotations

import re

# Words longer than this are suspicious merges (very few real English words exceed 20 chars)
_MERGE_THRESHOLD = 20

# Sequences of N+ tokens all <= this length suggest fragmentation
_FRAG_TOKEN_MAX_LEN = 3
_FRAG_SEQ_MIN_LEN = 3  # need at least 3 short tokens in a row

# Common short-word sequences to exclude (avoid false positives)
_COMMON_SHORT = frozenset(
    w.lower()
    for w in [
        "i", "a", "an", "am", "as", "at", "be", "by", "do", "go", "he",
        "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so",
        "to", "up", "us", "we", "the", "and", "but", "for", "not", "you",
        "all", "any", "can", "had", "has", "her", "him", "his", "how", "its",
        "let", "may", "new", "now", "old", "our", "out", "own", "say", "she",
        "too", "use", "was", "who", "boy", "did", "get", "got", "has", "him",
        "hot", "man", "men", "put", "ran", "red", "run", "saw", "set", "sit",
        "top", "two", "war", "why", "big", "end", "far", "few",
    ]
)


def word_accuracy(text: str) -> float:
    """Return a 0.0–1.0 penalty score for word boundary errors.

    0.0 = no issues detected, 1.0 = severe fragmentation/merging.
    """
    if not text or not text.strip():
        return 1.0

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
    if len(tokens) < 2:
        return 0.0

    total = len(tokens)
    issues = 0

    # Detect merges: very long tokens
    for t in tokens:
        if len(t) > _MERGE_THRESHOLD:
            issues += 1

    # Detect fragmentation: runs of short tokens that aren't common words
    run_len = 0
    for t in tokens:
        if len(t) <= _FRAG_TOKEN_MAX_LEN and t.lower() not in _COMMON_SHORT:
            run_len += 1
        else:
            if run_len >= _FRAG_SEQ_MIN_LEN:
                issues += run_len
            run_len = 0
    # Final run
    if run_len >= _FRAG_SEQ_MIN_LEN:
        issues += run_len

    return min(issues / total, 1.0)
