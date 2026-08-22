"""Legacy ground-truth similarity metric based on SequenceMatcher."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


def _normalize(text: str) -> str:
    """Normalize text to remove format-only differences.

    - Unicode NFKC normalization (curly quotes → straight, etc.)
    - Collapse all whitespace (spaces, tabs, newlines) to single space
    - Strip leading/trailing whitespace
    - Lowercase
    - Remove decorative characters (dashes used as separators, underscores)
    """
    # NFKC normalizes ligatures, compatibility chars, curly quotes etc.
    text = unicodedata.normalize("NFKC", text)
    # Replace common unicode punctuation variants
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
    # Remove lines that are purely decorative (only dashes, underscores, equals)
    lines = text.split("\n")
    lines = [line for line in lines if not re.match(r'^[\s\-_=]+$', line)]
    text = "\n".join(lines)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    return text


def ground_truth_similarity(ocr_text: str, ground_truth: str) -> float:
    """Compute normalized similarity between OCR output and ground truth.

    Returns a float 0.0 (no match) to 1.0 (perfect match).
    Both texts are normalized before comparison to avoid penalizing
    format-only differences (whitespace, line breaks, case, unicode variants).
    """
    if not ocr_text or not ground_truth:
        return 0.0

    norm_ocr = _normalize(ocr_text)
    norm_gt = _normalize(ground_truth)

    if not norm_ocr or not norm_gt:
        return 0.0

    # SequenceMatcher.ratio() gives 2*M/T where M is the number of matching
    # characters and T is the total length of both sequences. It is retained
    # for backward compatibility; use ocr_bench.evaluate for standard CER/WER.
    return SequenceMatcher(None, norm_ocr, norm_gt).ratio()
