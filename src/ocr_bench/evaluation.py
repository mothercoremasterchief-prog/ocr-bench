"""Ground-truth OCR evaluation metrics.

The heuristic :func:`ocr_bench.score` API remains useful when no transcript is
available.  A benchmark with ground truth should use edit-distance metrics,
which are directly interpretable and do not depend on an English dictionary.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Sequence, TypedDict
import unicodedata

try:  # Optional accelerator installed by the benchmark extra.
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
except ImportError:  # pragma: no cover - exercised in minimal installations
    _RapidLevenshtein = None


class EvaluationResult(TypedDict):
    """Standard page-level OCR assessment and its raw denominators."""

    character_error_rate: float
    word_error_rate: float
    bag_of_words_error_rate: float
    reading_order_error: float
    character_errors: int
    word_errors: int
    bag_of_words_errors: int
    reference_characters: int
    reference_words: int


def normalize_for_evaluation(text: str, *, case_sensitive: bool = False) -> str:
    """Normalize text before ground-truth comparison.

    NFC makes canonically equivalent Unicode compare equally. Whitespace is
    collapsed so harmless line wrapping does not dominate transcription
    quality, while the sequence of words is retained for reading-order checks.
    Punctuation and diacritics remain significant.
    """

    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if case_sensitive else normalized.casefold()


def _levenshtein(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Return Levenshtein edit distance using linear memory."""

    if _RapidLevenshtein is not None:
        return int(_RapidLevenshtein.distance(reference, hypothesis))

    if len(reference) < len(hypothesis):
        # Keep the working row on the shorter sequence. Distance is symmetric.
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, 1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, 1):
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            substitution = previous[hyp_index - 1] + (ref_item != hyp_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _rate(errors: int, reference_size: int, hypothesis_size: int) -> float:
    """Normalize errors by reference size, including the empty-reference case."""

    if reference_size:
        return errors / reference_size
    return 0.0 if hypothesis_size == 0 else 1.0


def evaluate(
    text: str,
    ground_truth: str,
    *,
    case_sensitive: bool = False,
) -> EvaluationResult:
    """Evaluate an OCR transcript against page-level ground truth.

    Three base assessments are returned:

    * character error rate (CER), for exact transcription fidelity;
    * word error rate (WER), which is sensitive to word sequence; and
    * bag-of-words error rate (bWER), which ignores word order.

    ``reading_order_error`` is derived from ``WER - bWER``. It is diagnostic,
    not a fourth independent assessment. Rates can exceed 1.0 when a model
    inserts more content than the reference contains.
    """

    hypothesis = normalize_for_evaluation(text, case_sensitive=case_sensitive)
    reference = normalize_for_evaluation(
        ground_truth, case_sensitive=case_sensitive
    )

    character_errors = _levenshtein(reference, hypothesis)
    reference_characters = len(reference)
    character_error_rate = _rate(
        character_errors, reference_characters, len(hypothesis)
    )

    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    word_errors = _levenshtein(reference_tokens, hypothesis_tokens)
    reference_words = len(reference_tokens)
    word_error_rate = _rate(word_errors, reference_words, len(hypothesis_tokens))

    reference_bag = Counter(reference_tokens)
    hypothesis_bag = Counter(hypothesis_tokens)
    missing_words = sum((reference_bag - hypothesis_bag).values())
    extra_words = sum((hypothesis_bag - reference_bag).values())
    # Allow a missing and an extra word to pair as one substitution. This is the
    # order-independent analogue of WER and therefore shares its denominator.
    bag_of_words_errors = max(missing_words, extra_words)
    bag_of_words_error_rate = _rate(
        bag_of_words_errors, reference_words, len(hypothesis_tokens)
    )

    reading_order_errors = max(0, word_errors - bag_of_words_errors)
    reading_order_error = _rate(
        reading_order_errors, reference_words, len(hypothesis_tokens)
    )

    return EvaluationResult(
        character_error_rate=round(character_error_rate, 6),
        word_error_rate=round(word_error_rate, 6),
        bag_of_words_error_rate=round(bag_of_words_error_rate, 6),
        reading_order_error=round(reading_order_error, 6),
        character_errors=character_errors,
        word_errors=word_errors,
        bag_of_words_errors=bag_of_words_errors,
        reference_characters=reference_characters,
        reference_words=reference_words,
    )


__all__ = ["EvaluationResult", "evaluate", "normalize_for_evaluation"]
