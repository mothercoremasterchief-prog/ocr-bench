"""Core scorer — combines all metrics into a single quality score."""

from __future__ import annotations

from typing import Optional, TypedDict

from ocr_bench.metrics import (
    typo_rate,
    gibberish_ratio,
    single_letter_ratio,
    char_separated_count,
    word_accuracy,
    ground_truth_similarity,
)


class ScoreResult(TypedDict):
    score: float
    typo_rate: float
    gibberish_ratio: float
    single_letter_ratio: float
    char_separated_count: int
    word_accuracy: float
    ground_truth_similarity: float | None


# Weights WITHOUT ground truth (sum to 1.0) — backward compatible
_WEIGHTS_NO_GT = {
    "typo_rate": 0.30,
    "gibberish_ratio": 0.30,
    "single_letter_ratio": 0.15,
    "char_separated": 0.10,
    "word_accuracy": 0.15,
}

# Weights WITH ground truth (sum to 1.0) — GT gets 25%, others scaled down
_WEIGHTS_GT = {
    "typo_rate": 0.20,
    "gibberish_ratio": 0.20,
    "single_letter_ratio": 0.10,
    "char_separated": 0.075,
    "word_accuracy": 0.10,
    "ground_truth_similarity": 0.325,
}


def score(text: str, ground_truth: Optional[str] = None) -> ScoreResult:
    """Score OCR output quality.

    Args:
        text: The OCR output text to score.
        ground_truth: Optional reference transcript. If provided, ground_truth_similarity
            is included in the composite score. If not, scoring works as before.

    Returns a dict with composite score (0-100) and per-metric breakdown.
    """
    if not text or not text.strip():
        return ScoreResult(
            score=0.0,
            typo_rate=1.0,
            gibberish_ratio=1.0,
            single_letter_ratio=1.0,
            char_separated_count=0,
            word_accuracy=1.0,
            ground_truth_similarity=0.0 if ground_truth else None,
        )

    tr = typo_rate(text)
    gr = gibberish_ratio(text)
    slr = single_letter_ratio(text)
    csc = char_separated_count(text)
    wa = word_accuracy(text)

    # Normalize char_separated_count to 0-1 range (cap at 10 occurrences)
    cs_norm = min(csc / 10.0, 1.0)

    if ground_truth:
        gts = ground_truth_similarity(text, ground_truth)
        weights = _WEIGHTS_GT
        penalty = (
            weights["typo_rate"] * tr
            + weights["gibberish_ratio"] * gr
            + weights["single_letter_ratio"] * slr
            + weights["char_separated"] * cs_norm
            + weights["word_accuracy"] * wa
            + weights["ground_truth_similarity"] * (1.0 - gts)
        )
    else:
        gts = None
        weights = _WEIGHTS_NO_GT
        penalty = (
            weights["typo_rate"] * tr
            + weights["gibberish_ratio"] * gr
            + weights["single_letter_ratio"] * slr
            + weights["char_separated"] * cs_norm
            + weights["word_accuracy"] * wa
        )

    composite = round(max(0.0, (1.0 - penalty) * 100), 1)

    return ScoreResult(
        score=composite,
        typo_rate=round(tr, 4),
        gibberish_ratio=round(gr, 4),
        single_letter_ratio=round(slr, 4),
        char_separated_count=csc,
        word_accuracy=round(wa, 4),
        ground_truth_similarity=round(gts, 4) if gts is not None else None,
    )
