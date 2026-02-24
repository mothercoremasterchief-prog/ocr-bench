"""Tests for ocr-bench scorer."""

import pytest
from ocr_bench.scorer import score


def test_perfect_text():
    result = score("The quick brown fox jumps over the lazy dog")
    assert result["score"] >= 90.0
    assert result["typo_rate"] < 0.05
    assert result["gibberish_ratio"] == 0.0
    assert result["single_letter_ratio"] == 0.0
    assert result["char_separated_count"] == 0


def test_empty_text():
    result = score("")
    assert result["score"] == 0.0


def test_gibberish_heavy():
    result = score("xkjhf qwrtpl bvnmcx zxcvbn asdfgh")
    assert result["gibberish_ratio"] > 0.5
    assert result["score"] < 50.0


def test_char_separated():
    result = score("H e l l o world, this is a t e s t")
    assert result["char_separated_count"] >= 1


def test_midword_space_insertion():
    """Mid-word splits like 'Ope nOC R' should be detected."""
    from ocr_bench.metrics.char_separated import char_separated_count
    assert char_separated_count("Ope nOC R com") >= 1
    assert char_separated_count("Te st ing the doc ume nt") >= 1
    # Common short words should NOT trigger
    assert char_separated_count("I am a fan of it") == 0


def test_single_letter_spam():
    result = score("b c d e f g h j k l m n o p q r s t u v w x y z")
    assert result["single_letter_ratio"] > 0.5
    assert result["score"] < 85.0


def test_mixed_quality():
    # Some good words, some typos
    result = score("The documnt contans sevral erors but is mostly readable text")
    assert 40.0 < result["score"] < 95.0
    assert result["typo_rate"] > 0.0


def test_word_accuracy_in_score():
    """word_accuracy field should be present in score result."""
    result = score("The quick brown fox jumps over the lazy dog")
    assert "word_accuracy" in result
    assert result["word_accuracy"] == 0.0


def test_word_merge_detection():
    """Very long merged tokens should be penalized."""
    from ocr_bench.metrics.word_accuracy import word_accuracy
    # Normal text
    assert word_accuracy("The quick brown fox") == 0.0
    # Merged word (>20 chars)
    assert word_accuracy("Thequickbrownfoxjumpsover the lazy dog") > 0.0


def test_word_fragmentation_detection():
    """Runs of short non-common tokens should be penalized."""
    from ocr_bench.metrics.word_accuracy import word_accuracy
    # Fragmented OCR: short non-dictionary fragments in sequence
    assert word_accuracy("hel lo wor ld fro mth edo cum ent") > 0.0
    # Common short words should not trigger
    assert word_accuracy("I am a fan of it and he is too") == 0.0


def test_real_ocr_output():
    """Simulate realistic OCR output with minor issues."""
    text = (
        "Invoice Number: 12345\n"
        "Date: January 15, 2024\n"
        "Total Amount: $1,234.56\n"
        "Thank you for your purchase."
    )
    result = score(text)
    assert result["score"] >= 70.0
