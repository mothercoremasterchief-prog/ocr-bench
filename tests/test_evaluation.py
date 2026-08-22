"""Tests for standard ground-truth OCR evaluation."""

import pytest

from ocr_bench.evaluation import evaluate, normalize_for_evaluation


def test_identical_text_has_no_errors():
    result = evaluate("Hello, world!", "Hello, world!")
    assert result["character_error_rate"] == 0.0
    assert result["word_error_rate"] == 0.0
    assert result["bag_of_words_error_rate"] == 0.0
    assert result["reading_order_error"] == 0.0


def test_character_error_rate_counts_a_substitution():
    result = evaluate("cut", "cat")
    assert result["character_errors"] == 1
    assert result["character_error_rate"] == pytest.approx(1 / 3, abs=1e-6)


def test_word_substitution_counts_once_in_wer_and_bag_wer():
    result = evaluate("one too three", "one two three")
    assert result["word_error_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["bag_of_words_error_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["reading_order_error"] == 0.0


def test_word_reordering_isolated_from_recognition_errors():
    result = evaluate("three two one", "one two three")
    assert result["bag_of_words_error_rate"] == 0.0
    assert result["word_error_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["reading_order_error"] == result["word_error_rate"]


def test_normalization_ignores_case_and_whitespace_by_default():
    result = evaluate("HELLO\n\nworld", "hello world")
    assert result["character_error_rate"] == 0.0
    assert normalize_for_evaluation("e\u0301") == "é"


def test_case_sensitive_mode_retains_case_errors():
    result = evaluate("HELLO", "hello", case_sensitive=True)
    assert result["character_error_rate"] == 1.0


def test_empty_reference_behavior_is_bounded_and_explicit():
    assert evaluate("", "")["character_error_rate"] == 0.0
    result = evaluate("hallucinated text", "")
    assert result["character_error_rate"] == 1.0
    assert result["word_error_rate"] == 1.0
    assert result["bag_of_words_error_rate"] == 1.0
