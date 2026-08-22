"""Tests for ground-truth similarity metric and scorer integration."""

from ocr_bench.metrics.ground_truth_similarity import ground_truth_similarity, _normalize
from ocr_bench.scorer import score


class TestNormalize:
    def test_whitespace_collapse(self):
        assert _normalize("hello   world") == _normalize("hello world")

    def test_newlines_ignored(self):
        assert _normalize("hello\nworld") == _normalize("hello world")

    def test_case_insensitive(self):
        assert _normalize("HELLO") == _normalize("hello")

    def test_smart_quotes(self):
        assert _normalize("\u201cHello\u201d") == _normalize('"Hello"')

    def test_decorative_lines_stripped(self):
        assert _normalize("Hello\n-------\nWorld") == _normalize("Hello World")


class TestGroundTruthSimilarity:
    def test_identical(self):
        assert ground_truth_similarity("Hello world", "Hello world") == 1.0

    def test_format_differences_tolerated(self):
        ocr = "Hello   world\nfoo  bar"
        gt = "Hello world foo bar"
        assert ground_truth_similarity(ocr, gt) > 0.99

    def test_case_differences_tolerated(self):
        assert ground_truth_similarity("HELLO WORLD", "hello world") > 0.99

    def test_real_ocr_errors_penalized(self):
        gt = "The quick brown fox jumps over the lazy dog"
        ocr = "Teh quikc brwon fox jmups oevr teh lzay dgo"
        sim = ground_truth_similarity(ocr, gt)
        assert 0.5 < sim < 0.95  # penalized but not zero

    def test_empty_texts(self):
        assert ground_truth_similarity("", "hello") == 0.0
        assert ground_truth_similarity("hello", "") == 0.0

    def test_completely_wrong(self):
        assert ground_truth_similarity("xxxxx yyyyy zzzzz", "the quick brown fox") < 0.3


class TestScorerWithGroundTruth:
    def test_without_gt_backward_compatible(self):
        result = score("The quick brown fox jumps over the lazy dog")
        assert result["ground_truth_similarity"] is None
        assert result["score"] >= 90.0

    def test_with_perfect_gt(self):
        text = "Hello world, this is a test"
        result = score(text, ground_truth=text)
        assert result["ground_truth_similarity"] is not None
        assert result["ground_truth_similarity"] > 0.99
        assert result["score"] >= 90.0

    def test_with_poor_gt_match(self):
        gt = "The quick brown fox jumps over the lazy dog"
        ocr = "xxxxx yyyyy zzzzz qqqqq wwwww"
        result_with = score(ocr, ground_truth=gt)
        result_without = score(ocr)
        # With GT, score should be lower since text doesn't match
        assert result_with["ground_truth_similarity"] < 0.3
        assert result_with["score"] <= result_without["score"]

    def test_gt_field_present_when_provided(self):
        result = score("test text", ground_truth="test text")
        assert "ground_truth_similarity" in result
        assert result["ground_truth_similarity"] is not None
