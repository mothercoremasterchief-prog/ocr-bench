"""Integration tests for ground-truth metrics in the benchmark harness."""

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_harness_module():
    path = REPO_ROOT / "benchmarks" / "harness.py"
    spec = importlib.util.spec_from_file_location("benchmark_harness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_records_standard_evaluation_and_summary(monkeypatch):
    harness = _load_harness_module()
    monkeypatch.setattr(harness, "collect_images", lambda cfg, base: {"page": b"x"})
    monkeypatch.setattr(
        harness, "load_ground_truth", lambda cfg, base: {"page": "one two three"}
    )
    monkeypatch.setattr(
        harness,
        "call_endpoint",
        lambda endpoint, image: ("one too three", 12.4, None, None),
    )
    config = {
        "endpoints": [{"name": "test-engine", "type": "local"}],
        "corpus": {},
    }

    output = harness.run_benchmark(config)
    page = output["results"]["test-engine"]["page"]
    summary = output["summary"]["test-engine"]

    assert page["evaluation"]["word_error_rate"] == 0.333333
    assert page["evaluation"]["bag_of_words_error_rate"] == 0.333333
    assert summary["macro_wer"] == 0.333333
    assert summary["micro_wer"] == 0.333333
    assert summary["valid_benchmark"] is True


def test_endpoint_error_invalidates_benchmark(monkeypatch):
    harness = _load_harness_module()
    monkeypatch.setattr(harness, "collect_images", lambda cfg, base: {"page": b"x"})
    monkeypatch.setattr(
        harness, "load_ground_truth", lambda cfg, base: {"page": "reference"}
    )
    monkeypatch.setattr(
        harness,
        "call_endpoint",
        lambda endpoint, image: ("", 20.0, None, "endpoint failed"),
    )
    config = {
        "endpoints": [{"name": "test-engine", "type": "local"}],
        "corpus": {},
    }

    output = harness.run_benchmark(config)
    summary = output["summary"]["test-engine"]

    assert summary["total_errors"] == 1
    assert summary["valid_benchmark"] is False
    assert summary["macro_cer"] is None


def test_builtin_minimum_profile_loads():
    harness = _load_harness_module()
    profile = harness.load_profile("minimum")

    assert profile["id"] == "mvb-en-business-v1"
    assert len(profile["selected_images"]) == 8
