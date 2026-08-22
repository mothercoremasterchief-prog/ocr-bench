"""Tests for the minimum viable benchmark profile and its evidence."""

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "benchmarks" / "profiles" / "minimum.json"


def _load_analysis_module():
    path = REPO_ROOT / "benchmarks" / "analyze_minimum.py"
    spec = importlib.util.spec_from_file_location("analyze_minimum", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minimum_profile_has_exactly_eight_labeled_images():
    profile = json.loads(PROFILE_PATH.read_text())
    selected = profile["selected_images"]

    assert len(selected) == 8
    assert len(selected) == len(set(selected))
    for image in selected:
        assert (REPO_ROOT / "benchmarks" / "images" / f"{image}.png").is_file()
        assert (
            REPO_ROOT / "benchmarks" / "ground-truth" / f"{image}.txt"
        ).is_file()


def test_declared_profile_passes_and_is_empirically_minimal():
    analysis = _load_analysis_module()
    report = analysis.analyze(PROFILE_PATH)

    assert report["selected_passes"] is True
    assert report["selected_is_minimal"] is True
    assert report["minimum_passing_size"] == 8
    assert report["passing_candidates_by_size"].get("7", 0) == 0


def test_discrimination_counts_ties_not_just_spread():
    """A metric that ties most engines cannot rank them, however wide its range.

    This is the failure the fidelity numbers hide: ground_truth_similarity
    reproduces the ten-page ranking almost perfectly (rho 0.998) while giving
    13 of 30 engines the identical value.
    """

    module = _load_analysis_module()

    tied = module.discrimination([0.9567] * 13 + [0.1] + [0.99])
    assert tied["engines"] == 15
    assert tied["largest_tie"] == 13
    assert tied["distinct_values"] == 3
    assert tied["distinct_ratio"] == 3 / 15
    # Wide spread, almost no ordering power — spread alone would look healthy.
    assert tied["spread"] > 0.8


def test_discrimination_rewards_a_metric_that_orders_engines():
    module = _load_analysis_module()

    separated = module.discrimination([0.01, 0.02, 0.03, 0.04, 0.05])
    assert separated["largest_tie"] == 1
    assert separated["distinct_ratio"] == 1.0
    assert separated["distinct_values"] == 5


def test_discrimination_rounds_before_comparing():
    """Values differing below the reported precision are a tie, not a rank."""

    module = _load_analysis_module()

    result = module.discrimination([0.95670001, 0.95670002, 0.5])
    assert result["largest_tie"] == 2
    assert result["distinct_values"] == 2


def test_discrimination_handles_an_empty_run():
    module = _load_analysis_module()

    empty = module.discrimination([])
    assert empty["engines"] == 0
    assert empty["largest_tie"] == 0
    assert empty["spread"] == 0.0

