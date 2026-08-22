#!/usr/bin/env python3
"""Verify the minimum benchmark profile against saved full-corpus runs.

The search is deliberately exhaustive: the current corpus has only ten pages,
so every coverage-valid subset can be compared with the full ten-page result.
The newest pre-validation run for each engine is used for selection, and the
May 2026 rebuild is held out for validation.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = BENCHMARK_DIR / "profiles" / "minimum.json"


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end + 1) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float:
    """Return Spearman's rho, including average ranks for ties."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal lists of length >= 2")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left_ranks)
    right_variance = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_variance * right_variance)
    return numerator / denominator if denominator else 0.0


def _metric_value(record: dict[str, Any], metric: str) -> float | None:
    value = record.get(metric)
    if value is None and metric == "ground_truth_similarity":
        value = record.get("metrics", {}).get(metric)
    if value is None:
        return None
    value = float(value)
    return value / 100 if metric == "score" else value


def discrimination(values: list[float]) -> dict[str, Any]:
    """How well a metric can actually ORDER engines.

    Fidelity (Spearman vs the full page set) says the subset preserves the old
    ranking. It says nothing about whether that ranking is meaningful: a metric
    that assigns 13 of 30 engines the identical value reproduces itself
    perfectly and still cannot tell them apart.

    `distinct_ratio` is the share of engines receiving a unique value, and
    `largest_tie` is the biggest group sharing one value. On the May 2026 run
    ground_truth_similarity ties 13 engines at 0.9567 (distinct_ratio 0.53)
    while CER ties at most 3 (0.83) — which is why CER is the ranking metric.
    """

    rounded = [round(v, 4) for v in values]
    counts: dict[float, int] = {}
    for value in rounded:
        counts[value] = counts.get(value, 0) + 1
    total = len(rounded) or 1
    ordered = sorted(rounded)
    return {
        "engines": len(rounded),
        "distinct_values": len(counts),
        "distinct_ratio": len(counts) / total,
        "largest_tie": max(counts.values()) if counts else 0,
        "spread": (ordered[-1] - ordered[0]) if ordered else 0.0,
    }



def _cer_means(result_path: Path, images: list[str]) -> list[float]:
    """Per-engine mean CER over `images`, recomputed from stored transcripts.

    Returns [] when the evaluation module or the transcripts are unavailable,
    so an older result file without `full_text` degrades to "no CER row"
    rather than breaking the whole report.
    """

    try:
        sys.path.insert(0, str(BENCHMARK_DIR.parent / "src"))
        from ocr_bench.evaluation import evaluate
    except Exception:
        return []

    payload = json.loads(result_path.read_text())
    means: list[float] = []
    for image_results in payload.get("results", {}).values():
        if not isinstance(image_results, dict):
            continue
        rates: list[float] = []
        for image in images:
            record = image_results.get(image)
            if (
                not isinstance(record, dict)
                or record.get("error")
                or not record.get("full_text")
                or not record.get("ground_truth")
            ):
                rates = []
                break
            rates.append(
                evaluate(record["full_text"], record["ground_truth"])[
                    "character_error_rate"
                ]
            )
        if rates:
            means.append(sum(rates) / len(rates))
    return means


def load_complete_rows(
    result_path: Path,
    images: list[str],
    metric: str,
) -> tuple[str, dict[str, tuple[float, ...]]]:
    """Load complete, error-free engine rows from a harness result file."""

    payload = json.loads(result_path.read_text())
    rows: dict[str, tuple[float, ...]] = {}
    for engine, image_results in payload.get("results", {}).items():
        if not isinstance(image_results, dict):
            continue
        values: list[float] = []
        for image in images:
            record = image_results.get(image)
            if not isinstance(record, dict) or record.get("error"):
                break
            value = _metric_value(record, metric)
            if value is None:
                break
            values.append(value)
        if len(values) == len(images):
            rows[engine] = tuple(values)
    return payload.get("run_timestamp", ""), rows


def load_training_rows(
    result_paths: Iterable[Path],
    validation_path: Path,
    images: list[str],
    metric: str,
) -> dict[str, tuple[float, ...]]:
    """Keep the newest complete pre-validation row for each engine."""

    newest: dict[str, tuple[str, str, tuple[float, ...]]] = {}
    for path in sorted(result_paths):
        if path.resolve() == validation_path.resolve():
            continue
        try:
            timestamp, rows = load_complete_rows(path, images, metric)
        except (json.JSONDecodeError, OSError):
            continue
        for engine, row in rows.items():
            candidate = (timestamp, str(path), row)
            if engine not in newest or candidate[:2] >= newest[engine][:2]:
                newest[engine] = candidate
    return {engine: value[2] for engine, value in newest.items()}


def subset_fidelity(
    rows: dict[str, tuple[float, ...]], subset_indexes: tuple[int, ...]
) -> dict[str, float]:
    """Compare engine means and ranks for a subset against the full corpus."""

    full_means = [statistics.mean(row) for row in rows.values()]
    subset_means = [
        statistics.mean(row[index] for index in subset_indexes)
        for row in rows.values()
    ]
    absolute_errors = sorted(
        abs(full - subset) for full, subset in zip(full_means, subset_means)
    )
    p95_index = math.ceil(0.95 * len(absolute_errors)) - 1
    return {
        "spearman": spearman_correlation(full_means, subset_means),
        "mean_absolute_error": statistics.mean(absolute_errors),
        "p95_absolute_error": absolute_errors[p95_index],
        "worst_absolute_error": absolute_errors[-1],
    }


def meets_thresholds(
    fidelity: dict[str, float], thresholds: dict[str, float]
) -> bool:
    return (
        fidelity["spearman"] >= thresholds["minimum_spearman"]
        and fidelity["mean_absolute_error"]
        <= thresholds["maximum_mean_absolute_error"]
        and fidelity["p95_absolute_error"]
        <= thresholds["maximum_p95_absolute_error"]
        and fidelity["worst_absolute_error"]
        <= thresholds["maximum_worst_absolute_error"]
    )


def has_required_coverage(
    subset: set[str], coverage_groups: dict[str, list[str]]
) -> bool:
    return all(subset.intersection(candidates) for candidates in coverage_groups.values())


def analyze(profile_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text())
    selection = profile["selection_analysis"]
    images = selection["candidate_images"]
    selected_images = profile["selected_images"]
    selected_indexes = tuple(images.index(image) for image in selected_images)
    thresholds = selection["thresholds"]
    coverage_groups = selection["coverage_groups"]
    validation_path = BENCHMARK_DIR / selection["validation_results"]
    result_paths = list((BENCHMARK_DIR / "results").glob("*/*.json"))

    datasets: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {}
    for metric in selection["fidelity_metrics"]:
        _, validation_rows = load_complete_rows(validation_path, images, metric)
        training_rows = load_training_rows(
            result_paths, validation_path, images, metric
        )
        if len(training_rows) < 2 or len(validation_rows) < 2:
            raise ValueError(f"Not enough complete rows to analyze {metric}")
        datasets[metric] = {
            "training": training_rows,
            "validation": validation_rows,
        }

    candidates_by_size: dict[int, list[tuple[str, ...]]] = {}
    passing_by_size: dict[int, list[tuple[str, ...]]] = {}
    for size in range(1, len(images) + 1):
        for subset_tuple in combinations(images, size):
            subset = set(subset_tuple)
            if not has_required_coverage(subset, coverage_groups):
                continue
            candidates_by_size.setdefault(size, []).append(subset_tuple)
            subset_indexes = tuple(images.index(image) for image in subset_tuple)
            passes = True
            for metric_datasets in datasets.values():
                for rows in metric_datasets.values():
                    if not meets_thresholds(
                        subset_fidelity(rows, subset_indexes), thresholds
                    ):
                        passes = False
                        break
                if not passes:
                    break
            if passes:
                passing_by_size.setdefault(size, []).append(subset_tuple)

    minimum_size = min(passing_by_size) if passing_by_size else None
    selected_fidelity: dict[str, dict[str, dict[str, float]]] = {}
    selected_passes = has_required_coverage(set(selected_images), coverage_groups)
    for metric, metric_datasets in datasets.items():
        selected_fidelity[metric] = {}
        for split, rows in metric_datasets.items():
            fidelity = subset_fidelity(rows, selected_indexes)
            selected_fidelity[metric][split] = fidelity
            selected_passes = selected_passes and meets_thresholds(
                fidelity, thresholds
            )

    training_counts = {
        metric: len(metric_datasets["training"])
        for metric, metric_datasets in datasets.items()
    }
    validation_counts = {
        metric: len(metric_datasets["validation"])
        for metric, metric_datasets in datasets.items()
    }
    # Discriminative power of each metric on the SELECTED subset, measured on
    # the held-out validation engines. Fidelity alone cannot reveal a metric
    # that ranks by coin flip because it ties most engines together.
    discrimination_report: dict[str, Any] = {}
    for metric, metric_datasets in datasets.items():
        rows = metric_datasets["validation"]
        means = [
            sum(values[i] for i in selected_indexes) / len(selected_indexes)
            for values in rows.values()
        ]
        discrimination_report[metric] = discrimination(means)

    # CER is recomputed from the stored transcripts, so a past run can be
    # re-scored for free. It is reported for comparison only — the fidelity
    # thresholds above assume a higher-is-better metric, and CER is inverted.
    cer_means = _cer_means(validation_path, selected_images)
    if len(cer_means) >= 2:
        discrimination_report["character_error_rate"] = discrimination(cer_means)

    is_minimal = selected_passes and len(selected_images) == minimum_size
    return {
        "discrimination": discrimination_report,
        "profile_id": profile["id"],
        "selected_images": selected_images,
        "selected_size": len(selected_images),
        "minimum_passing_size": minimum_size,
        "selected_passes": selected_passes,
        "selected_is_minimal": is_minimal,
        "training_engine_counts": training_counts,
        "validation_engine_counts": validation_counts,
        "thresholds": thresholds,
        "selected_fidelity": selected_fidelity,
        "coverage_valid_candidates_by_size": {
            str(size): len(subsets)
            for size, subsets in candidates_by_size.items()
        },
        "passing_candidates_by_size": {
            str(size): len(subsets) for size, subsets in passing_by_size.items()
        },
    }


def _print_report(report: dict[str, Any]) -> None:
    status = "PASS" if report["selected_is_minimal"] else "FAIL"
    print(f"{report['profile_id']}: {status}")
    print(
        f"Selected {report['selected_size']} images; minimum passing size: "
        f"{report['minimum_passing_size']}"
    )
    print("Images: " + ", ".join(report["selected_images"]))
    print(
        "Complete engines (training / held out): "
        + ", ".join(
            f"{metric}={report['training_engine_counts'][metric]}/"
            f"{report['validation_engine_counts'][metric]}"
            for metric in report["training_engine_counts"]
        )
    )
    print("\nDeclared subset fidelity (rho, MAE, p95, worst):")
    for metric, splits in report["selected_fidelity"].items():
        for split, values in splits.items():
            print(
                f"  {metric:24s} {split:10s} "
                f"{values['spearman']:.4f}, "
                f"{values['mean_absolute_error']:.4f}, "
                f"{values['p95_absolute_error']:.4f}, "
                f"{values['worst_absolute_error']:.4f}"
            )
    print("\nPassing subsets by size:")
    for size, count in report["passing_candidates_by_size"].items():
        print(f"  {size}: {count}")

    disc = report.get("discrimination")
    if disc:
        print("\nDiscriminative power on the selected subset")
        print("  (can the metric actually ORDER engines, or does it tie them?)")
        for metric, stats in disc.items():
            print(
                f"  {metric:24s} distinct {stats['distinct_values']:>3d}/"
                f"{stats['engines']:<3d} ({stats['distinct_ratio']:.2f})  "
                f"largest tie {stats['largest_tie']:>2d}  "
                f"spread {stats['spread']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the minimum OCR benchmark profile"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Path to a benchmark profile JSON file",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero unless the declared profile passes and is minimal",
    )
    args = parser.parse_args()
    report = analyze(args.profile)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    if args.check and not report["selected_is_minimal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
