"""Conservative validation-only READY threshold policy for capture validity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


READY_CLASS_INDEX = 0
DEFAULT_CONFIDENCE_Z = 1.96
DEFAULT_RESOLUTION = 0.001


def ready_statistics(
    labels: Iterable[int],
    target_scores: Iterable[float],
    threshold: float,
    ready_class_index: int = READY_CLASS_INDEX,
    confidence_z: float = DEFAULT_CONFIDENCE_Z,
) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(target_scores, dtype=np.float64)
    if labels_array.shape != scores_array.shape or labels_array.size == 0:
        raise ValueError("Labels and scores must be non-empty arrays with equal shape.")

    positives = labels_array == ready_class_index
    predicted_ready = scores_array >= threshold
    true_positives = int(np.sum(predicted_ready & positives))
    false_positives = int(np.sum(predicted_ready & ~positives))
    false_negatives = int(np.sum(~predicted_ready & positives))
    true_negatives = int(np.sum(~predicted_ready & ~positives))
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    negative_count = false_positives + true_negatives
    false_ready_rate = false_positives / max(negative_count, 1)
    denominator = 1.0 + confidence_z**2 / max(negative_count, 1)
    centre = (
        false_ready_rate + confidence_z**2 / (2 * max(negative_count, 1))
    ) / denominator
    margin = (
        confidence_z
        * np.sqrt(
            false_ready_rate * (1.0 - false_ready_rate) / max(negative_count, 1)
            + confidence_z**2 / (4 * max(negative_count, 1) ** 2)
        )
        / denominator
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "false_ready_rate": float(false_ready_rate),
        "false_ready_rate_wilson_upper_95": float(centre + margin),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def select_ready_threshold(
    labels: Iterable[int],
    target_scores: Iterable[float],
    maximum_false_ready_rate: float,
    threshold_resolution: float = DEFAULT_RESOLUTION,
) -> dict[str, float | int | bool | str]:
    if not 0.0 < threshold_resolution <= 1.0:
        raise ValueError("threshold_resolution must be in (0, 1].")
    steps = int(round(1.0 / threshold_resolution))
    candidates = [
        ready_statistics(labels, target_scores, threshold)
        for threshold in np.linspace(0.0, 1.0, steps + 1)
    ]
    feasible = [
        candidate
        for candidate in candidates
        if candidate["false_ready_rate_wilson_upper_95"]
        <= maximum_false_ready_rate
    ]
    pool = feasible
    selection_mode = "wilson_upper_95"
    if not pool:
        pool = [candidate for candidate in candidates if candidate["false_positives"] == 0]
        selection_mode = "exploratory_zero_observed_false_ready"
    if not pool:
        raise RuntimeError("No zero-false-READY threshold is available at this resolution.")
    selected = max(
        pool,
        key=lambda item: (item["recall"], item["precision"], -item["threshold"]),
    )
    selected["selection_mode"] = selection_mode
    selected["release_target_satisfied"] = bool(feasible)
    selected["threshold_resolution"] = float(threshold_resolution)
    selected["selected_on_split"] = "validation"
    selected["test_used_for_selection"] = False
    return selected


def _read_predictions(path: Path) -> tuple[list[int], list[float]]:
    label_ids: list[int] = []
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label_ids.append(int(row["true_label_id"]))
            scores.append(float(row["probability_target_lesion"]))
    if not label_ids:
        raise ValueError(f"Prediction file is empty: {path}")
    return label_ids, scores


def analyze_prediction_exports(
    validation_path: Path,
    test_path: Path,
    output_path: Path,
    maximum_false_ready_rate: float = 0.01,
    threshold_resolution: float = DEFAULT_RESOLUTION,
) -> dict[str, object]:
    validation_labels, validation_scores = _read_predictions(validation_path)
    selection = select_ready_threshold(
        validation_labels,
        validation_scores,
        maximum_false_ready_rate,
        threshold_resolution,
    )
    test_labels, test_scores = _read_predictions(test_path)
    test_result = ready_statistics(test_labels, test_scores, selection["threshold"])
    report = {
        "schema_version": 1,
        "policy": "validation_only_zero_observed_false_ready_fallback",
        "target_false_ready_rate": maximum_false_ready_rate,
        "selection": selection,
        "test_reported_after_freeze": test_result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-false-ready-rate", type=float, default=0.01)
    parser.add_argument("--threshold-resolution", type=float, default=DEFAULT_RESOLUTION)
    args = parser.parse_args()
    print(json.dumps(analyze_prediction_exports(
        args.validation_predictions,
        args.test_predictions,
        args.output,
        args.maximum_false_ready_rate,
        args.threshold_resolution,
    ), indent=2))


if __name__ == "__main__":
    main()
