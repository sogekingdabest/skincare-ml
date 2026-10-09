"""Compare two capture-validity runs on the exact same held-out samples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def compare_predictions(
    baseline: list[dict[str, str]], candidate: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, object]]:
    baseline_by_hash = {row["sha256"]: row for row in baseline}
    candidate_by_hash = {row["sha256"]: row for row in candidate}
    if len(baseline_by_hash) != len(baseline) or len(candidate_by_hash) != len(candidate):
        raise ValueError("Prediction SHA-256 values must be unique.")
    if set(baseline_by_hash) != set(candidate_by_hash):
        raise ValueError("Runs do not cover the same held-out samples.")

    transitions: Counter[str] = Counter()
    ready_changes: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for digest in sorted(baseline_by_hash):
        old = baseline_by_hash[digest]
        new = candidate_by_hash[digest]
        for field in ("split", "true_label", "group_id", "source"):
            if old[field] != new[field]:
                raise ValueError(f"Held-out provenance mismatch for {digest}: {field}")
        old_correct = old["predicted_label"] == old["true_label"]
        new_correct = new["predicted_label"] == new["true_label"]
        transition = f"{'correct' if old_correct else 'wrong'}->{'correct' if new_correct else 'wrong'}"
        transitions[transition] += 1
        old_errors = set(filter(None, old["error_types"].split("|")))
        new_errors = set(filter(None, new["error_types"].split("|")))
        for error in ("ready_false_positive", "ready_false_negative"):
            if error in old_errors and error not in new_errors:
                ready_changes[f"{error}_fixed"] += 1
            if error not in old_errors and error in new_errors:
                ready_changes[f"{error}_new"] += 1
        rows.append({
            "split": old["split"], "sha256": digest, "group_id": old["group_id"],
            "source": old["source"], "true_label": old["true_label"],
            "baseline_predicted_label": old["predicted_label"],
            "candidate_predicted_label": new["predicted_label"],
            "baseline_probability_target_lesion": old["probability_target_lesion"],
            "candidate_probability_target_lesion": new["probability_target_lesion"],
            "target_probability_delta": format(
                float(new["probability_target_lesion"]) - float(old["probability_target_lesion"]),
                ".9g",
            ),
            "baseline_ready_threshold": old["ready_threshold"],
            "candidate_ready_threshold": new["ready_threshold"],
            "baseline_error_types": old["error_types"],
            "candidate_error_types": new["error_types"],
            "argmax_transition": transition,
        })
    summary = {
        "held_out_samples": len(rows),
        "held_out_identity_verified": True,
        "argmax_transitions": dict(sorted(transitions.items())),
        "net_argmax_corrections": transitions["wrong->correct"] - transitions["correct->wrong"],
        "ready_changes": dict(sorted(ready_changes.items())),
    }
    return rows, summary


def metric_summary(baseline: dict, candidate: dict) -> dict[str, object]:
    old_report = baseline["test_multiclass_report"]
    new_report = candidate["test_multiclass_report"]
    old_ready = baseline["test_ready_gate"]
    new_ready = candidate["test_ready_gate"]
    return {
        "test_accuracy": {"baseline": old_report["accuracy"], "candidate": new_report["accuracy"], "delta": new_report["accuracy"] - old_report["accuracy"]},
        "test_macro_f1": {"baseline": old_report["macro avg"]["f1-score"], "candidate": new_report["macro avg"]["f1-score"], "delta": new_report["macro avg"]["f1-score"] - old_report["macro avg"]["f1-score"]},
        "target_argmax_recall": {"baseline": old_report["target_lesion"]["recall"], "candidate": new_report["target_lesion"]["recall"], "delta": new_report["target_lesion"]["recall"] - old_report["target_lesion"]["recall"]},
        "target_argmax_precision": {"baseline": old_report["target_lesion"]["precision"], "candidate": new_report["target_lesion"]["precision"], "delta": new_report["target_lesion"]["precision"] - old_report["target_lesion"]["precision"]},
        "ready_recall": {"baseline": old_ready["recall"], "candidate": new_ready["recall"], "delta": new_ready["recall"] - old_ready["recall"]},
        "ready_precision": {"baseline": old_ready["precision"], "candidate": new_ready["precision"], "delta": new_ready["precision"] - old_ready["precision"]},
        "ready_false_positives": {"baseline": old_ready["false_positives"], "candidate": new_ready["false_positives"], "delta": new_ready["false_positives"] - old_ready["false_positives"]},
        "ready_false_negatives": {"baseline": old_ready["false_negatives"], "candidate": new_ready["false_negatives"], "delta": new_ready["false_negatives"] - old_ready["false_negatives"]},
        "ready_threshold": {"baseline": old_ready["threshold"], "candidate": new_ready["threshold"], "delta": new_ready["threshold"] - old_ready["threshold"]},
        "ready_roc_auc": {"baseline": old_ready["roc_auc"], "candidate": new_ready["roc_auc"], "delta": new_ready["roc_auc"] - old_ready["roc_auc"]},
        "ready_average_precision": {"baseline": old_ready["average_precision"], "candidate": new_ready["average_precision"], "delta": new_ready["average_precision"] - old_ready["average_precision"]},
        "tflite_argmax_agreement": {"baseline": baseline["parity"]["argmax_agreement"], "candidate": candidate["parity"]["argmax_agreement"]},
        "tflite_ready_agreement": {"baseline": baseline["parity"]["ready_decision_agreement"], "candidate": candidate["parity"]["ready_decision_agreement"]},
        "candidate_release_eligible": candidate["release_gate"]["eligible"],
    }


def compare_runs(
    baseline_metrics_path: Path,
    candidate_metrics_path: Path,
    baseline_prediction_paths: list[Path],
    candidate_prediction_paths: list[Path],
    output_dir: Path,
    baseline_name: str = "v3",
    candidate_name: str = "v4",
) -> dict[str, object]:
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    candidate_metrics = json.loads(candidate_metrics_path.read_text(encoding="utf-8"))
    baseline_predictions = [
        row for path in baseline_prediction_paths for row in read_csv(path)
    ]
    candidate_predictions = [row for path in candidate_prediction_paths for row in read_csv(path)]
    rows, prediction_summary = compare_predictions(
        baseline_predictions, candidate_predictions
    )
    report = {
        "schema_version": 1,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "prediction_comparison": prediction_summary,
        "metrics": metric_summary(baseline_metrics, candidate_metrics),
        "candidate_dataset_release": candidate_metrics["dataset_release"],
        "candidate_release_gate": candidate_metrics["release_gate"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--candidate-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="v3")
    parser.add_argument("--candidate-name", default="v4")
    args = parser.parse_args()
    print(
        json.dumps(
            compare_runs(
                args.baseline_metrics,
                args.candidate_metrics,
                args.baseline_predictions,
                args.candidate_predictions,
                args.output_dir,
                baseline_name=args.baseline_name,
                candidate_name=args.candidate_name,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
