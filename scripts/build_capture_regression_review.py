"""Build a local visual review pack for newly introduced READY false positives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_capture_error_review import build_review_queue, render_contact_sheet


REGRESSION_FIELDS = (
    "review_id", "split", "sha256", "source", "group_id", "image_path",
    "true_label", "baseline_predicted_label", "candidate_predicted_label",
    "baseline_probability_target_lesion", "candidate_probability_target_lesion",
    "baseline_ready_threshold", "candidate_ready_threshold",
    "baseline_error_types", "candidate_error_types", "review_decision",
    "review_notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_new_ready_false_positives(
    comparisons: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected = []
    for row in comparisons:
        old_errors = set(filter(None, row["baseline_error_types"].split("|")))
        new_errors = set(filter(None, row["candidate_error_types"].split("|")))
        if "ready_false_positive" not in old_errors and "ready_false_positive" in new_errors:
            selected.append(row)
    return selected


def build_regression_review(
    comparison_path: Path,
    candidate_predictions_paths: list[Path],
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    comparisons = select_new_ready_false_positives(read_csv(comparison_path))
    comparison_by_hash = {row["sha256"]: row for row in comparisons}
    all_candidate_predictions = [
        row for path in candidate_predictions_paths for row in read_csv(path)
    ]
    candidate_predictions = [
        row for row in all_candidate_predictions
        if row["sha256"] in comparison_by_hash
    ]
    if len({row["sha256"] for row in candidate_predictions}) != len(candidate_predictions):
        raise ValueError("Candidate prediction inputs contain duplicate SHA-256 values.")
    if len(candidate_predictions) != len(comparisons):
        raise ValueError("Candidate predictions do not cover every selected regression.")

    image_rows = build_review_queue(
        candidate_predictions, read_csv(dataset_root / "manifest.csv"), dataset_root
    )
    queue = []
    for image_row in image_rows:
        comparison = comparison_by_hash[image_row["sha256"]]
        queue.append({
            "review_id": image_row["review_id"],
            "split": image_row["split"],
            "sha256": image_row["sha256"],
            "source": image_row["source"],
            "group_id": image_row["group_id"],
            "image_path": image_row["image_path"],
            "true_label": image_row["true_label"],
            "baseline_predicted_label": comparison["baseline_predicted_label"],
            "candidate_predicted_label": comparison["candidate_predicted_label"],
            "baseline_probability_target_lesion": comparison["baseline_probability_target_lesion"],
            "candidate_probability_target_lesion": comparison["candidate_probability_target_lesion"],
            "baseline_ready_threshold": comparison["baseline_ready_threshold"],
            "candidate_ready_threshold": comparison["candidate_ready_threshold"],
            "baseline_error_types": comparison["baseline_error_types"],
            "candidate_error_types": comparison["candidate_error_types"],
            "review_decision": "",
            "review_notes": "",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "new_ready_false_positives.csv"
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGRESSION_FIELDS)
        writer.writeheader()
        writer.writerows(queue)

    sheet_rows = [{
        **row,
        "predicted_label": row["candidate_predicted_label"],
        "probability_target_lesion": row["candidate_probability_target_lesion"],
        "error_types": "new READY false positive",
    } for row in queue]
    sheet_path = output_dir / "new_ready_false_positives.jpg"
    render_contact_sheet(sheet_rows, sheet_path, columns=3)

    report = {
        "schema_version": 1,
        "purpose": "held_out_v4_new_ready_false_positive_review",
        "items": len(queue),
        "all_held_out": all(row["split"] in {"validation", "test"} for row in queue),
        "training_use_prohibited": True,
        "sources": sorted({row["source"] for row in queue}),
        "queue": queue_path.name,
        "contact_sheet": sheet_path.name if queue else None,
    }
    (output_dir / "review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_regression_review(
        args.comparison, args.candidate_predictions, args.dataset_root, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
