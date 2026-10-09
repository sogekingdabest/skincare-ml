from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DECISIONS = {"keep_label", "correct_to_target", "ambiguous", "exclude"}
REVIEWER_KINDS = {"human", "assisted"}
REQUIRED_DECISION_FIELDS = {
    "review_id",
    "decision",
    "reviewer",
    "reviewer_kind",
    "confidence",
    "rationale",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def focused_errors(queue: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in queue
        if row["true_label"] == "skin_without_target"
        and row["predicted_label"] == "target_lesion"
        and "argmax_error" in row["error_types"].split("|")
    ]


def validate_decisions(
    focus: list[dict[str, str]], decisions: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    expected = {row["review_id"] for row in focus}
    if not decisions:
        raise ValueError("The decision file is empty.")
    missing_columns = REQUIRED_DECISION_FIELDS - set(decisions[0])
    if missing_columns:
        raise ValueError(f"Decision file is missing columns: {sorted(missing_columns)}")
    by_id: dict[str, dict[str, str]] = {}
    for row in decisions:
        review_id = row["review_id"].strip()
        if review_id in by_id:
            raise ValueError(f"Duplicate review decision: {review_id}")
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Unsupported decision for {review_id}: {row['decision']}")
        if row["reviewer_kind"] not in REVIEWER_KINDS:
            raise ValueError(
                f"Unsupported reviewer_kind for {review_id}: {row['reviewer_kind']}"
            )
        if not row["reviewer"].strip() or not row["rationale"].strip():
            raise ValueError(f"Reviewer and rationale are required for {review_id}.")
        by_id[review_id] = row
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        unexpected = sorted(set(by_id) - expected)
        raise ValueError(
            f"Decision coverage mismatch; missing={missing}, unexpected={unexpected}"
        )
    return by_id


def finalize_review(
    queue_path: Path,
    decisions_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    focus = focused_errors(read_csv(queue_path))
    decisions = validate_decisions(focus, read_csv(decisions_path))
    manifest = {row["sha256"].lower(): row for row in read_csv(manifest_path)}
    if any(row["split"] == "train" for row in focus):
        raise ValueError("Focused error review unexpectedly contains training samples.")

    adjudicated: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    for sample in focus:
        source = manifest.get(sample["sha256"].lower())
        if source is None:
            raise ValueError(f"Sample absent from manifest: {sample['sha256']}")
        decision = decisions[sample["review_id"]]
        requires_human_confirmation = (
            source.get("annotation_source") != "human"
            and decision["reviewer_kind"] != "human"
        )
        combined = {
            "review_id": sample["review_id"],
            "sha256": sample["sha256"],
            "group_id": sample["group_id"],
            "source": sample["source"],
            "split": sample["split"],
            "current_label": sample["true_label"],
            "model_prediction": sample["predicted_label"],
            "probability_target_lesion": sample["probability_target_lesion"],
            "original_annotation_source": source.get("annotation_source", ""),
            "original_annotation_reviewer": source.get("annotation_reviewer", ""),
            "decision": decision["decision"],
            "reviewer": decision["reviewer"],
            "reviewer_kind": decision["reviewer_kind"],
            "confidence": decision["confidence"],
            "rationale": decision["rationale"],
            "requires_human_confirmation": str(
                requires_human_confirmation
            ).lower(),
        }
        adjudicated.append(combined)
        if decision["decision"] == "keep_label":
            references.append(
                {
                    "review_id": sample["review_id"],
                    "sha256": sample["sha256"],
                    "group_id": sample["group_id"],
                    "source": sample["source"],
                    "split": sample["split"],
                    "label": sample["true_label"],
                    "probability_target_lesion": sample[
                        "probability_target_lesion"
                    ],
                    "role": "held_out_evaluation_hard_negative",
                    "training_eligible": "false",
                    "requires_human_confirmation": str(
                        requires_human_confirmation
                    ).lower(),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    adjudication_path = output_dir / "hard_negative_adjudication.csv"
    reference_path = output_dir / "held_out_hard_negative_references.csv"
    for path, rows in (
        (adjudication_path, adjudicated),
        (reference_path, references),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    decision_counts = Counter(row["decision"] for row in adjudicated)
    report = {
        "schema_version": 1,
        "review_scope": "skin_without_target_to_target_lesion_errors",
        "samples": len(adjudicated),
        "decisions": dict(sorted(decision_counts.items())),
        "already_human_labeled": sum(
            row["original_annotation_source"] == "human" for row in adjudicated
        ),
        "requires_human_confirmation": sum(
            row["requires_human_confirmation"] == "true" for row in adjudicated
        ),
        "held_out_hard_negatives": len(references),
        "training_samples_created": 0,
        "split_leakage_prevented": True,
        "artifacts": {
            "adjudication": adjudication_path.name,
            "held_out_references": reference_path.name,
        },
    }
    (output_dir / "hard_negative_review_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and finalize the focused capture hard-negative review."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize_review(
        args.queue, args.decisions, args.manifest, args.output_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
