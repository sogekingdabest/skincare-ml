from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_capture_error_review import render_contact_sheet


CANDIDATE_FIELDS = (
    "review_id",
    "rank",
    "split",
    "sha256",
    "group_id",
    "source",
    "image_path",
    "true_label",
    "predicted_label",
    "probability_target_lesion",
    "probability_skin_without_target",
    "probability_no_skin",
    "predicted_ready",
    "error_types",
    "selection_reason",
    "original_annotation_source",
    "original_annotation_reviewer",
    "requires_human_confirmation",
    "approved_for_reweighting",
    "review_decision",
    "review_notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_candidates(
    predictions: list[dict[str, str]],
    manifest: list[dict[str, str]],
    dataset_root: Path,
    held_out_references: list[dict[str, str]] | None = None,
    maximum_candidates: int | None = None,
) -> list[dict[str, str]]:
    by_digest = {row["sha256"].lower(): row for row in manifest}
    held_out_groups = {
        row["group_id"] for row in (held_out_references or []) if row.get("group_id")
    }
    eligible = [
        row
        for row in predictions
        if row["split"] == "train"
        and row["true_label"] == "skin_without_target"
        and row["predicted_label"] == "target_lesion"
        and "argmax_error" in row["error_types"].split("|")
    ]
    eligible.sort(
        key=lambda row: float(row["probability_target_lesion"]), reverse=True
    )
    if maximum_candidates is not None:
        eligible = eligible[:maximum_candidates]

    candidates: list[dict[str, str]] = []
    seen_groups: set[str] = set()
    for rank, prediction in enumerate(eligible, start=1):
        digest = prediction["sha256"].lower()
        source = by_digest.get(digest)
        if source is None:
            raise ValueError(f"Prediction digest absent from manifest: {digest}")
        if source.get("split") != "train":
            raise ValueError(
                f"Prediction/manifest split mismatch for {digest}: {source.get('split')}"
            )
        if prediction["group_id"] in held_out_groups:
            raise ValueError(
                f"Training candidate overlaps a held-out group: {prediction['group_id']}"
            )
        if prediction["group_id"] in seen_groups:
            raise ValueError(
                f"Duplicate training group in candidate queue: {prediction['group_id']}"
            )
        seen_groups.add(prediction["group_id"])
        image_path = (dataset_root / source["image_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        original_annotation_source = source.get("annotation_source", "")
        candidates.append(
            {
                "review_id": f"train-{int(prediction['record_index']):04d}",
                "rank": str(rank),
                "split": "train",
                "sha256": digest,
                "group_id": prediction["group_id"],
                "source": prediction["source"],
                "image_path": str(image_path),
                "true_label": prediction["true_label"],
                "predicted_label": prediction["predicted_label"],
                "probability_target_lesion": prediction[
                    "probability_target_lesion"
                ],
                "probability_skin_without_target": prediction[
                    "probability_skin_without_target"
                ],
                "probability_no_skin": prediction["probability_no_skin"],
                "predicted_ready": prediction["predicted_ready"],
                "error_types": prediction["error_types"],
                "selection_reason": (
                    "ready_false_positive"
                    if "ready_false_positive" in prediction["error_types"].split("|")
                    else "argmax_target"
                ),
                "original_annotation_source": original_annotation_source,
                "original_annotation_reviewer": source.get(
                    "annotation_reviewer", ""
                ),
                "requires_human_confirmation": str(
                    original_annotation_source != "human"
                ).lower(),
                "approved_for_reweighting": "false",
                "review_decision": "",
                "review_notes": "",
            }
        )
    return candidates


def mine_training_hard_negatives(
    predictions_path: Path,
    dataset_root: Path,
    output_dir: Path,
    held_out_references_path: Path | None = None,
    maximum_candidates: int | None = None,
    sheet_size: int = 12,
) -> dict[str, object]:
    held_out = (
        read_csv(held_out_references_path) if held_out_references_path else []
    )
    candidates = select_candidates(
        read_csv(predictions_path),
        read_csv(dataset_root / "manifest.csv"),
        dataset_root,
        held_out,
        maximum_candidates,
    )
    if not candidates:
        raise ValueError("No training hard-negative candidates matched the policy.")
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "training_hard_negative_candidates.csv"
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)

    sheets: list[str] = []
    for start in range(0, len(candidates), sheet_size):
        page = start // sheet_size + 1
        path = output_dir / f"training_hard_negatives_{page:02d}.jpg"
        render_contact_sheet(
            candidates[start : start + sheet_size],
            path,
            columns=3,
            card_width=360,
            image_height=310,
        )
        sheets.append(path.name)

    report = {
        "schema_version": 1,
        "policy": "train_skin_without_target_predicted_target_lesion",
        "candidates": len(candidates),
        "ready_false_positives": sum(
            row["selection_reason"] == "ready_false_positive" for row in candidates
        ),
        "argmax_target_below_ready_threshold": sum(
            row["selection_reason"] == "argmax_target" for row in candidates
        ),
        "sources": dict(sorted(Counter(row["source"] for row in candidates).items())),
        "annotation_sources": dict(
            sorted(
                Counter(
                    row["original_annotation_source"] for row in candidates
                ).items()
            )
        ),
        "requires_human_confirmation": sum(
            row["requires_human_confirmation"] == "true" for row in candidates
        ),
        "approved_for_reweighting": 0,
        "held_out_group_overlap": 0,
        "contact_sheets": sheets,
        "queue": queue_path.name,
    }
    (output_dir / "training_hard_negative_mining_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine difficult skin_without_target samples only from train."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-out-references", type=Path)
    parser.add_argument("--maximum-candidates", type=int)
    parser.add_argument("--sheet-size", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = mine_training_hard_negatives(
        args.predictions,
        args.dataset_root,
        args.output_dir,
        args.held_out_references,
        args.maximum_candidates,
        args.sheet_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
