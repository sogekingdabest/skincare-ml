"""Consolidate independent capture reviews without inferring any decision.

Visual consensus and licence verification are deliberately separate gates.
Only unanimous, sufficiently reviewed, licence-verified inclusions are exported
as training-ready annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {"target_lesion", "skin_without_target", "no_skin"}
DECISIONS = {"include", "exclude"}
LICENSE_STATES = {"unreviewed", "verified", "rejected"}
REQUIRED_BATCH_COLUMNS = {
    "candidate_id",
    "source",
    "source_path",
    "group_id",
    "license",
    "license_status",
}
REQUIRED_REVIEW_COLUMNS = {
    "candidate_id",
    "reviewer_id",
    "reviewed_at",
    "decision",
    "review_label",
    "exclusion_reason",
    "license_review_status",
}
ADJUDICATION_COLUMNS = [
    "candidate_id",
    "source",
    "source_dataset",
    "source_path",
    "group_id",
    "subject_id",
    "session_id",
    "proposed_label",
    "final_label",
    "visual_status",
    "license_gate",
    "training_ready",
    "review_count",
    "required_reviews",
    "reviewers",
    "decisions",
    "review_labels",
    "exclusion_reasons",
    "license_review_states",
    "source_license",
    "source_license_status",
]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, fields


def load_batch(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows, fields = read_csv(path)
    missing = REQUIRED_BATCH_COLUMNS - set(fields)
    if missing:
        raise ValueError(f"Batch is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("The review batch is empty.")
    identifiers = [row["candidate_id"] for row in rows]
    if any(not candidate_id for candidate_id in identifiers):
        raise ValueError("candidate_id cannot be empty.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id must be unique in the batch.")
    return rows, fields


def load_review(
    path: Path, valid_ids: set[str]
) -> tuple[str, dict[str, dict[str, str]]]:
    rows, fields = read_csv(path)
    missing = REQUIRED_REVIEW_COLUMNS - set(fields)
    if missing:
        raise ValueError(f"{path}: missing review columns: {sorted(missing)}")
    if not rows:
        raise ValueError(f"{path}: review file is empty.")

    reviewer_ids = {row["reviewer_id"] for row in rows}
    if "" in reviewer_ids or len(reviewer_ids) != 1:
        raise ValueError(f"{path}: every row must have the same reviewer_id.")
    reviewer_id = next(iter(reviewer_ids))
    reviews: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        candidate_id = row["candidate_id"]
        if candidate_id not in valid_ids:
            raise ValueError(
                f"{path}:{line_number}: unknown candidate {candidate_id!r}."
            )
        if candidate_id in reviews:
            raise ValueError(
                f"{path}:{line_number}: duplicate candidate {candidate_id!r}."
            )
        decision = row["decision"]
        label = row["review_label"]
        reason = row["exclusion_reason"]
        licence_state = row["license_review_status"]
        if decision not in DECISIONS:
            raise ValueError(
                f"{path}:{line_number}: invalid decision {decision!r}."
            )
        if decision == "include" and label not in LABELS:
            raise ValueError(
                f"{path}:{line_number}: included review requires a valid label."
            )
        if decision == "exclude" and not reason:
            raise ValueError(
                f"{path}:{line_number}: excluded review requires a reason."
            )
        if licence_state not in LICENSE_STATES:
            raise ValueError(
                f"{path}:{line_number}: invalid licence state {licence_state!r}."
            )
        reviews[candidate_id] = row
    return reviewer_id, reviews


def licence_gate(reviews: list[dict[str, str]]) -> str:
    states = [review["license_review_status"] for review in reviews]
    if "rejected" in states:
        return "rejected"
    if states and all(state == "verified" for state in states):
        return "verified"
    return "pending"


def adjudicate_candidate(
    batch_row: dict[str, str],
    reviews: list[dict[str, str]],
    min_reviews: int,
) -> dict[str, str]:
    decisions = [review["decision"] for review in reviews]
    labels = [
        review["review_label"]
        for review in reviews
        if review["decision"] == "include"
    ]
    final_label = ""
    if len(reviews) < min_reviews:
        visual_status = "pending_review"
    elif len(set(decisions)) != 1:
        visual_status = "needs_adjudication"
    elif decisions[0] == "exclude":
        visual_status = "excluded"
    elif len(set(labels)) != 1:
        visual_status = "needs_adjudication"
    else:
        visual_status = "approved"
        final_label = labels[0]

    licence_status = licence_gate(reviews)
    training_ready = visual_status == "approved" and licence_status == "verified"
    return {
        "candidate_id": batch_row["candidate_id"],
        "source": batch_row["source"],
        "source_dataset": batch_row.get("source_dataset", ""),
        "source_path": batch_row["source_path"],
        "group_id": batch_row["group_id"],
        "subject_id": batch_row.get("subject_id", ""),
        "session_id": batch_row.get("session_id", ""),
        "proposed_label": batch_row.get("proposed_label", ""),
        "final_label": final_label,
        "visual_status": visual_status,
        "license_gate": licence_status,
        "training_ready": str(training_ready).lower(),
        "review_count": str(len(reviews)),
        "required_reviews": str(min_reviews),
        "reviewers": json.dumps(
            [review["reviewer_id"] for review in reviews], ensure_ascii=False
        ),
        "decisions": json.dumps(decisions, ensure_ascii=False),
        "review_labels": json.dumps(labels, ensure_ascii=False),
        "exclusion_reasons": json.dumps(
            [
                review["exclusion_reason"]
                for review in reviews
                if review["decision"] == "exclude"
            ],
            ensure_ascii=False,
        ),
        "license_review_states": json.dumps(
            [review["license_review_status"] for review in reviews],
            ensure_ascii=False,
        ),
        "source_license": batch_row["license"],
        "source_license_status": batch_row["license_status"],
    }


def consolidate(
    batch_rows: list[dict[str, str]],
    review_sets: list[tuple[str, dict[str, dict[str, str]]]],
    min_reviews: int,
) -> list[dict[str, str]]:
    reviewer_ids = [reviewer_id for reviewer_id, _ in review_sets]
    if len(reviewer_ids) != len(set(reviewer_ids)):
        raise ValueError("Every review file must belong to a distinct reviewer.")
    consolidated = []
    for batch_row in batch_rows:
        candidate_id = batch_row["candidate_id"]
        reviews = [
            review_by_id[candidate_id]
            for _, review_by_id in review_sets
            if candidate_id in review_by_id
        ]
        consolidated.append(adjudicate_candidate(batch_row, reviews, min_reviews))
    return consolidated


def approved_annotation(
    batch_row: dict[str, str], adjudicated: dict[str, str]
) -> dict[str, str]:
    result = dict(batch_row)
    result["label"] = adjudicated["final_label"]
    result["review_status"] = "approved"
    result["review_label"] = adjudicated["final_label"]
    result["include"] = "true"
    result["license_status"] = "verified"
    result["adjudicated_reviewers"] = adjudicated["reviewers"]
    return result


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_report(
    rows: list[dict[str, str]], reviewer_ids: list[str], min_reviews: int
) -> dict:
    visual = Counter(row["visual_status"] for row in rows)
    licences = Counter(row["license_gate"] for row in rows)
    ready = [row for row in rows if row["training_ready"] == "true"]
    return {
        "schema_version": 1,
        "candidates": len(rows),
        "reviewers": reviewer_ids,
        "minimum_independent_reviews": min_reviews,
        "visual_status": dict(sorted(visual.items())),
        "license_gate": dict(sorted(licences.items())),
        "training_ready": len(ready),
        "training_ready_labels": dict(
            sorted(Counter(row["final_label"] for row in ready).items())
        ),
        "training_ready_sources": dict(
            sorted(Counter(row["source"] for row in ready).items())
        ),
        "policy": {
            "visual": "unanimous decision and label",
            "licence": "all contributing reviews verified",
            "automatic_label_inference": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate independent capture-validity reviews."
    )
    parser.add_argument("batch", type=Path)
    parser.add_argument("reviews", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=2,
        help="Independent reviews required for visual consensus (default: 2).",
    )
    args = parser.parse_args()
    if args.min_reviews < 1:
        raise ValueError("--min-reviews must be at least 1.")

    batch_rows, batch_fields = load_batch(args.batch.resolve())
    valid_ids = {row["candidate_id"] for row in batch_rows}
    review_sets = [load_review(path.resolve(), valid_ids) for path in args.reviews]
    consolidated = consolidate(batch_rows, review_sets, args.min_reviews)

    output_dir = args.output_dir.resolve()
    output_paths = {
        "adjudication": output_dir / "adjudication.csv",
        "approved": output_dir / "approved_annotations.csv",
        "report": output_dir / "adjudication_report.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite adjudication outputs: "
            + ", ".join(str(path) for path in existing)
        )

    by_id = {row["candidate_id"]: row for row in batch_rows}
    ready_rows = [
        approved_annotation(by_id[row["candidate_id"]], row)
        for row in consolidated
        if row["training_ready"] == "true"
    ]
    approved_fields = list(batch_fields)
    for field in ("label", "adjudicated_reviewers"):
        if field not in approved_fields:
            approved_fields.append(field)
    write_csv_atomic(output_paths["adjudication"], ADJUDICATION_COLUMNS, consolidated)
    write_csv_atomic(output_paths["approved"], approved_fields, ready_rows)
    report = build_report(
        consolidated, [reviewer_id for reviewer_id, _ in review_sets], args.min_reviews
    )
    output_paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Adjudication written to {output_dir}")


if __name__ == "__main__":
    main()
