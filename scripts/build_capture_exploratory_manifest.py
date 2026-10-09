"""Build a provenance-aware exploratory capture-validity manifest.

This command deliberately distinguishes human reviews from high-confidence
vision-assisted labels.  The result is suitable for an internal engineering
baseline, but it is not a gold evaluation set and it is not release-ready while
source licences remain unverified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from adjudicate_capture_reviews import load_batch, load_review, write_csv_atomic
from build_capture_tfrecords import CLASS_NAMES, SPLITS, assign_grouped_splits


ANNOTATION_COLUMNS = [
    "candidate_id",
    "source_path",
    "label",
    "group_id",
    "original_group_id",
    "source",
    "original_sha256",
    "subject_id",
    "session_id",
    "device_id",
    "skin_tone",
    "body_site",
    "license",
    "consent_scope",
    "split",
    "is_real_capture",
    "bad_framing",
    "bad_light",
    "blur",
    "low_resolution",
    "too_far",
    "notes",
    "annotation_source",
    "annotation_reviewer",
    "annotation_confidence",
    "weak_supervision",
    "requires_human_review",
    "source_license_status",
    "release_eligible",
]
EXCLUSION_COLUMNS = [
    "candidate_id",
    "source",
    "original_group_id",
    "proposed_label",
    "selected_reviewer",
    "decision",
    "review_label",
    "review_confidence",
    "requires_human_review",
    "exclusion_reason",
]


def parse_source_prefixes(values: list[str]) -> dict[str, PurePosixPath]:
    result: dict[str, PurePosixPath] = {}
    for value in values:
        source, separator, prefix = value.partition("=")
        source = source.strip()
        posix = PurePosixPath(prefix.strip().replace("\\", "/"))
        if not separator or not source or not prefix.strip():
            raise ValueError(
                f"Invalid --source-prefix {value!r}; expected SOURCE=RELATIVE_PATH."
            )
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"Source prefix must be relative and contained: {prefix!r}")
        if source in result:
            raise ValueError(f"Duplicate source prefix for {source!r}.")
        result[source] = posix
    return result


def safe_source_path(
    source_root: Path,
    prefix: PurePosixPath,
    source_path: str,
) -> tuple[str, Path]:
    relative = PurePosixPath(source_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe source_path {source_path!r}.")
    combined = PurePosixPath(*prefix.parts, *relative.parts)
    resolved = source_root.joinpath(*combined.parts).resolve()
    if not resolved.is_relative_to(source_root) or not resolved.is_file():
        raise FileNotFoundError(f"Missing or unsafe source image: {resolved}")
    return combined.as_posix(), resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_review(
    candidate_id: str,
    human: dict[str, dict[str, str]],
    assisted: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]] | None:
    if candidate_id in human:
        return "human", human[candidate_id]
    if candidate_id in assisted:
        return "assisted", assisted[candidate_id]
    return None


def _exclusion_reason(origin: str, review: dict[str, str]) -> str:
    if review["decision"] == "exclude":
        reason = review.get("exclusion_reason", "") or "unspecified"
        return f"{origin}_excluded:{reason}"
    if origin == "assisted":
        if review.get("requires_human_review", "").lower() != "false":
            return "assisted_requires_human_review"
        if review.get("review_confidence", "").lower() != "high":
            return "assisted_not_high_confidence"
    return ""


def _merge_duplicate_groups(rows: list[dict[str, str]]) -> int:
    groups_by_digest: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_digest[row["original_sha256"]].add(row["original_group_id"])
    duplicate_clusters = 0
    for digest, groups in groups_by_digest.items():
        if len(groups) <= 1:
            continue
        duplicate_clusters += 1
        merged = f"duplicate:{digest[:20]}"
        for row in rows:
            if row["original_sha256"] == digest:
                row["group_id"] = merged
    return duplicate_clusters


def build_exploratory_manifest(
    batch_rows: list[dict[str, str]],
    human_reviewer: str,
    human: dict[str, dict[str, str]],
    assisted_reviewer: str,
    assisted: dict[str, dict[str, str]],
    source_root: Path,
    source_prefixes: dict[str, PurePosixPath],
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    valid_ids = {row["candidate_id"] for row in batch_rows}
    reviewed_ids = set(human) | set(assisted)
    missing = valid_ids - reviewed_ids
    unknown = reviewed_ids - valid_ids
    if missing or unknown:
        raise ValueError(
            f"Review coverage mismatch: {len(missing)} missing and {len(unknown)} unknown."
        )

    annotations: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    overlaps = set(human) & set(assisted)
    for batch_row in batch_rows:
        candidate_id = batch_row["candidate_id"]
        selected = _selected_review(candidate_id, human, assisted)
        assert selected is not None
        origin, review = selected
        exclusion = _exclusion_reason(origin, review)
        reviewer = human_reviewer if origin == "human" else assisted_reviewer
        if exclusion:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "source": batch_row["source"],
                    "original_group_id": batch_row["group_id"],
                    "proposed_label": batch_row.get("proposed_label", ""),
                    "selected_reviewer": reviewer,
                    "decision": review["decision"],
                    "review_label": review.get("review_label", ""),
                    "review_confidence": review.get("review_confidence", ""),
                    "requires_human_review": review.get(
                        "requires_human_review", ""
                    ),
                    "exclusion_reason": exclusion,
                }
            )
            continue

        label = review["review_label"]
        if label not in CLASS_NAMES:
            raise ValueError(f"Candidate {candidate_id}: invalid label {label!r}.")
        source = batch_row["source"]
        if source not in source_prefixes:
            raise ValueError(f"No --source-prefix configured for {source!r}.")
        normalized_path, absolute_path = safe_source_path(
            source_root, source_prefixes[source], batch_row["source_path"]
        )
        digest = sha256_file(absolute_path)
        source_licence_status = batch_row.get("license_status", "")
        review_licence_status = review.get("license_review_status", "")
        release_eligible = (
            source_licence_status == "verified" and review_licence_status == "verified"
        )
        annotations.append(
            {
                "candidate_id": candidate_id,
                "source_path": normalized_path,
                "label": label,
                "group_id": batch_row["group_id"],
                "original_group_id": batch_row["group_id"],
                "source": source,
                "original_sha256": digest,
                "subject_id": batch_row.get("subject_id", ""),
                "session_id": batch_row.get("session_id", ""),
                "device_id": "",
                "skin_tone": batch_row.get("skin_tone", ""),
                "body_site": batch_row.get("body_site", ""),
                "license": batch_row.get("license", ""),
                "consent_scope": "",
                "split": "",
                "is_real_capture": "true",
                "bad_framing": "false",
                "bad_light": "false",
                "blur": "false",
                "low_resolution": "false",
                "too_far": "false",
                "notes": "",
                "annotation_source": origin,
                "annotation_reviewer": reviewer,
                "annotation_confidence": (
                    "human_reviewed"
                    if origin == "human"
                    else review["review_confidence"]
                ),
                "weak_supervision": str(origin == "assisted").lower(),
                "requires_human_review": "false",
                "source_license_status": source_licence_status,
                "release_eligible": str(release_eligible).lower(),
            }
        )

    duplicate_clusters = _merge_duplicate_groups(annotations)
    assign_grouped_splits(annotations, seed=seed)
    groups_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in annotations:
        groups_to_splits[row["group_id"]].add(row["split"])
    if any(len(values) != 1 for values in groups_to_splits.values()):
        raise AssertionError("A group crossed split boundaries.")

    report = {
        "schema_version": 1,
        "task": "capture_semantic_gate_exploratory",
        "seed": seed,
        "batch_candidates": len(batch_rows),
        "retained": len(annotations),
        "excluded": len(exclusions),
        "human_assisted_overlap": len(overlaps),
        "overlap_policy": "human_review_wins",
        "duplicate_hash_clusters": duplicate_clusters,
        "labels": dict(sorted(Counter(row["label"] for row in annotations).items())),
        "annotation_sources": dict(
            sorted(Counter(row["annotation_source"] for row in annotations).items())
        ),
        "exclusion_reasons": dict(
            sorted(Counter(row["exclusion_reason"] for row in exclusions).items())
        ),
        "split_counts": {
            split: dict(
                sorted(
                    Counter(
                        row["label"] for row in annotations if row["split"] == split
                    ).items()
                )
            )
            for split in SPLITS
        },
        "split_annotation_sources": {
            split: dict(
                sorted(
                    Counter(
                        row["annotation_source"]
                        for row in annotations
                        if row["split"] == split
                    ).items()
                )
            )
            for split in SPLITS
        },
        "release_eligible": sum(
            row["release_eligible"] == "true" for row in annotations
        ),
        "evaluation_warning": (
            "Validation and test contain assisted labels; no_skin has no human "
            "gold annotations in this pilot."
        ),
        "release_warning": (
            "Exploratory engineering data only until source licences and trained-"
            "weight permissions are verified."
        ),
    }
    return annotations, exclusions, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exploratory manifest from human and assisted reviews."
    )
    parser.add_argument("batch", type=Path)
    parser.add_argument("human_review", type=Path)
    parser.add_argument("assisted_review", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-prefix", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    batch_rows, _ = load_batch(args.batch.resolve())
    valid_ids = {row["candidate_id"] for row in batch_rows}
    human_reviewer, human = load_review(args.human_review.resolve(), valid_ids)
    assisted_reviewer, assisted = load_review(
        args.assisted_review.resolve(), valid_ids
    )
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    prefixes = parse_source_prefixes(args.source_prefix)
    annotations, exclusions, report = build_exploratory_manifest(
        batch_rows,
        human_reviewer,
        human,
        assisted_reviewer,
        assisted,
        source_root,
        prefixes,
        args.seed,
    )

    output_dir = args.output_dir.resolve()
    outputs = {
        "annotations": output_dir / "annotations.csv",
        "excluded": output_dir / "excluded.csv",
        "report": output_dir / "manifest_report.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite exploratory manifest outputs: "
            + ", ".join(str(path) for path in existing)
        )
    write_csv_atomic(outputs["annotations"], ANNOTATION_COLUMNS, annotations)
    write_csv_atomic(outputs["excluded"], EXCLUSION_COLUMNS, exclusions)
    outputs["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Exploratory manifest written to {output_dir}")


if __name__ == "__main__":
    main()
