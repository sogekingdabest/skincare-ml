"""Validate a curated capture-validity manifest without loading TensorFlow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {"target_lesion", "skin_without_target", "no_skin"}
SPLITS = {"train", "validation", "test"}
REQUIRED = {"image_path", "label", "group_id", "source", "sha256"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Validate structure without recomputing every image SHA-256.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a work-in-progress manifest that does not contain every class.",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Manifest contains no samples.")

    labels = Counter()
    sources = Counter()
    groups_to_splits: dict[str, set[str]] = defaultdict(set)
    hashes_to_splits: dict[str, set[str]] = defaultdict(set)
    missing_files = []
    hash_mismatches = []
    invalid_rows = []
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    split_values = []

    for line_number, row in enumerate(rows, start=2):
        label = row["label"].strip()
        if label not in LABELS:
            raise ValueError(f"Line {line_number}: unknown label {label!r}")
        labels[label] += 1
        source = row["source"].strip()
        group_id = row["group_id"].strip()
        digest = row["sha256"].strip().lower()
        image_value = row["image_path"].strip().replace("\\", "/")
        if not source or not group_id or not image_value:
            invalid_rows.append(f"line {line_number} has empty required values")
        if not SHA256_PATTERN.fullmatch(digest):
            invalid_rows.append(f"line {line_number} has an invalid SHA-256")
        if Path(image_value).is_absolute() or ".." in Path(image_value).parts:
            invalid_rows.append(f"line {line_number} has an unsafe image_path")
        sources[source] += 1
        labels_by_hash[digest].add(label)

        split = row.get("split", "").strip().lower()
        split_values.append(split)
        if split and split not in SPLITS:
            raise ValueError(f"Line {line_number}: unknown split {split!r}")
        if split:
            groups_to_splits[group_id].add(split)
            hashes_to_splits[digest].add(split)

        image_path = (root / image_value).resolve()
        if not image_path.is_relative_to(root):
            invalid_rows.append(f"line {line_number} escapes the dataset root")
            continue
        if not image_path.is_file():
            missing_files.append(str(image_path))
            continue
        if not args.skip_hash_check:
            expected = digest
            actual = sha256_file(image_path)
            if expected != actual:
                hash_mismatches.append((str(image_path), expected, actual))

    leaking_groups = {
        group: splits
        for group, splits in groups_to_splits.items()
        if len(splits) > 1
    }
    leaking_hashes = {
        digest: splits
        for digest, splits in hashes_to_splits.items()
        if len(splits) > 1
    }
    errors = []
    missing_labels = LABELS - set(labels)
    if missing_labels and not args.allow_incomplete:
        errors.append(f"missing classes: {sorted(missing_labels)}")
    if any(split_values) and not all(split_values):
        errors.append("split must be populated for every row or omitted for every row")
    conflicting_hashes = {
        digest: digest_labels
        for digest, digest_labels in labels_by_hash.items()
        if len(digest_labels) > 1
    }
    if conflicting_hashes:
        errors.append(f"{len(conflicting_hashes)} duplicates have conflicting labels")
    if invalid_rows:
        errors.append(f"{len(invalid_rows)} rows violate the manifest contract")
    if missing_files:
        errors.append(f"{len(missing_files)} image files are missing")
    if hash_mismatches:
        errors.append(f"{len(hash_mismatches)} SHA-256 values do not match")
    if leaking_groups:
        errors.append(f"{len(leaking_groups)} groups cross split boundaries")
    if leaking_hashes:
        errors.append(f"{len(leaking_hashes)} duplicates cross split boundaries")
    if errors:
        raise RuntimeError("; ".join(errors))

    print(f"Manifest validation: OK ({len(rows):,} images)")
    print(f"Labels: {dict(sorted(labels.items()))}")
    print(f"Sources: {dict(sorted(sources.items()))}")
    if all(split_values):
        split_counts = Counter(split_values)
        print(f"Splits: {dict(sorted(split_counts.items()))}")


if __name__ == "__main__":
    main()
