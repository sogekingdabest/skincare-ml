"""Validate and stage a private Kaggle TFRecord dataset upload.

The staged directory must contain TFRecords and their audit metadata, never
source pixels. Kaggle datasets are private by default; the documented upload
command deliberately omits ``--public``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_capture_tfrecords import CLASS_NAMES, SPLITS, sha256_file


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_ID = "daniolaetafaria/skincare-capture-validity-tfrecords"
DEFAULT_TITLE = "SkinCare - Capture Validity TFRecords (Private)"


def validate_release(root: Path) -> dict:
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError("Unsupported TFRecord schema version.")
    if metadata.get("classes") != list(CLASS_NAMES):
        raise ValueError("Unexpected capture-validity class contract.")
    if metadata.get("publication_scope") != "private_training_only":
        raise ValueError(
            "This staging command is only for the private exploratory release."
        )
    if metadata.get("release_eligible") is not False:
        raise ValueError("Exploratory TFRecords must not be marked release eligible.")
    weight_metadata = metadata.get("sample_weight", {})
    if weight_metadata.get("feature") != "sample_weight":
        raise ValueError("Missing v4 sample-weight contract.")
    if weight_metadata.get("held_out_all_standard") is not True:
        raise ValueError("Validation and test weights must remain standard.")

    images = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if images:
        raise ValueError(f"Refusing to upload {len(images)} source image files.")

    shards = metadata.get("shards", [])
    if not shards:
        raise ValueError("No TFRecord shards are declared.")
    observed_splits = set()
    for shard in shards:
        path = root / shard["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != shard["bytes"]:
            raise ValueError(f"Size mismatch for {path.name}.")
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path.name}.")
        observed_splits.add(shard["split"])
    if observed_splits != set(SPLITS):
        raise ValueError(f"Unexpected shard splits: {sorted(observed_splits)}")
    return metadata


def stage(root: Path, dataset_id: str, title: str) -> Path:
    metadata = validate_release(root)
    kaggle_metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "other"}],
    }
    destination = root / "dataset-metadata.json"
    destination.write_text(
        json.dumps(kaggle_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    notice = root / "PRIVATE_DATASET_NOTICE.md"
    notice.write_text(
        "# Private exploratory training data\n\n"
        "This package contains TFRecords only; it intentionally contains no "
        "source image files. It is restricted to private exploratory training "
        "and must not be made public. Reconstruct source pixels from their "
        "official upstream datasets using the repository scripts.\n\n"
        f"Examples: {metadata['examples']}  \n"
        f"Manifest SHA-256: `{metadata['manifest_sha256']}`\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--id", default=DEFAULT_ID)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = stage(root, args.id, args.title)
    print(f"Staged private Kaggle metadata: {destination}")
    print("Upload without --public: kaggle datasets create -p <root> --dir-mode zip")


if __name__ == "__main__":
    main()
