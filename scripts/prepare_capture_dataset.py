"""Materialize a reviewed capture-validity dataset from an annotation CSV.

The script deliberately does not infer labels. Every source image must have
been reviewed and assigned one of the three semantic labels in LABELS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


LABELS = {"target_lesion", "skin_without_target", "no_skin"}
REQUIRED_COLUMNS = {"source_path", "label", "group_id", "source"}
OUTPUT_COLUMNS = [
    "candidate_id",
    "image_path",
    "label",
    "group_id",
    "source",
    "sha256",
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
    "original_group_id",
    "original_sha256",
    "annotation_source",
    "annotation_reviewer",
    "annotation_confidence",
    "weak_supervision",
    "requires_human_review",
    "source_license_status",
    "release_eligible",
]
OPTIONAL_COLUMNS = set(OUTPUT_COLUMNS) - {
    "image_path",
    "label",
    "group_id",
    "source",
    "sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centre_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def load_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing annotation columns: {sorted(missing)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("The annotation CSV contains no samples.")
    return rows


def validate_annotation(row: dict[str, str], line_number: int) -> None:
    if row["label"] not in LABELS:
        raise ValueError(
            f"Line {line_number}: unknown label {row['label']!r}."
        )
    for column in ("source_path", "group_id", "source"):
        if not row[column]:
            raise ValueError(f"Line {line_number}: {column} cannot be empty.")


def materialize_image(source: Path, destination: Path, size: int) -> None:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = centre_square(image)
        if size:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            destination,
            format="JPEG",
            quality=95,
            optimize=True,
            progressive=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a private capture-validity dataset from reviewed labels."
    )
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Base for source_path. Defaults to the annotation CSV directory.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Square materialization size; use 0 to preserve crop resolution.",
    )
    args = parser.parse_args()

    annotations_path = args.annotations.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root
        else annotations_path.parent
    )
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() or (output_root.exists() and any(output_root.iterdir())):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset directory: {output_root}"
        )
    if args.size < 0:
        raise ValueError("--size must be zero or a positive integer.")

    annotations = load_annotations(annotations_path)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root / ".building"
    temporary_root.mkdir()

    output_rows: list[dict[str, str]] = []
    labels_by_digest: dict[str, str] = {}
    counts: Counter[str] = Counter()
    try:
        for line_number, row in enumerate(annotations, start=2):
            validate_annotation(row, line_number)
            source_value = PurePosixPath(row["source_path"].replace("\\", "/"))
            if source_value.is_absolute() or ".." in source_value.parts:
                raise ValueError(
                    f"Line {line_number}: source_path must stay inside source-root."
                )
            source_path = source_root.joinpath(*source_value.parts).resolve()
            if not source_path.is_relative_to(source_root):
                raise ValueError(
                    f"Line {line_number}: source_path escapes source-root."
                )
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Line {line_number}: missing source image {source_path}"
                )

            staged = temporary_root / f"{line_number:08d}.jpg"
            materialize_image(source_path, staged, args.size)
            digest = sha256_file(staged)
            previous_label = labels_by_digest.get(digest)
            if previous_label and previous_label != row["label"]:
                raise ValueError(
                    f"Line {line_number}: duplicate image has conflicting labels "
                    f"{previous_label!r} and {row['label']!r}."
                )
            labels_by_digest[digest] = row["label"]

            relative = Path("images") / row["label"] / f"{digest[:20]}.jpg"
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                staged.unlink()
            else:
                staged.replace(destination)

            output_row = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
            output_row["image_path"] = relative.as_posix()
            output_row["sha256"] = digest
            output_rows.append(output_row)
            counts[row["label"]] += 1

        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(output_rows)

        build_report = {
            "samples": len(output_rows),
            "unique_images": len(labels_by_digest),
            "labels": dict(sorted(counts.items())),
            "materialized_size": args.size or "preserved",
            "crop": "centre_square",
            "annotations": annotations_path.name,
        }
        (output_root / "build_report.json").write_text(
            json.dumps(build_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if temporary_root.exists() and not any(temporary_root.iterdir()):
            temporary_root.rmdir()

    print(f"Dataset prepared: {output_root}")
    print(json.dumps(build_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
