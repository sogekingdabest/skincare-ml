"""Safely extract and verify a Kaggle capture-review source archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


REQUIRED_MANIFEST = {
    "candidate_id",
    "source",
    "source_path",
    "group_id",
    "sha256",
    "bytes",
    "image_width",
    "image_height",
}
EXPECTED_SOURCES = {"pad_ufes_20", "scin"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return Path(*value.parts)


def read_manifest(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    try:
        payload = archive.read("manifest.csv")
    except KeyError as error:
        raise ValueError("Archive does not contain manifest.csv.") from error
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    missing = REQUIRED_MANIFEST - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    rows = [
        {key: (value or "").strip() for key, value in row.items()}
        for row in reader
    ]
    if not rows:
        raise ValueError("Archive manifest is empty.")
    return rows


def expected_from_batch(path: Path) -> set[tuple[str, str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (row["candidate_id"], row["source"], row["source_path"])
            for row in reader
            if row["source"] in EXPECTED_SOURCES
        }


def extract_archive(
    archive_path: Path,
    output_root: Path,
    batch_path: Path | None = None,
) -> dict:
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing {output_root}")
    staging = output_root.with_name(f".{output_root.name}.building")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            for name in members:
                safe_member(name)
            manifest = read_manifest(archive)
            archive.extractall(staging)

        manifest_ids = [row["candidate_id"] for row in manifest]
        if len(manifest_ids) != len(set(manifest_ids)):
            raise ValueError("Manifest candidate_id values are not unique.")
        if batch_path:
            expected = expected_from_batch(batch_path.resolve())
            actual = {
                (row["candidate_id"], row["source"], row["source_path"])
                for row in manifest
            }
            if actual != expected:
                raise ValueError(
                    f"Archive selection differs from batch: expected {len(expected)}, got {len(actual)}."
                )

        for row in manifest:
            relative = safe_member(
                (PurePosixPath(row["source"]) / row["source_path"]).as_posix()
            )
            image_path = (staging / relative).resolve()
            if not image_path.is_relative_to(staging) or not image_path.is_file():
                raise FileNotFoundError(relative)
            if image_path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"Size mismatch for {relative.as_posix()}")
            if sha256_file(image_path) != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {relative.as_posix()}")
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened)
                if image.size != (int(row["image_width"]), int(row["image_height"])):
                    raise ValueError(f"Dimension mismatch for {relative.as_posix()}")
                image.verify()

        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    report = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "rows": len(manifest),
        "groups": len({row["group_id"] for row in manifest}),
        "sources": dict(sorted(Counter(row["source"] for row in manifest).items())),
        "approved_training_samples": 0,
    }
    (output_root / "local_verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args()
    report = extract_archive(args.archive, args.output_root, args.batch)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
