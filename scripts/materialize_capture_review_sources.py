"""Package only the PAD-UFES/SCIN images selected for capture review.

The canonical script is staged into a private, CPU-only Kaggle kernel. The
staging step injects the immutable pilot selection into ``SELECTION_JSON``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


SELECTION_JSON = "__SELECTION_JSON__"
SOURCE_CONFIG = {
    "pad_ufes_20": {"marker": "metadata.csv"},
    "scin": {"marker": "scin_cases.csv"},
}
MANIFEST_COLUMNS = [
    "candidate_id",
    "source",
    "source_path",
    "group_id",
    "sha256",
    "bytes",
    "image_width",
    "image_height",
]


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe source_path: {value!r}")
    return Path(*path.parts)


def discover_source_roots(input_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for source, config in SOURCE_CONFIG.items():
        matches = sorted(input_root.rglob(config["marker"]))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {config['marker']!r} for {source}; found {len(matches)}."
            )
        marker = matches[0]
        root = marker.parent
        if source == "scin" and root.name == "dataset":
            root = root.parent
        roots[source] = root
    return roots


def resolve_source_file(root: Path, source_path: str) -> Path:
    relative = safe_relative(source_path)
    direct = (root / relative).resolve()
    if direct.is_relative_to(root) and direct.is_file():
        return direct
    matches = [
        path.resolve()
        for path in root.rglob(relative.name)
        if path.is_file() and path.as_posix().endswith(relative.as_posix())
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not resolve {source_path!r} below {root}; found {len(matches)}."
        )
    return matches[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        image.verify()
    return width, height


def manifest_bytes(rows: list[dict[str, str]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=MANIFEST_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def package_selection(
    selection: list[dict[str, str]],
    source_roots: dict[str, Path],
    archive_path: Path,
) -> dict:
    if not selection:
        raise ValueError("The embedded review selection is empty.")
    identifiers = [row["candidate_id"] for row in selection]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id must be unique in the review selection.")
    unexpected = {row["source"] for row in selection} - set(SOURCE_CONFIG)
    if unexpected:
        raise ValueError(f"Unsupported review sources: {sorted(unexpected)}")
    missing_roots = {row["source"] for row in selection} - set(source_roots)
    if missing_roots:
        raise ValueError(f"Missing source roots: {sorted(missing_roots)}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for row in sorted(selection, key=lambda item: item["candidate_id"]):
            source = row["source"]
            relative = safe_relative(row["source_path"])
            source_file = resolve_source_file(source_roots[source], row["source_path"])
            width, height = inspect_image(source_file)
            digest = file_sha256(source_file)
            archive_name = (PurePosixPath(source) / PurePosixPath(relative.as_posix())).as_posix()
            archive.write(source_file, archive_name)
            manifest.append(
                {
                    "candidate_id": row["candidate_id"],
                    "source": source,
                    "source_path": relative.as_posix(),
                    "group_id": row["group_id"],
                    "sha256": digest,
                    "bytes": str(source_file.stat().st_size),
                    "image_width": str(width),
                    "image_height": str(height),
                }
            )
        archive.writestr("manifest.csv", manifest_bytes(manifest))

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(manifest),
        "groups": len({row["group_id"] for row in manifest}),
        "sources": dict(sorted(Counter(row["source"] for row in manifest).items())),
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": file_sha256(archive_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes(manifest)).hexdigest(),
        "approved_training_samples": 0,
    }


def main() -> None:
    selection = json.loads(SELECTION_JSON)
    roots = discover_source_roots(Path("/kaggle/input"))
    output_root = Path("/kaggle/working")
    archive_path = output_root / "capture_review_sources_v1.zip"
    report = package_selection(selection, roots, archive_path)
    (output_root / "capture_review_sources_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
