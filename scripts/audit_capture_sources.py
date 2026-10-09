"""Audit public sources and build a review queue for capture-validity data.

This is deliberately a curation tool, not an automatic labeller. Source
metadata can propose a semantic label, but every candidate remains pending
until a reviewer confirms it. The resulting CSV is therefore never accepted
directly by the TFRecord builder.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SEMANTIC_LABELS = ("target_lesion", "skin_without_target", "no_skin")
MILESTONES = {
    "target_lesion": 1500,
    "skin_without_target": 1500,
    "no_skin": 2000,
}
MOLE_LIKE_TERMS = (
    "melanocytic nevus",
    "melanoma",
    "atypical nevus",
    "dysplastic nevus",
)
SUFFICIENT_GRADABILITY = {
    "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT",
    "YES",
}
QUEUE_COLUMNS = [
    "candidate_id",
    "source",
    "source_dataset",
    "source_path",
    "group_id",
    "subject_id",
    "session_id",
    "source_split",
    "proposed_label",
    "proposal_reason",
    "eligibility",
    "review_status",
    "review_label",
    "include",
    "license",
    "license_status",
    "condition_labels",
    "shot_type",
    "skin_tone",
    "body_site",
    "image_width",
    "image_height",
    "sha256",
    "file_exists",
    "notes",
]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def find_unique(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {name!r} below {root}; found {len(matches)}."
        )
    return matches[0]


def discover_dataset_root(configured: Path, marker: str, source: str) -> Path:
    """Resolve a Kaggle mount even when its folder name differs from the slug."""
    search_roots = [configured]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and configured != kaggle_input:
        search_roots.append(kaggle_input)
    matches: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            matches = sorted(path for path in search_root.rglob(marker) if path.is_file())
        if len(matches) == 1:
            break
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous {source} mount: found {len(matches)} {marker!r} files "
                f"below {search_root}."
            )
    if len(matches) != 1:
        mounted = (
            sorted(path.name for path in kaggle_input.iterdir())
            if kaggle_input.exists()
            else []
        )
        raise FileNotFoundError(
            f"Could not discover {source}: expected one {marker!r}. "
            f"Configured root={configured}; Kaggle mounts={mounted}."
        )
    marker_path = matches[0]
    if source == "scin" and marker_path.parent.name == "dataset":
        return marker_path.parent.parent
    return marker_path.parent


def normalized_relative(path: str) -> Path:
    value = PurePosixPath(path.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Unsafe source path: {path!r}")
    return Path(*value.parts)


def image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name].append(path)
    return index


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path | None, hash_images: bool) -> dict[str, str]:
    if path is None or not path.is_file():
        return {
            "image_width": "",
            "image_height": "",
            "sha256": "",
            "file_exists": "false",
        }
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        image.verify()
    return {
        "image_width": str(width),
        "image_height": str(height),
        "sha256": file_sha256(path) if hash_images else "",
        "file_exists": "true",
    }


def candidate_id(source: str, group_id: str, source_path: str) -> str:
    value = f"{source}\0{group_id}\0{source_path}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def make_candidate(**values: str) -> dict[str, str]:
    row = {column: "" for column in QUEUE_COLUMNS}
    row.update(values)
    row["review_status"] = "pending"
    row["candidate_id"] = candidate_id(
        row["source"], row["group_id"], row["source_path"]
    )
    return row


def parse_condition_labels(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def audit_pad(root: Path, hash_images: bool) -> tuple[list[dict[str, str]], dict]:
    metadata_path = find_unique(root, "metadata.csv")
    rows = read_csv(
        metadata_path,
        {"patient_id", "lesion_id", "img_id", "diagnostic", "region"},
    )
    images = image_index(root)
    candidates: list[dict[str, str]] = []
    ambiguous_names = 0
    for row in rows:
        matches = images.get(row["img_id"], [])
        if len(matches) > 1:
            ambiguous_names += 1
        image_path = matches[0] if len(matches) == 1 else None
        source_path = row["img_id"]
        if image_path:
            source_path = image_path.relative_to(root).as_posix()
        group_id = f"pad_ufes_20:{row['patient_id']}"
        candidates.append(
            make_candidate(
                source="pad_ufes_20",
                source_dataset="mahdavi1202/skin-cancer",
                source_path=source_path,
                group_id=group_id,
                subject_id=row["patient_id"],
                session_id=row["lesion_id"],
                proposed_label="target_lesion",
                proposal_reason=(
                    "PAD-UFES clinical lesion image; confirm a centered mole-like "
                    "target and usable acquisition quality."
                ),
                eligibility="candidate",
                license="CC BY 4.0 (verify upstream PAD-UFES release)",
                license_status="verify_before_release",
                condition_labels=row["diagnostic"],
                body_site=row["region"],
                **inspect_image(image_path, hash_images),
            )
        )
    report = source_report("pad_ufes_20", candidates)
    report.update(
        metadata_rows=len(rows),
        ambiguous_image_names=ambiguous_names,
        metadata_file=metadata_path.relative_to(root).as_posix(),
    )
    return candidates, report


def audit_scin(root: Path, hash_images: bool) -> tuple[list[dict[str, str]], dict]:
    cases_path = find_unique(root, "scin_cases.csv")
    labels_path = find_unique(root, "scin_labels.csv")
    cases = read_csv(cases_path, {"case_id", "fitzpatrick_skin_type"})
    labels = read_csv(
        labels_path,
        {
            "case_id",
            "dermatologist_skin_condition_on_label_name",
            "dermatologist_gradable_for_skin_condition_1",
        },
    )
    labels_by_case = {row["case_id"]: row for row in labels}
    candidates: list[dict[str, str]] = []
    missing_label_rows = 0
    for case in cases:
        label_row = labels_by_case.get(case["case_id"], {})
        if not label_row:
            missing_label_rows += 1
        conditions = parse_condition_labels(
            label_row.get("dermatologist_skin_condition_on_label_name", "")
        )
        normalized_conditions = " | ".join(conditions)
        mole_like = any(
            term in condition.lower()
            for condition in conditions
            for term in MOLE_LIKE_TERMS
        )
        for index in range(1, 4):
            recorded_path = case.get(f"image_{index}_path", "")
            if not recorded_path:
                continue
            relative = normalized_relative(recorded_path)
            direct = root / relative
            if not direct.is_file():
                matches = list(root.rglob(relative.name))
                direct = matches[0] if len(matches) == 1 else None
            shot_type = case.get(f"image_{index}_shot_type", "")
            gradability = label_row.get(
                f"dermatologist_gradable_for_skin_condition_{index}", ""
            )
            sufficient = gradability in SUFFICIENT_GRADABILITY
            too_far = shot_type == "AT_DISTANCE"
            if not sufficient or too_far:
                proposed_label = ""
                eligibility = "exclude_quality"
                reason = (
                    "Source metadata marks insufficient gradability or an "
                    "at-distance image; keep for deterministic-gate evaluation."
                )
            elif mole_like:
                proposed_label = "target_lesion"
                eligibility = "candidate"
                reason = (
                    "SCIN dermatology labels include a mole-like condition; "
                    "manual review must confirm the central target."
                )
            elif conditions:
                proposed_label = "skin_without_target"
                eligibility = "candidate"
                reason = (
                    "Non-mole dermatological finding; manual review must confirm "
                    "usable skin and absence of a centered mole-like target."
                )
            else:
                proposed_label = ""
                eligibility = "needs_metadata_review"
                reason = "No usable condition label; do not infer a semantic class."
            group_id = f"scin:{case['case_id']}"
            actual_path = direct if isinstance(direct, Path) else None
            source_path = (
                actual_path.relative_to(root).as_posix()
                if actual_path and actual_path.is_relative_to(root)
                else relative.as_posix()
            )
            candidates.append(
                make_candidate(
                    source="scin",
                    source_dataset="emmanuelkiriinya/scin-google-dermatology-dataset",
                    source_path=source_path,
                    group_id=group_id,
                    subject_id=case["case_id"],
                    session_id=case["case_id"],
                    proposed_label=proposed_label,
                    proposal_reason=reason,
                    eligibility=eligibility,
                    license="SCIN Data Use License",
                    license_status="verify_before_use_and_weight_release",
                    condition_labels=normalized_conditions,
                    shot_type=shot_type,
                    skin_tone=case.get("fitzpatrick_skin_type", ""),
                    notes=f"source_gradability={gradability}",
                    **inspect_image(actual_path, hash_images),
                )
            )
    report = source_report("scin", candidates)
    report.update(
        cases=len(cases),
        label_rows=len(labels),
        cases_without_label_row=missing_label_rows,
        cases_file=cases_path.relative_to(root).as_posix(),
        labels_file=labels_path.relative_to(root).as_posix(),
    )
    return candidates, report


def read_source_splits(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for filename, split in (("TrainImages.txt", "train"), ("TestImages.txt", "test")):
        matches = list(root.rglob(filename))
        if len(matches) != 1:
            continue
        for line in matches[0].read_text(encoding="utf-8-sig").splitlines():
            value = line.strip().replace("\\", "/")
            if value:
                result[Path(value).name] = split
    return result


def audit_indoor(
    root: Path,
    hash_images: bool,
    per_category: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict]:
    if per_category <= 0:
        raise ValueError("--indoor-per-category must be positive.")
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    by_category: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        by_category[path.parent.name].append(path)
    source_splits = read_source_splits(root)
    candidates: list[dict[str, str]] = []
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda path: hashlib.sha256(
                f"{seed}:{path.relative_to(root).as_posix()}".encode()
            ).digest(),
        )
        for path in ranked[:per_category]:
            source_path = path.relative_to(root).as_posix()
            group_id = f"mit_indoor:{source_path}"
            candidates.append(
                make_candidate(
                    source="mit_indoor",
                    source_dataset="itsahmad/indoor-scenes-cvpr-2019",
                    source_path=source_path,
                    group_id=group_id,
                    source_split=source_splits.get(path.name, ""),
                    proposed_label="no_skin",
                    proposal_reason=(
                        "Indoor-scene negative; manual review must reject people, "
                        "prominent skin and license-incompatible images."
                    ),
                    eligibility="candidate",
                    license="Research purposes only (official MIT Indoor page)",
                    license_status="blocked_research_only",
                    condition_labels=category,
                    **inspect_image(path, hash_images),
                )
            )
    report = source_report("mit_indoor", candidates)
    report.update(
        available_images=len(images),
        categories=len(by_category),
        selected_per_category=per_category,
    )
    return candidates, report


def source_report(source: str, rows: Iterable[dict[str, str]]) -> dict:
    materialized = list(rows)
    return {
        "source": source,
        "candidates": len(materialized),
        "independent_groups": len({row["group_id"] for row in materialized}),
        "proposed_labels": dict(
            sorted(Counter(row["proposed_label"] or "unassigned" for row in materialized).items())
        ),
        "eligibility": dict(sorted(Counter(row["eligibility"] for row in materialized).items())),
        "missing_files": sum(row["file_exists"] != "true" for row in materialized),
        "license_status": dict(
            sorted(Counter(row["license_status"] for row in materialized).items())
        ),
    }


def write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict) -> str:
    lines = [
        "# Capture source curation audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This report contains proposals only. No row is approved for training until "
        "manual review and license checks are complete.",
        "",
        "## Readiness",
        "",
        f"- Overall status: **{report['readiness']['status']}**",
        f"- Pending manual review: {report['readiness']['pending_review']}",
        f"- Missing image files: {report['readiness']['missing_files']}",
        f"- Rows with unresolved licence review: {report['readiness']['license_unresolved']}",
        "- Approved training samples: 0",
        "",
        "## Proposed eligible candidates",
        "",
        "| Class | Proposed | Bootstrap goal | Gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in SEMANTIC_LABELS:
        item = report["class_readiness"][label]
        lines.append(
            f"| `{label}` | {item['proposed_eligible']} | {item['goal']} | {item['gap']} |"
        )
    lines.extend(["", "## Sources", ""])
    for source in report["sources"]:
        lines.extend(
            [
                f"### {source['source']}",
                "",
                f"- Candidates: {source['candidates']}",
                f"- Independent groups: {source['independent_groups']}",
                f"- Proposed labels: `{json.dumps(source['proposed_labels'], sort_keys=True)}`",
                f"- Eligibility: `{json.dumps(source['eligibility'], sort_keys=True)}`",
                f"- Missing files: {source['missing_files']}",
                f"- License state: `{json.dumps(source['license_status'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "Do not build TFRecords yet. Review `capture_review_queue.csv`, resolve "
            "source rights, freeze accepted labels, and only then materialize the "
            "private dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(rows: list[dict[str, str]], sources: list[dict], hash_images: bool) -> dict:
    eligible = [row for row in rows if row["eligibility"] == "candidate"]
    proposed = Counter(row["proposed_label"] for row in eligible)
    class_readiness = {
        label: {
            "proposed_eligible": proposed[label],
            "goal": goal,
            "gap": max(goal - proposed[label], 0),
        }
        for label, goal in MILESTONES.items()
    }
    missing = sum(row["file_exists"] != "true" for row in rows)
    unresolved = sum(row["license_status"] != "approved" for row in rows)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "capture_semantic_gate_source_curation",
        "hash_images": hash_images,
        "queue_rows": len(rows),
        "sources": sources,
        "class_readiness": class_readiness,
        "readiness": {
            "status": "BLOCKED_PENDING_REVIEW",
            "pending_review": len(rows),
            "missing_files": missing,
            "license_unresolved": unresolved,
            "approved_training_samples": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit capture-validity source datasets without assigning final labels."
    )
    parser.add_argument(
        "--pad-root",
        type=Path,
        default=Path("/kaggle/input/skin-cancer"),
    )
    parser.add_argument(
        "--scin-root",
        type=Path,
        default=Path("/kaggle/input/scin-google-dermatology-dataset"),
    )
    parser.add_argument(
        "--indoor-root",
        type=Path,
        default=Path("/kaggle/input/indoor-scenes-cvpr-2019"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/capture_curation"),
    )
    parser.add_argument("--indoor-per-category", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Hash all selected images. Useful for the final audit, slower for discovery.",
    )
    args = parser.parse_args()

    pad_root = discover_dataset_root(args.pad_root.resolve(), "metadata.csv", "pad_ufes_20")
    scin_root = discover_dataset_root(args.scin_root.resolve(), "scin_cases.csv", "scin")
    indoor_root = discover_dataset_root(
        args.indoor_root.resolve(), "TrainImages.txt", "mit_indoor"
    )

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    source_reports: list[dict] = []
    for audit in (
        lambda: audit_pad(pad_root, args.hash_images),
        lambda: audit_scin(scin_root, args.hash_images),
        lambda: audit_indoor(
            indoor_root,
            args.hash_images,
            args.indoor_per_category,
            args.seed,
        ),
    ):
        rows, source = audit()
        all_rows.extend(rows)
        source_reports.append(source)

    identifiers = [row["candidate_id"] for row in all_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate identifiers are not unique.")

    all_rows.sort(key=lambda row: (row["source"], row["group_id"], row["source_path"]))
    report = build_report(all_rows, source_reports, args.hash_images)
    write_queue(output_root / "capture_review_queue.csv", all_rows)
    (output_root / "capture_source_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "capture_source_audit.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(markdown_report(report))


if __name__ == "__main__":
    main()
