"""Build a rights-traceable Open Images V7 ``no_skin`` review queue.

The adapter only proposes labels. It requires a human-verified positive label
for an approved scene/object class, an explicit human-verified negative label
for Person, and no positive label for any configured human/body class. A
reviewer must still inspect every selected image before it can enter training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from audit_capture_sources import QUEUE_COLUMNS, candidate_id, inspect_image


CC_BY_2_0 = "https://creativecommons.org/licenses/by/2.0/"
DOWNLOAD_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"

SAFE_SCENE_CLASSES = {
    "/m/01j2bj": "Bathroom",
    "/m/02_58j": "Bedroom",
    "/m/03__z0": "Bookcase",
    "/m/0cgh4": "Building",
    "/m/01s105": "Cabinetry",
    "/m/03gfsp": "Ceiling",
    "/m/02522": "Computer monitor",
    "/m/0b3fp9": "Countertop",
    "/m/01y9k5": "Desk",
    "/m/02dgv": "Door",
    "/m/0l7_8": "Floor",
    "/m/0c_jw": "Furniture",
    "/m/03jm5": "House",
    "/m/0d4wf": "Kitchen",
    "/m/0fx9l": "Microwave oven",
    "/m/040b_t": "Refrigerator",
    "/m/0gjbg72": "Shelf",
    "/m/04bcr3": "Table",
    "/m/07c52": "Television",
    "/m/09qqq": "Wall",
    "/m/0174k2": "Washing machine",
    "/m/0d4v4": "Window",
}

HUMAN_CLASSES = {
    "/m/01g317": "Person",
    "/m/02p0tk3": "Human body",
    "/m/0dzct": "Human face",
    "/m/04yx4": "Man",
    "/m/03bt1vf": "Woman",
    "/m/01bl7v": "Boy",
    "/m/05r655": "Girl",
    "/m/0ytgt": "Child",
    "/m/0jnvp": "Baby",
    "/m/06z04": "Skin",
    "/oi/m_0dzf4": "Arm",
    "/oi/m_035r7c": "Leg",
    "/oi/m_0k65p": "Hand",
    "/oi/m_031n1": "Foot",
}
PERSON_MID = "/m/01g317"

ATTRIBUTION_COLUMNS = [
    "openimages_image_id",
    "original_url",
    "original_landing_url",
    "author_profile_url",
    "author",
    "title",
    "original_size",
    "original_md5",
    "rotation",
    "positive_scene_labels",
    "negative_human_labels",
    "person_negative_sources",
    "download_url",
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


def validate_class_descriptions(path: Path) -> None:
    rows = read_csv(path, {"LabelName", "DisplayName"})
    actual = {row["LabelName"]: row["DisplayName"] for row in rows}
    expected = SAFE_SCENE_CLASSES | HUMAN_CLASSES
    mismatches = {
        mid: (name, actual.get(mid, "<missing>"))
        for mid, name in expected.items()
        if actual.get(mid) != name
    }
    if mismatches:
        raise ValueError(f"Open Images class descriptions do not match: {mismatches}")


def load_label_evidence(path: Path) -> dict[str, dict[str, dict[str, set[str]]]]:
    relevant = set(SAFE_SCENE_CLASSES) | set(HUMAN_CLASSES)
    evidence: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"positive": set(), "negative": set()})
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "Source", "LabelName", "Confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            mid = (row.get("LabelName") or "").strip()
            if mid not in relevant:
                continue
            confidence = float((row.get("Confidence") or "").strip())
            state = "positive" if confidence == 1.0 else "negative"
            if confidence not in {0.0, 1.0}:
                raise ValueError(f"Unexpected confidence {confidence} for {mid}")
            evidence[row["ImageID"]][mid][state].add(row["Source"])
    return evidence


def qualifies(labels: dict[str, dict[str, set[str]]]) -> bool:
    positive_scene = any(labels[mid]["positive"] for mid in SAFE_SCENE_CLASSES)
    person_explicitly_absent = bool(labels[PERSON_MID]["negative"])
    positive_human = any(labels[mid]["positive"] for mid in HUMAN_CLASSES)
    return positive_scene and person_explicitly_absent and not positive_human


def stable_key(seed: int, *parts: str) -> bytes:
    return hashlib.sha256("\0".join((str(seed), *parts)).encode()).digest()


def select_diverse(
    candidates: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    by_primary: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        primary = row["positive_scene_labels"].split(" | ", 1)[0]
        by_primary[primary].append(row)
    for primary, rows in by_primary.items():
        rows.sort(key=lambda row: stable_key(seed, primary, row["openimages_image_id"]))
    selected: list[dict[str, str]] = []
    strata = sorted(by_primary)
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if by_primary[stratum]:
                selected.append(by_primary[stratum].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def build_candidates(
    class_descriptions: Path,
    labels_path: Path,
    metadata_path: Path,
    count: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict]:
    validate_class_descriptions(class_descriptions)
    evidence = load_label_evidence(labels_path)
    metadata = read_csv(
        metadata_path,
        {
            "ImageID",
            "Subset",
            "OriginalURL",
            "OriginalLandingURL",
            "License",
            "AuthorProfileURL",
            "Author",
            "Title",
            "OriginalSize",
            "OriginalMD5",
            "Rotation",
        },
    )
    eligible: list[dict[str, str]] = []
    rejection = Counter()
    for meta in metadata:
        labels = evidence.get(meta["ImageID"])
        if not labels or not qualifies(labels):
            rejection["label_evidence"] += 1
            continue
        if meta["Subset"] != "validation":
            rejection["wrong_split"] += 1
            continue
        if meta["License"] != CC_BY_2_0:
            rejection["license"] += 1
            continue
        if not meta["OriginalLandingURL"] or not meta["Author"]:
            rejection["missing_attribution"] += 1
            continue
        positive = sorted(
            SAFE_SCENE_CLASSES[mid]
            for mid in SAFE_SCENE_CLASSES
            if labels[mid]["positive"]
        )
        negative = sorted(
            HUMAN_CLASSES[mid]
            for mid in HUMAN_CLASSES
            if labels[mid]["negative"] and not labels[mid]["positive"]
        )
        image_id = meta["ImageID"]
        eligible.append(
            {
                "openimages_image_id": image_id,
                "source": "open_images_v7",
                "source_dataset": "Open Images Dataset V7 validation",
                "source_split": "validation",
                "group_id": f"open_images_v7:{image_id}",
                "proposed_label": "no_skin",
                "proposal_reason": (
                    "Human-verified scene/object label plus explicit Person=0; "
                    "manual review must still reject any visible skin or person."
                ),
                "eligibility": "candidate",
                "review_status": "pending",
                "review_label": "",
                "include": "",
                "license": meta["License"],
                "license_status": "metadata_cc_by_2_0_verify_original",
                "condition_labels": " | ".join(positive),
                "positive_scene_labels": " | ".join(positive),
                "negative_human_labels": " | ".join(negative),
                "person_negative_sources": " | ".join(
                    sorted(labels[PERSON_MID]["negative"])
                ),
                "original_url": meta["OriginalURL"],
                "original_landing_url": meta["OriginalLandingURL"],
                "author_profile_url": meta["AuthorProfileURL"],
                "author": meta["Author"],
                "title": meta["Title"],
                "original_size": meta["OriginalSize"],
                "original_md5": meta["OriginalMD5"],
                "rotation": meta["Rotation"],
                "download_url": DOWNLOAD_URL.format(image_id=image_id),
                "notes": "Open Images labels are screening evidence, not a final annotation.",
            }
        )
    selected = select_diverse(eligible, min(count, len(eligible)), seed)
    report = {
        "metadata_rows": len(metadata),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "shortfall": max(count - len(selected), 0),
        "rejection": dict(sorted(rejection.items())),
        "selected_scene_labels": dict(
            sorted(Counter(row["positive_scene_labels"].split(" | ", 1)[0] for row in selected).items())
        ),
    }
    return selected, report


def download_one(row: dict[str, str], image_root: Path, retries: int = 3) -> None:
    destination = image_root / f"{row['openimages_image_id']}.jpg"
    if destination.is_file():
        return
    temporary = destination.with_suffix(".jpg.part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                row["download_url"], headers={"User-Agent": "skincare-ml-curation/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            temporary.replace(destination)
            inspect_image(destination, hash_images=False)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)


def materialize_queue(
    selected: list[dict[str, str]], output_root: Path, workers: int
) -> list[dict[str, str]]:
    image_root = output_root / "images" / "no_skin"
    image_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, row, image_root) for row in selected]
        for future in as_completed(futures):
            future.result()
    queue: list[dict[str, str]] = []
    for row in selected:
        source_path = f"images/no_skin/{row['openimages_image_id']}.jpg"
        image_path = output_root / Path(source_path)
        completed = {column: "" for column in QUEUE_COLUMNS + ATTRIBUTION_COLUMNS}
        completed.update(row)
        completed["source_path"] = source_path
        completed["candidate_id"] = candidate_id(
            completed["source"], completed["group_id"], source_path
        )
        completed.update(inspect_image(image_path, hash_images=True))
        queue.append(completed)
    return queue


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_queue(base_path: Path, rows: list[dict[str, str]], output_path: Path) -> None:
    with base_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames or [])
        base_rows = list(reader)
    fields = base_fields + [
        field for field in ATTRIBUTION_COLUMNS if field not in base_fields
    ]
    write_csv(output_path, base_rows + rows, fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--count", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-queue", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already downloaded images and refresh queue/report files.",
    )
    args = parser.parse_args()
    if args.count <= 0 or args.workers <= 0:
        raise ValueError("--count and --workers must be positive.")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected, report = build_candidates(
        args.metadata_root / "oidv7-class-descriptions.csv",
        args.metadata_root / "oidv7-val-annotations-human-imagelabels.csv",
        args.metadata_root / "validation-images-with-rotation.csv",
        args.count,
        args.seed,
    )
    queue = materialize_queue(selected, output_root, args.workers)
    fields = QUEUE_COLUMNS + [
        field for field in ATTRIBUTION_COLUMNS if field not in QUEUE_COLUMNS
    ]
    write_csv(output_root / "openimages_no_skin_review_queue.csv", queue, fields)
    if args.base_queue:
        merge_queue(
            args.base_queue.resolve(),
            queue,
            output_root / "combined_capture_review_queue.csv",
        )
    report.update(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "downloaded_rows": len(queue),
            "independent_groups": len({row["group_id"] for row in queue}),
            "license_status": dict(Counter(row["license_status"] for row in queue)),
            "approved_training_samples": 0,
        }
    )
    (output_root / "openimages_no_skin_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
