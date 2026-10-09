"""Prepare and validate an auditable, vision-assisted capture review.

The tool never calls a remote model. ``prepare`` creates deterministic contact
sheets plus an annotation queue that a vision-capable reviewer can inspect.
``finalize`` converts completed queue rows into the independent review schema
consumed by ``adjudicate_capture_reviews.py``.

Licences are deliberately left unreviewed: visual inspection cannot verify a
dataset licence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from review_capture_batch import (
    ImageResolver,
    LABELS,
    QUALITY_REASONS,
    REVIEWER_PATTERN,
    REVIEW_COLUMNS,
    parse_source_roots,
    read_batch,
)


CONFIDENCE_LEVELS = ("high", "medium", "low")
QUEUE_COLUMNS = [
    "sequence",
    "sheet",
    "position",
    "candidate_id",
    "source",
    "source_path",
    "proposed_label",
    "image_available",
    "quality_status",
    "quality_reason",
    "review_label",
    "exclusion_reason",
    "review_confidence",
    "reviewer_notes",
    "reviewed_at",
]
ASSISTED_REVIEW_COLUMNS = REVIEW_COLUMNS + [
    "review_confidence",
    "requires_human_review",
    "assistance_method",
]
TILE_SIZE = 320
LABEL_HEIGHT = 42


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, fields


def reviewed_ids(paths: list[Path]) -> set[str]:
    identifiers: set[str] = set()
    for path in paths:
        rows, fields = read_csv(path)
        if "candidate_id" not in fields:
            raise ValueError(f"{path}: missing candidate_id column.")
        identifiers.update(row["candidate_id"] for row in rows if row["candidate_id"])
    return identifiers


def center_square(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def render_sheet(
    entries: list[dict[str, str]],
    resolver: ImageResolver,
    path: Path,
    columns: int,
) -> None:
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * TILE_SIZE, rows * (TILE_SIZE + LABEL_HEIGHT)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, entry in enumerate(entries):
        x = (index % columns) * TILE_SIZE
        y = (index // columns) * (TILE_SIZE + LABEL_HEIGHT)
        image_path = resolver.get(entry["candidate_id"])
        if image_path is None:
            tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "#3a3a3a")
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.text((18, TILE_SIZE // 2), "IMAGE NOT FOUND", fill="white", font=font)
        else:
            with Image.open(image_path) as source:
                tile = ImageOps.fit(
                    center_square(source),
                    (TILE_SIZE, TILE_SIZE),
                    method=Image.Resampling.LANCZOS,
                )
        sheet.paste(tile, (x, y))
        label = f"{entry['position']}  {entry['candidate_id'][:10]}"
        draw.rectangle(
            (x, y + TILE_SIZE, x + TILE_SIZE, y + TILE_SIZE + LABEL_HEIGHT),
            fill="white",
        )
        draw.text((x + 8, y + TILE_SIZE + 8), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=92, optimize=True)


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def prepare(args: argparse.Namespace) -> dict:
    batch_path = args.batch.resolve()
    rows, _ = read_batch(batch_path)
    skipped = reviewed_ids([path.resolve() for path in args.skip_review])
    valid_ids = {row["candidate_id"] for row in rows}
    unknown = skipped - valid_ids
    if unknown:
        raise ValueError(f"Skip reviews contain {len(unknown)} candidates outside the batch.")
    selected = [row for row in rows if row["candidate_id"] not in skipped]
    if args.label:
        selected = [row for row in selected if row["proposed_label"] == args.label]
    if not selected:
        raise ValueError("No candidates remain after applying the filters.")

    output_dir = args.output_dir.resolve()
    queue_path = output_dir / "assisted_review_queue.csv"
    sheets_dir = output_dir / "sheets"
    if queue_path.exists() or sheets_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing assisted-review pack: {output_dir}"
        )

    roots = parse_source_roots(args.source_root)
    resolver = ImageResolver(batch_path.parent, roots, selected)
    queue: list[dict[str, str]] = []
    for offset, row in enumerate(selected, start=1):
        sheet_number = (offset - 1) // args.sheet_size + 1
        position = (offset - 1) % args.sheet_size + 1
        queue.append(
            {
                "sequence": str(offset),
                "sheet": f"sheet_{sheet_number:03d}.jpg",
                "position": f"{position:02d}",
                "candidate_id": row["candidate_id"],
                "source": row["source"],
                "source_path": row["source_path"],
                "proposed_label": row["proposed_label"],
                "image_available": str(resolver.available(row["candidate_id"])).lower(),
                "quality_status": "",
                "quality_reason": "",
                "review_label": "",
                "exclusion_reason": "",
                "review_confidence": "",
                "reviewer_notes": "",
                "reviewed_at": "",
            }
        )

    for start in range(0, len(queue), args.sheet_size):
        entries = queue[start : start + args.sheet_size]
        render_sheet(entries, resolver, sheets_dir / entries[0]["sheet"], args.columns)
    write_csv_atomic(queue_path, QUEUE_COLUMNS, queue)

    report = {
        "batch": str(batch_path),
        "queue": str(queue_path),
        "total_batch": len(rows),
        "skipped_existing_reviews": len(skipped),
        "queued": len(queue),
        "available_images": sum(row["image_available"] == "true" for row in queue),
        "sheets": len({row["sheet"] for row in queue}),
        "sheet_size": args.sheet_size,
        "center_square_preview": True,
    }
    (output_dir / "pack_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def normalize_queue_row(row: dict[str, str], reviewer: str) -> dict[str, str] | None:
    quality = row["quality_status"]
    quality_reason = row["quality_reason"]
    label = row["review_label"]
    exclusion = row["exclusion_reason"]
    confidence = row["review_confidence"]
    touched = any((quality, quality_reason, label, exclusion, confidence))
    if not touched:
        return None
    if quality not in {"pass", "fail"}:
        raise ValueError("quality_status must be pass or fail.")
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"review_confidence must be one of {CONFIDENCE_LEVELS}.")
    if quality == "fail":
        if quality_reason not in QUALITY_REASONS:
            raise ValueError(f"quality failure requires one of {QUALITY_REASONS}.")
        if label or exclusion:
            raise ValueError("quality failures cannot also carry a semantic decision.")
        decision = "exclude"
        exclusion = quality_reason
    else:
        if quality_reason:
            raise ValueError("quality pass cannot carry quality_reason.")
        if label in LABELS and not exclusion:
            decision = "include"
        elif not label and exclusion:
            decision = "exclude"
        else:
            raise ValueError(
                "quality pass requires exactly one review_label or exclusion_reason."
            )
    reviewed_at = row["reviewed_at"] or datetime.now(timezone.utc).isoformat()
    requires_human = confidence != "high" or decision == "exclude"
    return {
        "candidate_id": row["candidate_id"],
        "reviewer_id": reviewer,
        "reviewed_at": reviewed_at,
        "quality_status": quality,
        "quality_reason": quality_reason if quality == "fail" else "",
        "decision": decision,
        "review_label": label if decision == "include" else "",
        "include": str(decision == "include").lower(),
        "exclusion_reason": exclusion if decision == "exclude" else "",
        "reviewer_notes": row["reviewer_notes"][:2000],
        "license_review_status": "unreviewed",
        "license_review_notes": "Visual assistance does not verify licences.",
        "review_confidence": confidence,
        "requires_human_review": str(requires_human).lower(),
        "assistance_method": "codex_vision_manual_v1",
    }


def finalize(args: argparse.Namespace) -> dict:
    if not REVIEWER_PATTERN.fullmatch(args.reviewer):
        raise ValueError("Invalid reviewer identifier.")
    batch_rows, _ = read_batch(args.batch.resolve())
    valid_ids = {row["candidate_id"] for row in batch_rows}
    queue, fields = read_csv(args.queue.resolve())
    missing = set(QUEUE_COLUMNS) - set(fields)
    if missing:
        raise ValueError(f"Queue is missing columns: {sorted(missing)}")
    identifiers = [row["candidate_id"] for row in queue]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Queue candidate_id values must be unique.")
    unknown = set(identifiers) - valid_ids
    if unknown:
        raise ValueError(f"Queue contains {len(unknown)} candidates outside the batch.")

    reviews: list[dict[str, str]] = []
    errors: list[str] = []
    for line_number, row in enumerate(queue, start=2):
        try:
            review = normalize_queue_row(row, args.reviewer)
            if review is not None:
                reviews.append(review)
        except ValueError as error:
            errors.append(f"line {line_number} ({row['candidate_id']}): {error}")
    if errors:
        raise ValueError("Invalid assisted decisions:\n" + "\n".join(errors[:20]))
    if not args.allow_partial and len(reviews) != len(queue):
        raise ValueError(
            f"Queue is incomplete: {len(reviews)}/{len(queue)} rows classified. "
            "Use --allow-partial only for an explicitly partial review file."
        )
    if not reviews:
        raise ValueError("No completed assisted decisions were found.")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review file: {output}")
    reviews.sort(key=lambda row: row["candidate_id"])
    write_csv_atomic(output, ASSISTED_REVIEW_COLUMNS, reviews)
    return {
        "output": str(output),
        "reviewer": args.reviewer,
        "queue_rows": len(queue),
        "completed": len(reviews),
        "partial": len(reviews) != len(queue),
        "confidence": dict(
            sorted(Counter(row["review_confidence"] for row in reviews).items())
        ),
        "requires_human_review": sum(
            row["requires_human_review"] == "true" for row in reviews
        ),
        "licence_review_status": "unreviewed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create contact sheets and queue.")
    prepare_parser.add_argument("batch", type=Path)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument(
        "--source-root", action="append", default=[], metavar="SOURCE=PATH"
    )
    prepare_parser.add_argument("--skip-review", action="append", default=[], type=Path)
    prepare_parser.add_argument("--label", choices=LABELS)
    prepare_parser.add_argument("--sheet-size", type=int, default=16)
    prepare_parser.add_argument("--columns", type=int, default=4)

    finalize_parser = subparsers.add_parser("finalize", help="Build a strict review CSV.")
    finalize_parser.add_argument("batch", type=Path)
    finalize_parser.add_argument("queue", type=Path)
    finalize_parser.add_argument("--reviewer", required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        if not 1 <= args.sheet_size <= 30:
            raise ValueError("--sheet-size must be between 1 and 30.")
        if not 1 <= args.columns <= args.sheet_size:
            raise ValueError("--columns must be between 1 and --sheet-size.")
        report = prepare(args)
    else:
        report = finalize(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
