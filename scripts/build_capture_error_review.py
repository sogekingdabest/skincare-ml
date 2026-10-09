from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REVIEW_FIELDS = (
    "review_id",
    "split",
    "sha256",
    "source",
    "group_id",
    "image_path",
    "true_label",
    "predicted_label",
    "probability_target_lesion",
    "probability_skin_without_target",
    "probability_no_skin",
    "predicted_ready",
    "error_types",
    "review_decision",
    "review_notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_review_queue(
    predictions: list[dict[str, str]],
    manifest: list[dict[str, str]],
    dataset_root: Path,
) -> list[dict[str, str]]:
    by_digest = {row["sha256"].lower(): row for row in manifest}
    queue: list[dict[str, str]] = []
    for prediction in predictions:
        if not prediction.get("error_types"):
            continue
        digest = prediction["sha256"].lower()
        source = by_digest.get(digest)
        if source is None:
            raise ValueError(f"Prediction digest absent from manifest: {digest}")
        image_path = (dataset_root / source["image_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        queue.append(
            {
                "review_id": f"{prediction['split']}-{int(prediction['record_index']):04d}",
                "split": prediction["split"],
                "sha256": digest,
                "source": prediction["source"],
                "group_id": prediction["group_id"],
                "image_path": str(image_path),
                "true_label": prediction["true_label"],
                "predicted_label": prediction["predicted_label"],
                "probability_target_lesion": prediction[
                    "probability_target_lesion"
                ],
                "probability_skin_without_target": prediction[
                    "probability_skin_without_target"
                ],
                "probability_no_skin": prediction["probability_no_skin"],
                "predicted_ready": prediction["predicted_ready"],
                "error_types": prediction["error_types"],
                "review_decision": "",
                "review_notes": "",
            }
        )
    return queue


def _draw_card(row: dict[str, str], width: int, image_height: int) -> Image.Image:
    font = ImageFont.load_default()
    text_height = 82
    card = Image.new("RGB", (width, image_height + text_height), "white")
    with Image.open(row["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((width, image_height))
        x = (width - image.width) // 2
        y = (image_height - image.height) // 2
        card.paste(image, (x, y))
    draw = ImageDraw.Draw(card)
    p_target = float(row["probability_target_lesion"])
    lines = (
        f"{row['review_id']} | {row['source']}",
        f"true: {row['true_label']}",
        f"pred: {row['predicted_label']} | P(target)={p_target:.3f}",
        row["error_types"],
    )
    draw.multiline_text((5, image_height + 4), "\n".join(lines), fill="black", font=font)
    return card


def render_contact_sheet(
    rows: list[dict[str, str]],
    destination: Path,
    columns: int = 4,
    card_width: int = 240,
    image_height: int = 210,
) -> None:
    if not rows:
        return
    cards = [_draw_card(row, card_width, image_height) for row in rows]
    card_height = cards[0].height
    sheet = Image.new(
        "RGB",
        (columns * card_width, math.ceil(len(cards) / columns) * card_height),
        "#dddddd",
    )
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % columns) * card_width, (index // columns) * card_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def create_error_review(
    predictions_path: Path,
    dataset_root: Path,
    destination: Path,
) -> dict[str, object]:
    queue = build_review_queue(
        read_csv(predictions_path),
        read_csv(dataset_root / "manifest.csv"),
        dataset_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    queue_path = destination / "error_review_queue.csv"
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(queue)

    categories = ("argmax_error", "ready_false_positive", "ready_false_negative")
    sheets: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for split in ("validation", "test"):
        for category in categories:
            selected = [
                row
                for row in queue
                if row["split"] == split and category in row["error_types"].split("|")
            ]
            counts[f"{split}:{category}"] = len(selected)
            if selected:
                path = destination / f"{split}_{category}.jpg"
                render_contact_sheet(selected, path)
                sheets[f"{split}:{category}"] = path.name

    transitions = Counter(
        f"{row['split']}:{row['true_label']}->{row['predicted_label']}" for row in queue
        if "argmax_error" in row["error_types"].split("|")
    )
    report = {
        "schema_version": 1,
        "predictions": str(predictions_path),
        "review_queue": queue_path.name,
        "review_items": len(queue),
        "counts": dict(sorted(counts.items())),
        "argmax_transitions": dict(sorted(transitions.items())),
        "contact_sheets": sheets,
    }
    report_path = destination / "error_review_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join capture predictions with private images and build error sheets."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_error_review(args.predictions, args.dataset_root, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
