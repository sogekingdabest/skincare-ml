"""Build grouped, sharded TFRecords for the capture semantic gate.

The input is a reviewed ``manifest.csv`` plus its referenced square crops. The
script deduplicates identical bytes, resolves a deterministic group-aware split
when necessary and writes a training dataset that no longer depends on the
layout of the upstream public datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_NAMES = ("target_lesion", "skin_without_target", "no_skin")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {"image_path", "label", "group_id", "source", "sha256"}
QUALITY_COLUMNS = (
    "bad_framing",
    "bad_light",
    "blur",
    "low_resolution",
    "too_far",
)
TRUE_VALUES = {"1", "true", "yes"}
SCHEMA_VERSION = 2


def tensorflow_compression_type(compression: str) -> str:
    if compression == "NONE":
        return ""
    if compression == "GZIP":
        return "GZIP"
    raise ValueError(f"Unsupported TFRecord compression: {compression!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_grouped_splits(rows: list[dict[str, str]], seed: int) -> None:
    counts_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts_by_group[row["group_id"]][row["label"]] += 1
    for label in CLASS_NAMES:
        label_groups = {
            row["group_id"] for row in rows if row["label"] == label
        }
        if len(label_groups) < 7:
            raise ValueError(
                f"Automatic splitting needs at least seven independent groups "
                f"for {label!r}; found {len(label_groups)}."
            )

    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    total_by_label = Counter(row["label"] for row in rows)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    group_ids = sorted(counts_by_group)

    # Search deterministic hash partitions. This keeps mixed-label groups
    # intact without adding a heavy splitting dependency to the CPU builder.
    for attempt in range(4096):
        assignment: dict[str, str] = {}
        actual = {split: Counter() for split in SPLITS}
        for group_id in group_ids:
            digest = hashlib.sha256(
                f"{seed}:{attempt}:{group_id}".encode()
            ).digest()
            fraction = int.from_bytes(digest[:8], "big") / 2**64
            split = (
                "train"
                if fraction < ratios["train"]
                else "validation"
                if fraction < ratios["train"] + ratios["validation"]
                else "test"
            )
            assignment[group_id] = split
            actual[split].update(counts_by_group[group_id])

        if any(actual[split][label] == 0 for split in SPLITS for label in CLASS_NAMES):
            continue
        score = 0.0
        for split in SPLITS:
            for label in CLASS_NAMES:
                target = total_by_label[label] * ratios[split]
                score += ((actual[split][label] - target) / max(target, 1.0)) ** 2
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise ValueError(
            "Could not create a grouped split containing every class. "
            "Provide a reviewed fixed split in manifest.csv."
        )
    for row in rows:
        row["split"] = best_assignment[row["group_id"]]


def load_and_resolve_manifest(
    dataset_root: Path,
    seed: int,
    verify_hashes: bool,
    manifest_path: Path | None = None,
) -> tuple[list[dict[str, str]], int]:
    manifest_path = (manifest_path or dataset_root / "manifest.csv").resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("Manifest contains no samples.")

    unique_by_hash: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        if row["label"] not in CLASS_TO_INDEX:
            raise ValueError(
                f"Line {line_number}: unsupported semantic label {row['label']!r}."
            )
        if not row["group_id"] or not row["source"]:
            raise ValueError(f"Line {line_number}: group_id and source are required.")
        if any(row.get(column, "").lower() in TRUE_VALUES for column in QUALITY_COLUMNS):
            raise ValueError(
                f"Line {line_number}: quality-failure samples belong to the "
                "deterministic gate evaluation, not the semantic TFRecords."
            )
        image_path = (dataset_root / row["image_path"]).resolve()
        if not image_path.is_relative_to(dataset_root) or not image_path.is_file():
            raise FileNotFoundError(f"Line {line_number}: invalid image {image_path}")
        digest = row["sha256"].lower()
        if verify_hashes:
            actual = sha256_file(image_path)
            if actual != digest:
                raise ValueError(f"Line {line_number}: SHA-256 mismatch for {image_path}")
        row["absolute_path"] = str(image_path)
        row["sha256"] = digest
        try:
            sample_weight = float(row.get("sample_weight", "1.0") or "1.0")
        except ValueError as error:
            raise ValueError(
                f"Line {line_number}: sample_weight must be numeric."
            ) from error
        if not math.isfinite(sample_weight) or sample_weight <= 0:
            raise ValueError(
                f"Line {line_number}: sample_weight must be finite and positive."
            )
        if row.get("split", "").lower() in {"validation", "test"} and sample_weight != 1.0:
            raise ValueError(
                f"Line {line_number}: held-out sample_weight must remain 1.0."
            )
        row["sample_weight"] = format(sample_weight, ".8g")
        previous = unique_by_hash.get(digest)
        if previous and previous["label"] != row["label"]:
            raise ValueError(
                f"Duplicate {digest} has conflicting labels "
                f"{previous['label']!r} and {row['label']!r}."
            )
        if previous is None:
            unique_by_hash[digest] = row

    deduplicated = len(rows) - len(unique_by_hash)
    resolved = list(unique_by_hash.values())
    split_values = [row.get("split", "").lower() for row in resolved]
    if any(split_values) and not all(split_values):
        raise ValueError("split must be populated for every row or no rows.")
    if all(split_values):
        unknown = set(split_values) - set(SPLITS)
        if unknown:
            raise ValueError(f"Unknown split values: {sorted(unknown)}")
        for row, split in zip(resolved, split_values):
            row["split"] = split
    else:
        assign_grouped_splits(resolved, seed)

    groups_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in resolved:
        groups_to_splits[row["group_id"]].add(row["split"])
    leaking = [group for group, splits in groups_to_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"{len(leaking)} groups cross split boundaries.")

    for split in SPLITS:
        present = {row["label"] for row in resolved if row["split"] == split}
        missing = set(CLASS_NAMES) - present
        if missing:
            raise ValueError(f"{split} is missing classes: {sorted(missing)}")
    return resolved, deduplicated


def bytes_feature(tf: Any, value: bytes) -> Any:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int_feature(tf: Any, value: int) -> Any:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(tf: Any, value: float) -> Any:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def serialize_example(tf: Any, row: dict[str, str]) -> bytes:
    image_path = Path(row["absolute_path"])
    encoded = image_path.read_bytes()
    with Image.open(image_path) as image:
        width, height = image.size
        image_format = (image.format or image_path.suffix.lstrip(".")).lower()
    features = {
        "image/encoded": bytes_feature(tf, encoded),
        "image/format": bytes_feature(tf, image_format.encode()),
        "image/height": int_feature(tf, height),
        "image/width": int_feature(tf, width),
        "label/id": int_feature(tf, CLASS_TO_INDEX[row["label"]]),
        "label/name": bytes_feature(tf, row["label"].encode()),
        "group_id": bytes_feature(tf, row["group_id"].encode()),
        "source": bytes_feature(tf, row["source"].encode()),
        "sha256": bytes_feature(tf, row["sha256"].encode()),
        "sample_weight": float_feature(tf, float(row["sample_weight"])),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def write_resolved_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    fields = [
        "image_path",
        "label",
        "group_id",
        "source",
        "sha256",
        "split",
        "sample_weight",
        "training_action",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression", choices=("NONE", "GZIP"), default="NONE")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Reviewed manifest to apply while resolving images from dataset_root.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to serialize TFRecords. Run this script in "
            "the Kaggle reference environment or install requirements.txt."
        ) from error

    rows, deduplicated = load_and_resolve_manifest(
        dataset_root,
        seed=args.seed,
        verify_hashes=not args.skip_hash_check,
        manifest_path=args.manifest,
    )
    manifest_path = (args.manifest or dataset_root / "manifest.csv").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    options = tf.io.TFRecordOptions(
        compression_type=tensorflow_compression_type(args.compression)
    )
    extension = ".tfrec.gz" if args.compression == "GZIP" else ".tfrec"
    shard_metadata = []
    split_counts: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        rng.shuffle(split_rows)
        label_counts = Counter(row["label"] for row in split_rows)
        split_counts[split] = dict(sorted(label_counts.items()))
        shard_count = math.ceil(len(split_rows) / args.shard_size)
        for shard_index in range(shard_count):
            shard_rows = split_rows[
                shard_index * args.shard_size : (shard_index + 1) * args.shard_size
            ]
            name = f"{split}-{shard_index:05d}-of-{shard_count:05d}{extension}"
            path = output_root / name
            with tf.io.TFRecordWriter(str(path), options=options) as writer:
                for row in shard_rows:
                    writer.write(serialize_example(tf, row))
            shard_metadata.append(
                {
                    "file": name,
                    "split": split,
                    "examples": len(shard_rows),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    write_resolved_manifest(rows, output_root / "resolved_manifest.csv")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task": "capture_semantic_gate",
        "publication_scope": (
            "rights_verified"
            if rows
            and all(
                row.get("release_eligible", "").lower() in TRUE_VALUES
                for row in rows
            )
            else "private_training_only"
        ),
        "release_eligible": bool(rows)
        and all(
            row.get("release_eligible", "").lower() in TRUE_VALUES
            for row in rows
        ),
        "classes": list(CLASS_NAMES),
        "ready_class": "target_lesion",
        "compression": args.compression,
        "shard_size": args.shard_size,
        "seed": args.seed,
        "examples": len(rows),
        "deduplicated_examples": deduplicated,
        "groups": len({row["group_id"] for row in rows}),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_name": manifest_path.name,
        "source_counts": dict(
            sorted(Counter(row["source"] for row in rows).items())
        ),
        "annotation_source_counts": dict(
            sorted(
                Counter(
                    row.get("annotation_source", "unknown") or "unknown"
                    for row in rows
                ).items()
            )
        ),
        "weak_supervision_examples": sum(
            row.get("weak_supervision", "").lower() in TRUE_VALUES
            for row in rows
        ),
        "release_eligible_examples": sum(
            row.get("release_eligible", "").lower() in TRUE_VALUES
            for row in rows
        ),
        "source_license_status_counts": dict(
            sorted(
                Counter(
                    row.get("source_license_status", "unknown") or "unknown"
                    for row in rows
                ).items()
            )
        ),
        "split_counts": split_counts,
        "sample_weight": {
            "feature": "sample_weight",
            "standard": 1.0,
            "minimum": min(float(row["sample_weight"]) for row in rows),
            "maximum": max(float(row["sample_weight"]) for row in rows),
            "weighted_examples": sum(
                float(row["sample_weight"]) != 1.0 for row in rows
            ),
            "by_value": dict(
                sorted(
                    Counter(row["sample_weight"] for row in rows).items(),
                    key=lambda item: float(item[0]),
                )
            ),
            "held_out_all_standard": all(
                float(row["sample_weight"]) == 1.0
                for row in rows
                if row["split"] in {"validation", "test"}
            ),
        },
        "features": {
            "image/encoded": "bytes",
            "image/format": "bytes",
            "image/height": "int64",
            "image/width": "int64",
            "label/id": "int64",
            "label/name": "bytes",
            "group_id": "bytes",
            "source": "bytes",
            "sha256": "bytes",
            "sample_weight": "float32",
        },
        "shards": shard_metadata,
    }
    (output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
