"""Validate a built capture-validity TFRecord release end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_capture_tfrecords import CLASS_NAMES, CLASS_TO_INDEX, SPLITS, sha256_file


REQUIRED_FEATURES = {
    "image/encoded",
    "image/format",
    "image/height",
    "image/width",
    "label/id",
    "label/name",
    "group_id",
    "source",
    "sha256",
    "sample_weight",
}


def bytes_value(feature) -> bytes:
    values = feature.bytes_list.value
    if len(values) != 1:
        raise ValueError("Expected exactly one bytes value.")
    return values[0]


def int_value(feature) -> int:
    values = feature.int64_list.value
    if len(values) != 1:
        raise ValueError("Expected exactly one int64 value.")
    return int(values[0])


def float_value(feature) -> float:
    values = feature.float_list.value
    if len(values) != 1:
        raise ValueError("Expected exactly one float value.")
    return float(values[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata_path = root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required for TFRecord validation.") from error

    compression = "GZIP" if metadata.get("compression") == "GZIP" else ""
    counts = {split: Counter() for split in SPLITS}
    group_splits: dict[str, set[str]] = defaultdict(set)
    hashes = set()
    records = 0
    weighted_records = 0

    for shard in metadata.get("shards", []):
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Shard SHA-256 mismatch: {path.name}")
        shard_records = 0
        dataset = tf.data.TFRecordDataset(str(path), compression_type=compression)
        for serialized in dataset:
            example = tf.train.Example.FromString(bytes(serialized.numpy()))
            features = example.features.feature
            missing = REQUIRED_FEATURES - set(features)
            if missing:
                raise ValueError(f"{path.name}: missing features {sorted(missing)}")
            encoded = bytes_value(features["image/encoded"])
            digest = bytes_value(features["sha256"]).decode()
            if hashlib.sha256(encoded).hexdigest() != digest:
                raise ValueError(f"{path.name}: encoded image SHA-256 mismatch")
            if digest in hashes:
                raise ValueError(f"Duplicate image hash across shards: {digest}")
            hashes.add(digest)
            label = bytes_value(features["label/name"]).decode()
            label_id = int_value(features["label/id"])
            if label not in CLASS_TO_INDEX or CLASS_TO_INDEX[label] != label_id:
                raise ValueError(f"{path.name}: label contract mismatch")
            if int_value(features["image/height"]) <= 0 or int_value(
                features["image/width"]
            ) <= 0:
                raise ValueError(f"{path.name}: invalid image dimensions")
            group = bytes_value(features["group_id"]).decode()
            if not group or not bytes_value(features["source"]):
                raise ValueError(f"{path.name}: empty provenance feature")
            split = shard["split"]
            sample_weight = float_value(features["sample_weight"])
            if sample_weight <= 0:
                raise ValueError(f"{path.name}: sample_weight must be positive")
            if split in {"validation", "test"} and sample_weight != 1.0:
                raise ValueError(f"{path.name}: held-out sample_weight must be 1.0")
            weighted_records += sample_weight != 1.0
            counts[split][label] += 1
            group_splits[group].add(split)
            shard_records += 1
            records += 1
        if shard_records != shard["examples"]:
            raise ValueError(f"{path.name}: record count mismatch")

    expected_counts = {
        split: Counter(metadata["split_counts"][split]) for split in SPLITS
    }
    if counts != expected_counts:
        raise ValueError(f"Split counts mismatch: {counts} != {expected_counts}")
    if records != metadata["examples"]:
        raise ValueError("Total TFRecord count does not match metadata.")
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"{len(leaking)} groups cross split boundaries.")
    if set(metadata["classes"]) != set(CLASS_NAMES):
        raise ValueError("Metadata classes do not match the record contract.")
    if metadata.get("schema_version") != 2:
        raise ValueError("Unsupported TFRecord schema version.")
    weight_metadata = metadata.get("sample_weight", {})
    if weight_metadata.get("weighted_examples") != weighted_records:
        raise ValueError("Weighted-example count does not match metadata.")
    if weight_metadata.get("held_out_all_standard") is not True:
        raise ValueError("Metadata does not guarantee standard held-out weights.")

    print(
        json.dumps(
            {
                "status": "OK",
                "records": records,
                "unique_hashes": len(hashes),
                "groups": len(group_splits),
                "weighted_records": weighted_records,
                "split_counts": {
                    split: dict(sorted(counts[split].items())) for split in SPLITS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
