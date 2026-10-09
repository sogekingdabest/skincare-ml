from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


CLASS_NAMES = ("target_lesion", "skin_without_target", "no_skin")
READY_CLASS_INDEX = 0
PREDICTION_FIELDS = (
    "split",
    "record_index",
    "sha256",
    "group_id",
    "source",
    "true_label_id",
    "true_label",
    "predicted_label_id",
    "predicted_label",
    "probability_target_lesion",
    "probability_skin_without_target",
    "probability_no_skin",
    "ready_threshold",
    "predicted_ready",
    "error_types",
)


def classify_error_types(
    true_label_id: int,
    predicted_label_id: int,
    target_probability: float,
    ready_threshold: float,
) -> str:
    errors: list[str] = []
    if predicted_label_id != true_label_id:
        errors.append("argmax_error")
    predicted_ready = target_probability >= ready_threshold
    truly_ready = true_label_id == READY_CLASS_INDEX
    if predicted_ready and not truly_ready:
        errors.append("ready_false_positive")
    elif not predicted_ready and truly_ready:
        errors.append("ready_false_negative")
    return "|".join(errors)


def prediction_row(
    *,
    split: str,
    record_index: int,
    metadata: dict[str, Any],
    probabilities: Iterable[float],
    ready_threshold: float,
) -> dict[str, object]:
    values = [float(value) for value in probabilities]
    if len(values) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class probabilities, got {len(values)}."
        )
    true_label_id = int(metadata["label_id"])
    predicted_label_id = max(range(len(values)), key=values.__getitem__)
    target_probability = values[READY_CLASS_INDEX]
    return {
        "split": split,
        "record_index": record_index,
        "sha256": str(metadata["sha256"]),
        "group_id": str(metadata["group_id"]),
        "source": str(metadata["source"]),
        "true_label_id": true_label_id,
        "true_label": CLASS_NAMES[true_label_id],
        "predicted_label_id": predicted_label_id,
        "predicted_label": CLASS_NAMES[predicted_label_id],
        "probability_target_lesion": f"{values[0]:.9f}",
        "probability_skin_without_target": f"{values[1]:.9f}",
        "probability_no_skin": f"{values[2]:.9f}",
        "ready_threshold": f"{ready_threshold:.9f}",
        "predicted_ready": str(target_probability >= ready_threshold).lower(),
        "error_types": classify_error_types(
            true_label_id,
            predicted_label_id,
            target_probability,
            ready_threshold,
        ),
    }


def _decode(value: Any) -> str:
    raw = value.numpy()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _feature_spec(tf: Any) -> dict[str, Any]:
    return {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "label/id": tf.io.FixedLenFeature([], tf.int64),
        "group_id": tf.io.FixedLenFeature([], tf.string),
        "source": tf.io.FixedLenFeature([], tf.string),
        "sha256": tf.io.FixedLenFeature([], tf.string),
    }


def _load_release_metadata(tfrecord_root: Path) -> dict[str, Any]:
    path = tfrecord_root / "dataset_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if tuple(metadata.get("classes", ())) != CLASS_NAMES:
        raise ValueError("TFRecord class order does not match the model contract.")
    return metadata


def _split_files(tfrecord_root: Path, split: str) -> list[str]:
    files = sorted(tfrecord_root.glob(f"{split}-*.tfrec*"))
    if not files:
        raise FileNotFoundError(f"No TFRecord shards found for split {split!r}.")
    return [str(path) for path in files]


def export_predictions(
    model_path: Path,
    tfrecord_root: Path,
    metrics_path: Path,
    destination: Path,
    splits: tuple[str, ...] = ("validation", "test"),
    batch_size: int = 32,
) -> dict[str, object]:
    import numpy as np
    import tensorflow as tf

    release_metadata = _load_release_metadata(tfrecord_root)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ready_threshold = float(metrics["test_ready_gate"]["threshold"])
    compression = "GZIP" if release_metadata.get("compression") == "GZIP" else ""
    model = tf.keras.models.load_model(model_path)
    if tuple(model.output_shape) != (None, len(CLASS_NAMES)):
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")

    spec = _feature_spec(tf)
    rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for split in splits:
        records = tf.data.TFRecordDataset(
            _split_files(tfrecord_root, split),
            compression_type=compression,
            num_parallel_reads=1,
        )
        examples: list[dict[str, Any]] = []
        images: list[Any] = []
        for serialized in records:
            example = tf.io.parse_single_example(serialized, spec)
            image = tf.io.decode_image(
                example["image/encoded"], channels=3, expand_animations=False
            )
            image.set_shape([None, None, 3])
            image = tf.image.convert_image_dtype(image, tf.float32)
            images.append(tf.image.resize_with_pad(image, 192, 192))
            examples.append(
                {
                    "label_id": int(example["label/id"].numpy()),
                    "group_id": _decode(example["group_id"]),
                    "source": _decode(example["source"]),
                    "sha256": _decode(example["sha256"]),
                }
            )
        if not images:
            raise RuntimeError(f"Split {split!r} contains no records.")
        dataset = tf.data.Dataset.from_tensor_slices(tf.stack(images)).batch(batch_size)
        probabilities = np.asarray(model.predict(dataset, verbose=1))
        if len(probabilities) != len(examples):
            raise RuntimeError("Prediction and TFRecord metadata order mismatch.")
        for index, (example, probability) in enumerate(
            zip(examples, probabilities, strict=True)
        ):
            rows.append(
                prediction_row(
                    split=split,
                    record_index=index,
                    metadata=example,
                    probabilities=probability,
                    ready_threshold=ready_threshold,
                )
            )
        split_counts[split] = len(examples)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(destination),
        "rows": len(rows),
        "splits": split_counts,
        "ready_threshold": ready_threshold,
        "errors": sum(bool(row["error_types"]) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic per-sample predictions from capture TFRecords."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tfrecord-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"],
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_predictions(
        args.model,
        args.tfrecord_root,
        args.metrics,
        args.output,
        tuple(args.splits),
        args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
