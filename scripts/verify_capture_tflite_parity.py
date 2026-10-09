"""Export sample-level Keras/TFLite parity for capture-validity test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CLASS_NAMES = ("target_lesion", "skin_without_target", "no_skin")
PARITY_FIELDS = (
    "record_index", "sha256", "group_id", "source", "true_label",
    "keras_predicted_label", "tflite_predicted_label",
    "keras_probability_target_lesion", "tflite_probability_target_lesion",
    "target_probability_delta", "max_absolute_probability_error",
    "keras_ready", "tflite_ready", "argmax_agreement", "ready_agreement",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_parity_rows(
    keras_rows: list[dict[str, str]],
    tflite_probabilities: Iterable[Iterable[float]],
    threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    probabilities = [list(map(float, values)) for values in tflite_probabilities]
    if len(keras_rows) != len(probabilities):
        raise ValueError("Keras and TFLite predictions have different sample counts.")
    if not keras_rows:
        raise ValueError("Parity input is empty.")

    rows: list[dict[str, object]] = []
    for keras, lite in zip(keras_rows, probabilities, strict=True):
        if len(lite) != len(CLASS_NAMES):
            raise ValueError("Unexpected TFLite output class count.")
        keras_probabilities = [
            float(keras[f"probability_{class_name}"]) for class_name in CLASS_NAMES
        ]
        keras_argmax = max(range(len(CLASS_NAMES)), key=keras_probabilities.__getitem__)
        lite_argmax = max(range(len(CLASS_NAMES)), key=lite.__getitem__)
        errors = [abs(a - b) for a, b in zip(keras_probabilities, lite, strict=True)]
        keras_ready = keras_probabilities[0] >= threshold
        lite_ready = lite[0] >= threshold
        rows.append({
            "record_index": keras["record_index"],
            "sha256": keras["sha256"],
            "group_id": keras["group_id"],
            "source": keras["source"],
            "true_label": keras["true_label"],
            "keras_predicted_label": CLASS_NAMES[keras_argmax],
            "tflite_predicted_label": CLASS_NAMES[lite_argmax],
            "keras_probability_target_lesion": format(keras_probabilities[0], ".9g"),
            "tflite_probability_target_lesion": format(lite[0], ".9g"),
            "target_probability_delta": format(lite[0] - keras_probabilities[0], ".9g"),
            "max_absolute_probability_error": format(max(errors), ".9g"),
            "keras_ready": str(keras_ready).lower(),
            "tflite_ready": str(lite_ready).lower(),
            "argmax_agreement": str(keras_argmax == lite_argmax).lower(),
            "ready_agreement": str(keras_ready == lite_ready).lower(),
        })

    argmax_mismatches = [row for row in rows if row["argmax_agreement"] == "false"]
    ready_mismatches = [row for row in rows if row["ready_agreement"] == "false"]
    report = {
        "samples": len(rows),
        "argmax_mismatches": len(argmax_mismatches),
        "ready_mismatches": len(ready_mismatches),
        "argmax_agreement": 1.0 - len(argmax_mismatches) / len(rows),
        "ready_decision_agreement": 1.0 - len(ready_mismatches) / len(rows),
        "argmax_mismatch_sha256": [row["sha256"] for row in argmax_mismatches],
        "ready_mismatch_sha256": [row["sha256"] for row in ready_mismatches],
        "max_absolute_probability_error": max(
            float(row["max_absolute_probability_error"]) for row in rows
        ),
    }
    return rows, report


def _feature_spec(tf: Any) -> dict[str, Any]:
    return {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "sha256": tf.io.FixedLenFeature([], tf.string),
    }


def run_tflite(tflite_path: Path, tfrecord_root: Path) -> tuple[list[str], list[Any]]:
    import numpy as np
    import tensorflow as tf

    metadata = json.loads(
        (tfrecord_root / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    compression = "GZIP" if metadata.get("compression") == "GZIP" else ""
    files = sorted(tfrecord_root.glob("test-*.tfrec*"))
    if not files:
        raise FileNotFoundError("No test TFRecord shards found.")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != (1, 192, 192, 3):
        raise ValueError(f"Unexpected TFLite input shape: {input_detail['shape']}")

    hashes: list[str] = []
    probabilities: list[Any] = []
    records = tf.data.TFRecordDataset(
        [str(path) for path in files], compression_type=compression, num_parallel_reads=1
    )
    spec = _feature_spec(tf)
    for serialized in records:
        example = tf.io.parse_single_example(serialized, spec)
        image = tf.io.decode_image(
            example["image/encoded"], channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, 192, 192).numpy()
        interpreter.set_tensor(
            input_detail["index"], image[np.newaxis].astype(input_detail["dtype"])
        )
        interpreter.invoke()
        probabilities.append(interpreter.get_tensor(output_detail["index"])[0].copy())
        hashes.append(example["sha256"].numpy().decode("utf-8"))
    return hashes, probabilities


def verify_parity(
    tflite_path: Path,
    tfrecord_root: Path,
    keras_predictions_path: Path,
    metrics_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(metrics["test_ready_gate"]["threshold"])
    keras_rows = read_csv(keras_predictions_path)
    hashes, probabilities = run_tflite(tflite_path, tfrecord_root)
    if [row["sha256"] for row in keras_rows] != hashes:
        raise ValueError("TFRecord order/identity differs from Keras prediction export.")
    rows, report = build_parity_rows(keras_rows, probabilities, threshold)

    artifact_hash = sha256_file(tflite_path)
    report.update({
        "schema_version": 1,
        "threshold": threshold,
        "tflite_sha256": artifact_hash,
        "artifact_hash_matches_metrics": artifact_hash == metrics["artifacts"]["tflite_sha256"],
        "aggregate_matches_kaggle": {
            "argmax_agreement": abs(
                report["argmax_agreement"] - metrics["parity"]["argmax_agreement"]
            ) < 1e-12,
            "ready_decision_agreement": abs(
                report["ready_decision_agreement"]
                - metrics["parity"]["ready_decision_agreement"]
            ) < 1e-12,
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample_parity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PARITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--tfrecord-root", type=Path, required=True)
    parser.add_argument("--keras-predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_parity(
        args.tflite, args.tfrecord_root, args.keras_predictions,
        args.metrics, args.output_dir
    ), indent=2))


if __name__ == "__main__":
    main()
