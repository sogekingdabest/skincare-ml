"""Export and compare mobile TensorFlow Lite variants without retraining."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


SEED = 42
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
REPRESENTATIVE_SAMPLES = 512
PARITY_LIMITS = {
    "mean_absolute_error_max": 0.005,
    "p99_absolute_error_max": 0.02,
    "decision_agreement_min": 0.99,
}
OUTPUT_REPORT = Path("tflite_export_comparison.json")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as error:  # pragma: no cover - depends on the Kaggle image
    print(f"Deterministic operations could not be enabled: {error}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_input_file(filename: str) -> Path:
    matches = sorted(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under /kaggle/input")
    if len(matches) > 1:
        print(f"Multiple matches for {filename}; using {matches[0]}")
    return matches[0]


def resolve_shards(names: list[str]) -> list[str]:
    return [str(find_input_file(name)) for name in names]


def parse_record(serialized: tf.Tensor):
    schema = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "age_approx": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "sex": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "anatom_site_general_challenge": tf.io.FixedLenFeature(
            [], tf.int64, default_value=-1
        ),
    }
    example = tf.io.parse_single_example(serialized, schema)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    metadata = tf.stack(
        [
            tf.cast(example["age_approx"], tf.float32) / 100.0,
            tf.cast(example["sex"], tf.float32),
            tf.cast(example["anatom_site_general_challenge"], tf.float32)
            / 7.0,
        ]
    )
    target = tf.reshape(tf.cast(example["target"], tf.float32), [1])
    return (image, metadata), target


def make_dataset(
    filenames: list[str],
    *,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    dataset = dataset.with_options(options)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(
            4096,
            seed=SEED,
            reshuffle_each_iteration=False,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def representative_dataset(filenames: list[str]):
    dataset = make_dataset(filenames, batch_size=1, shuffle=True)
    for (images, metadata), _ in dataset.take(REPRESENTATIVE_SAMPLES):
        yield [images.numpy(), metadata.numpy()]


def convert_variant(
    model: tf.keras.Model,
    output_path: Path,
    variant: str,
    representative_filenames: list[str],
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if variant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif variant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(
            representative_filenames
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif variant != "float32":
        raise ValueError(f"Unknown export variant: {variant}")

    output_path.write_bytes(converter.convert())


def quantize_input(value: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return value.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError(f"Invalid quantization scale for {detail['name']}: {scale}")
    limits = np.iinfo(dtype)
    quantized = np.rint(value / scale + zero_point)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def dequantize_output(value: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError(f"Invalid output quantization scale: {scale}")
    return (value.astype(np.float32) - zero_point) * scale


def tensor_contract(detail: dict) -> dict:
    scale, zero_point = detail["quantization"]
    return {
        "name": detail["name"],
        "shape": [int(value) for value in detail["shape_signature"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization_scale": float(scale),
        "quantization_zero_point": int(zero_point),
    }


def predict_tflite(model_path: Path, dataset: tf.data.Dataset):
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    input_details = interpreter.get_input_details()
    if len(input_details) != 2:
        raise ValueError(f"Expected two inputs, found {len(input_details)}")

    image_detail = next(
        detail for detail in input_details if len(detail["shape_signature"]) == 4
    )
    metadata_detail = next(
        detail for detail in input_details if len(detail["shape_signature"]) == 2
    )
    predictions = []
    labels = []
    output_detail = interpreter.get_output_details()[0]

    for (images, metadata), targets in dataset:
        image_values = images.numpy()
        metadata_values = metadata.numpy()
        batch = image_values.shape[0]
        interpreter.resize_tensor_input(
            image_detail["index"],
            [batch, *IMAGE_SIZE, 3],
            strict=True,
        )
        interpreter.resize_tensor_input(
            metadata_detail["index"],
            [batch, 3],
            strict=True,
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_detail = interpreter.get_output_details()[0]
        image_detail = next(
            detail
            for detail in input_details
            if len(detail["shape_signature"]) == 4
        )
        metadata_detail = next(
            detail
            for detail in input_details
            if len(detail["shape_signature"]) == 2
        )
        interpreter.set_tensor(
            image_detail["index"],
            quantize_input(image_values, image_detail),
        )
        interpreter.set_tensor(
            metadata_detail["index"],
            quantize_input(metadata_values, metadata_detail),
        )
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        predictions.append(dequantize_output(output, output_detail).reshape(-1))
        labels.append(targets.numpy().reshape(-1))

    contract = {
        "inputs": [tensor_contract(detail) for detail in input_details],
        "outputs": [tensor_contract(output_detail)],
    }
    return np.concatenate(labels), np.concatenate(predictions), contract


def predict_keras(model: tf.keras.Model, dataset: tf.data.Dataset):
    predictions = []
    labels = []
    for inputs, targets in dataset:
        predictions.append(model(inputs, training=False).numpy().reshape(-1))
        labels.append(targets.numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(predictions)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    decisions = (probabilities >= threshold).astype(np.int32)
    integer_labels = labels.astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(
        integer_labels,
        decisions,
        labels=[0, 1],
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "roc_auc": float(roc_auc_score(integer_labels, probabilities)),
        "average_precision": float(
            average_precision_score(integer_labels, probabilities)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(
            precision_score(integer_labels, decisions, zero_division=0)
        ),
        "negative_predictive_value": float(npv),
        "f1": float(f1_score(integer_labels, decisions, zero_division=0)),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(integer_labels, decisions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(integer_labels, decisions)
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def parity_metrics(
    keras_probabilities: np.ndarray,
    tflite_probabilities: np.ndarray,
    threshold: float,
) -> dict:
    errors = np.abs(keras_probabilities - tflite_probabilities)
    agreement = np.mean(
        (keras_probabilities >= threshold)
        == (tflite_probabilities >= threshold)
    )
    result = {
        "mean_absolute_error": float(np.mean(errors)),
        "maximum_absolute_error": float(np.max(errors)),
        "p99_absolute_error": float(np.percentile(errors, 99)),
        "decision_agreement": float(agreement),
        "limits": PARITY_LIMITS,
    }
    result["passed"] = bool(
        result["mean_absolute_error"]
        <= PARITY_LIMITS["mean_absolute_error_max"]
        and result["p99_absolute_error"]
        <= PARITY_LIMITS["p99_absolute_error_max"]
        and result["decision_agreement"]
        >= PARITY_LIMITS["decision_agreement_min"]
    )
    return result


def evaluate_artifact(
    path: Path,
    test_dataset: tf.data.Dataset,
    keras_labels: np.ndarray,
    keras_probabilities: np.ndarray,
    threshold: float,
) -> dict:
    labels, probabilities, contract = predict_tflite(path, test_dataset)
    if not np.array_equal(labels, keras_labels):
        raise RuntimeError("Keras and TFLite evaluation labels differ")
    return {
        "status": "complete",
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_contract": contract,
        "test_metrics": classification_metrics(
            labels,
            probabilities,
            threshold,
        ),
        "keras_tflite_parity": parity_metrics(
            keras_probabilities,
            probabilities,
            threshold,
        ),
    }


model_path = find_input_file("skincare_model_multimodal.keras")
source_metrics_path = find_input_file("metrics_256.json")
source_dynamic_path = find_input_file("skincare_multimodal_256.tflite")
source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
threshold = float(source_metrics["threshold_selection"]["threshold"])
test_shards = resolve_shards(
    source_metrics["run"]["split_strategy"]["test_shards_2020"]
)
representative_shards = resolve_shards(
    source_metrics["run"]["split_strategy"]["training_shards_2020"]
)

print(f"Loading trained model: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)
test_dataset = make_dataset(test_shards, batch_size=BATCH_SIZE)
keras_labels, keras_probabilities = predict_keras(model, test_dataset)

report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
    },
    "source": {
        "keras_filename": model_path.name,
        "keras_sha256": sha256_file(model_path),
        "metrics_filename": source_metrics_path.name,
        "metrics_sha256": sha256_file(source_metrics_path),
        "decision_threshold": threshold,
        "test_examples": int(keras_labels.size),
        "representative_samples": REPRESENTATIVE_SAMPLES,
        "representative_split": "training_shards_2020",
    },
    "keras_test_metrics": classification_metrics(
        keras_labels,
        keras_probabilities,
        threshold,
    ),
    "variants": {},
}

report["variants"]["source_dynamic"] = evaluate_artifact(
    source_dynamic_path,
    test_dataset,
    keras_labels,
    keras_probabilities,
    threshold,
)

variant_paths = {
    "float32": Path("skincare_multimodal_256_float32.tflite"),
    "float16": Path("skincare_multimodal_256_float16.tflite"),
    "int8": Path("skincare_multimodal_256_int8.tflite"),
}
for variant, output_path in variant_paths.items():
    print(f"Converting {variant} -> {output_path}")
    try:
        convert_variant(model, output_path, variant, representative_shards)
        report["variants"][variant] = evaluate_artifact(
            output_path,
            test_dataset,
            keras_labels,
            keras_probabilities,
            threshold,
        )
    except Exception as error:
        print(f"{variant} failed: {type(error).__name__}: {error}")
        report["variants"][variant] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }

eligible = [
    (name, result)
    for name, result in report["variants"].items()
    if name != "source_dynamic"
    and result.get("status") == "complete"
    and result["keras_tflite_parity"]["passed"]
]
eligible.sort(key=lambda item: item[1]["size_bytes"])
report["recommendation"] = (
    {
        "variant": eligible[0][0],
        "filename": eligible[0][1]["filename"],
        "reason": "Smallest newly exported variant that passed the parity gate.",
    }
    if eligible
    else None
)

OUTPUT_REPORT.write_text(
    json.dumps(report, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
