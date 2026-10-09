"""Static checks for the independent Kaggle TFLite export kernel."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle" / "tflite_export"
SCRIPT_PATH = KERNEL_DIR / "export_tflite.py"
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"

source = SCRIPT_PATH.read_text(encoding="utf-8")
ast.parse(source, filename=str(SCRIPT_PATH))
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

expected_metadata = {
    "id": "daniolaetafaria/skincare-ai-tflite-export-benchmark",
    "code_file": SCRIPT_PATH.name,
    "kernel_type": "script",
    "enable_gpu": False,
    "enable_internet": False,
}
for field, expected in expected_metadata.items():
    actual = metadata.get(field)
    if actual != expected:
        raise AssertionError(f"{field}: expected {expected!r}, found {actual!r}")

required_sources = {
    "cdeotte/melanoma-256x256": metadata.get("dataset_sources", []),
    "daniolaetafaria/skincare-ai-multimodal-melanoma-classification": (
        metadata.get("kernel_sources", [])
    ),
}
for source_name, configured_sources in required_sources.items():
    if source_name not in configured_sources:
        raise AssertionError(f"Missing Kaggle source: {source_name}")

required_fragments = [
    'variant == "float16"',
    'variant == "int8"',
    "TFLITE_BUILTINS_INT8",
    "representative_dataset",
    "inference_input_type = tf.int8",
    "keras_tflite_parity",
    "decision_agreement",
    "sha256_file",
    "tensor_contract",
    "tflite_export_comparison.json",
]
for fragment in required_fragments:
    if fragment not in source:
        raise AssertionError(f"Missing export-kernel contract: {fragment}")

for forbidden in ("model.fit(", ".fit_generator(", "model.train_on_batch("):
    if forbidden in source:
        raise AssertionError(f"Export kernel must not retrain: {forbidden}")

print("TFLite export kernel validation passed.")
