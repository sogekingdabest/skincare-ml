"""Run structural checks against the capture-validity notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from build_capture_validity_notebook import CELLS, OUTPUT


def main() -> None:
    notebook = json.loads(OUTPUT.read_text(encoding="utf-8"))
    expected_cells = CELLS
    if notebook.get("cells") != expected_cells:
        raise RuntimeError(
            "The notebook differs from scripts/build_capture_validity_notebook.py."
        )

    code_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for source in code_sources:
        ast.parse(source)

    all_code = "\n".join(code_sources)
    required_fragments = (
        'CLASS_NAMES = ["target_lesion", "skin_without_target", "no_skin"]',
        'METADATA_PATH = DATASET_ROOT / "dataset_metadata.json"',
        'input_root.rglob("dataset_metadata.json")',
        "discover_dataset_root",
        "discover_backbone_weights",
        'weights=str(BACKBONE_WEIGHTS_PATH)',
        '"pinned_private_kaggle_dataset"',
        "tf.data.TFRecordDataset",
        "num_parallel_reads=AUTOTUNE if training else 1",
        '"group_id": tf.io.FixedLenFeature([], tf.string)',
        '"sha256": tf.io.FixedLenFeature([], tf.string)',
        '"sample_weight": tf.io.FixedLenFeature([], tf.float32)',
        'return image, label, example["sample_weight"]',
        "sample_images, sample_labels, sample_weights",
        'weight_contract.get("held_out_all_standard") is not True',
        "weighted_train_dataset = train_dataset.map",
        "sample_weights * tf.gather(class_weight_tensor, labels)",
        "GENERALIZATION_ACCURACY_GAP_WARNING = 0.10",
        "VALIDATION_LOSS_REBOUND_WARNING = 0.05",
        "UNDERFIT_ACCURACY_WARNING = 0.80",
        "history_frame",
        'diagnostic_status = "possible_overfitting"',
        'diagnostic_status = "possible_underfitting"',
        '"heuristic_only": True',
        "training_history.csv",
        "training_diagnostics.json",
        "training_curves.png",
        "MobileNetV3Small",
        "include_preprocessing=False",
        'name="mobilenet_input_rescaling"',
        "TARGET_FALSE_READY_RATE = 0.01",
        "THRESHOLD_RESOLUTION = 0.001",
        "MIN_RELEASE_VALIDATION_NON_TARGET = 400",
        '"exploratory_zero_observed_false_ready"',
        'if candidate["false_positives"] == 0',
        'selected["test_used_for_selection"] = False',
        'metrics["release_gate"]',
        '"false_ready_rate_wilson_upper_95"',
        '"false_ready_rate_no_skin"',
        "select_ready_threshold",
        "build_prediction_frame",
        "validation_predictions.csv",
        "test_predictions.csv",
        'errors.append("ready_false_positive")',
        'errors.append("ready_false_negative")',
        "tf.lite.TFLiteConverter.from_keras_model",
        "converter.target_spec.supported_types = [tf.float16]",
        "ready_decision_agreement",
        "capture_validity_metrics.json",
    )
    for fragment in required_fragments:
        if fragment not in all_code:
            raise RuntimeError(f"Notebook is missing contract fragment: {fragment}")

    forbidden_fragments = (
        "validation_dataset = test_dataset",
        "TARGET_FALSE_READY_RATE = 1.0",
        '"exploratory_observed_rate"',
        "warnings.filterwarnings('ignore')",
        'CLASS_NAMES = ["lesion", "healthy_skin", "no_skin", "poor_quality"]',
        "tf.io.read_file(path)",
        "class_weight=class_weights",
    )
    for fragment in forbidden_fragments:
        if fragment in all_code:
            raise RuntimeError(f"Notebook contains forbidden fragment: {fragment}")

    for cell in notebook["cells"]:
        if not cell.get("id"):
            raise RuntimeError("Every notebook cell must have a stable id.")
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            raise RuntimeError("The committed notebook contains outputs.")
        if cell.get("execution_count") is not None:
            raise RuntimeError("The committed notebook contains execution counts.")

    cell_ids = [cell["id"] for cell in notebook["cells"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("Notebook cell ids must be unique.")

    print("Capture-validity notebook validation: OK")


if __name__ == "__main__":
    main()
