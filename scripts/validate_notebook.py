"""Run lightweight structural checks against the committed training notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from prepare_notebook import prepare_notebook


NOTEBOOK_PATH = Path("notebooks/skincare_training.ipynb")


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    rebuilt = prepare_notebook(
        json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    )
    if notebook != rebuilt:
        raise RuntimeError(
            "The notebook differs from its canonical generated representation."
        )

    code_sources = {
        index: "".join(cell["source"])
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    }
    for source in code_sources.values():
        ast.parse(source)

    required_fragments = {
        1: (
            "RUN_384_EXPERIMENT = False",
            "tf.config.experimental.enable_op_determinism()",
        ),
        3: (
            "tf.image.convert_image_dtype",
            "tf.clip_by_value(image, 0.0, 1.0)",
            "tf.reshape(",
        ),
        7: (
            "tf.keras.losses.BinaryFocalCrossentropy",
            "apply_class_balancing=True",
        ),
        5: (
            "def count_tfrecord_items",
            "NUM_TRAINING_IMAGES = count_tfrecord_items",
            "NUM_VALIDATION_IMAGES = count_tfrecord_items",
            "NUM_TEST_IMAGES = count_tfrecord_items",
        ),
        9: (
            "include_preprocessing=False",
            "efficientnet_input_rescaling",
            "scale=2.0",
            "offset=-1.0",
            "base_model.trainable = False",
            "base_model(efficientnet_input, training=False)",
        ),
        11: (
            "def validate_batch_contract",
            "validate_batch_contract(train_generator",
            "validate_batch_contract(val_generator",
        ),
        13: (
            "isinstance(layer, layers.BatchNormalization)",
            "learning_rate=1e-5",
        ),
        15: (
            "best_model_multimodal_256.weights.h5",
            "if RUN_384_EXPERIMENT",
        ),
        17: (
            "if RUN_384_EXPERIMENT",
            "count_tfrecord_items",
        ),
        19: (
            "best_model_multimodal_256.weights.h5",
            "if RUN_384_EXPERIMENT",
        ),
        21: (
            "THRESHOLD_SELECTION_384 = None",
            "METRICS_KERAS_384_TEST = None",
        ),
        23: (
            "'external_range': [0.0, 1.0]",
            "'backbone_range': [-1.0, 1.0]",
            "'backbone_rescaling_embedded_in_model': True",
            "'training_images': NUM_TRAINING_IMAGES",
            "'status': 'skipped'",
        ),
    }
    for cell_index, fragments in required_fragments.items():
        source = code_sources[cell_index]
        for fragment in fragments:
            if fragment not in source:
                raise RuntimeError(
                    f"Cell {cell_index} is missing contract fragment: "
                    f"{fragment}"
                )

    environment_source = code_sources[1]
    if "warnings.filterwarnings('ignore')" in environment_source:
        raise RuntimeError("The notebook suppresses all Python warnings.")
    if "* 2071" in code_sources[5]:
        raise RuntimeError("The notebook still assumes a fixed shard size.")
    if "cache=True" in code_sources[5]:
        raise RuntimeError(
            "The validation pipeline caches decoded images in memory."
        )

    count_function = next(
        node
        for node in ast.parse(code_sources[5]).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "count_tfrecord_items"
    )
    count_namespace = {"Path": Path}
    exec(
        compile(
            ast.Module(body=[count_function], type_ignores=[]),
            filename="<count_tfrecord_items>",
            mode="exec",
        ),
        count_namespace,
    )
    parsed_count = count_namespace["count_tfrecord_items"](
        ["train00-2182.tfrec", "train01-858.tfrec"]
    )
    if parsed_count != 3040:
        raise RuntimeError("TFRecord filename counts are parsed incorrectly.")

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            raise RuntimeError("The committed notebook contains cell outputs.")
        if cell.get("execution_count") is not None:
            raise RuntimeError(
                "The committed notebook contains execution counters."
            )

    print("Notebook validation: OK")


if __name__ == "__main__":
    main()
