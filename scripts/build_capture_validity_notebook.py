"""Generate the canonical capture-validity training notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "capture_validity_training.ipynb"


def markdown(source: str) -> dict:
    normalized = dedent(source).strip()
    return {
        "cell_type": "markdown",
        "id": "md-" + hashlib.sha1(normalized.encode()).hexdigest()[:12],
        "metadata": {},
        "source": normalized.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    normalized = dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "code-" + hashlib.sha1(normalized.encode()).hexdigest()[:12],
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
        # SkinCare capture-validity classifier

        This notebook trains a lightweight acquisition gate for the square
        region enclosing the Android camera guide. It predicts three semantic
        states: `target_lesion`, `skin_without_target` and `no_skin`.

        It is not a diagnostic model. The Android `READY` state must combine
        its `target_lesion` probability with the existing focus, light, scale and
        temporal-stability checks.

        Dataset contract and annotation rules:
        [`docs/CAPTURE_VALIDITY_DATASET.md`](../docs/CAPTURE_VALIDITY_DATASET.md).
        The state-of-the-art review is in
        [`docs/CAPTURE_VALIDITY_RESEARCH.md`](../docs/CAPTURE_VALIDITY_RESEARCH.md).
        """
    ),
    code(
        """
        import hashlib
        import json
        import os
        import platform
        import random
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import sklearn
        import tensorflow as tf
        from sklearn.metrics import (
            average_precision_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
        from tensorflow.keras import callbacks, layers, models

        SEED = 42
        IMAGE_SIZE = 192
        BATCH_SIZE = 64
        HEAD_EPOCHS = 12
        FINE_TUNE_EPOCHS = 8
        GENERALIZATION_ACCURACY_GAP_WARNING = 0.10
        VALIDATION_LOSS_REBOUND_WARNING = 0.05
        UNDERFIT_ACCURACY_WARNING = 0.80
        TARGET_FALSE_READY_RATE = 0.01
        THRESHOLD_RESOLUTION = 0.001
        FALSE_READY_CONFIDENCE_Z = 1.96
        MIN_RELEASE_VALIDATION_NON_TARGET = 400
        BACKBONE_WEIGHTS_FILENAME = (
            "weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5"
        )
        BACKBONE_WEIGHTS_MD5 = "d3e8ec802a04aa4fc771ee12a9a9b836"
        CLASS_NAMES = ["target_lesion", "skin_without_target", "no_skin"]
        CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
        READY_CLASS_INDEX = CLASS_TO_INDEX["target_lesion"]

        CONFIGURED_DATASET_ROOT = Path(
            os.environ.get(
                "CAPTURE_DATASET_ROOT",
                "/kaggle/input/skincare-capture-validity-tfrecords",
            )
        )


        def discover_dataset_root(configured_root):
            if (configured_root / "dataset_metadata.json").is_file():
                return configured_root
            input_root = Path("/kaggle/input")
            candidates = (
                sorted(input_root.rglob("dataset_metadata.json"))
                if input_root.is_dir()
                else []
            )
            compatible = []
            for candidate in candidates:
                try:
                    candidate_metadata = json.loads(
                        candidate.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    candidate_metadata.get("schema_version") == 2
                    and candidate_metadata.get("classes") == CLASS_NAMES
                    and candidate_metadata.get("ready_class") == "target_lesion"
                ):
                    compatible.append(candidate.parent)
            if len(compatible) != 1:
                raise FileNotFoundError(
                    "Expected exactly one compatible capture TFRecord dataset; "
                    f"configured={configured_root}, discovered={compatible}, "
                    f"input_entries={sorted(path.name for path in input_root.glob('*')) if input_root.is_dir() else []}"
                )
            return compatible[0]


        DATASET_ROOT = discover_dataset_root(CONFIGURED_DATASET_ROOT)
        METADATA_PATH = DATASET_ROOT / "dataset_metadata.json"


        def discover_backbone_weights():
            input_root = Path("/kaggle/input")
            candidates = sorted(input_root.rglob(BACKBONE_WEIGHTS_FILENAME))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "Expected exactly one pinned MobileNetV3-Small weights file; "
                    f"found={candidates}"
                )
            actual_md5 = hashlib.md5(candidates[0].read_bytes()).hexdigest()
            if actual_md5 != BACKBONE_WEIGHTS_MD5:
                raise ValueError(
                    f"Backbone weights MD5 mismatch: {actual_md5}"
                )
            return candidates[0]


        BACKBONE_WEIGHTS_PATH = discover_backbone_weights()
        OUTPUT_DIR = Path(
            os.environ.get(
                "CAPTURE_OUTPUT_DIR",
                "/kaggle/working/capture-validity",
            )
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        random.seed(SEED)
        np.random.seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        tf.config.experimental.enable_op_determinism()

        print(f"Python: {platform.python_version()}")
        print(f"TensorFlow: {tf.__version__}")
        print(f"scikit-learn: {sklearn.__version__}")
        print(f"Dataset root: {DATASET_ROOT}")
        print(f"Backbone weights: {BACKBONE_WEIGHTS_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")
        """
    ),
    markdown(
        """
        ## 1. TFRecord release contract

        Dataset selection, deduplication and group-aware splitting happen once
        in the CPU curation phase. This training notebook accepts only the
        resulting sharded TFRecord release and verifies its class and split
        metadata before enabling the GPU workload.
        """
    ),
    code(
        """
        if not METADATA_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {METADATA_PATH}. Run scripts/build_capture_tfrecords.py "
                "and attach its private Kaggle dataset output."
            )

        dataset_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if dataset_metadata.get("schema_version") != 2:
            raise ValueError("Unsupported capture TFRecord schema version.")
        if dataset_metadata.get("classes") != CLASS_NAMES:
            raise ValueError(
                "Dataset/model class contract mismatch: "
                f"{dataset_metadata.get('classes')} != {CLASS_NAMES}"
            )
        if dataset_metadata.get("ready_class") != "target_lesion":
            raise ValueError("The TFRecord release has an unexpected ready class.")
        weight_contract = dataset_metadata.get("sample_weight", {})
        if weight_contract.get("feature") != "sample_weight":
            raise ValueError("The TFRecord release is missing the v4 weight contract.")
        if weight_contract.get("held_out_all_standard") is not True:
            raise ValueError("Validation/test sample weights must all remain 1.0.")

        split_counts = dataset_metadata["split_counts"]
        for split_name in ("train", "validation", "test"):
            counts = split_counts.get(split_name, {})
            missing = set(CLASS_NAMES) - set(counts)
            if missing:
                raise ValueError(
                    f"{split_name} is missing classes: {sorted(missing)}"
                )

        validation_non_target = sum(
            count
            for label, count in split_counts["validation"].items()
            if label != "target_lesion"
        )
        validation_sample_size_sufficient = (
            validation_non_target >= MIN_RELEASE_VALIDATION_NON_TARGET
        )
        if not validation_sample_size_sufficient:
            print(
                "WARNING: exploratory calibration only. The public release gate "
                "needs at least "
                f"{MIN_RELEASE_VALIDATION_NON_TARGET} non-target validation "
                f"images; found {validation_non_target}."
            )

        split_table = pd.DataFrame(split_counts).fillna(0).astype(int).transpose()
        display(split_table[CLASS_NAMES])
        print(f"Images: {dataset_metadata['examples']:,}")
        print(f"Groups: {dataset_metadata['groups']:,}")
        print(f"Compression: {dataset_metadata['compression']}")
        print(f"Weighted training examples: {weight_contract['weighted_examples']}")
        """
    ),
    markdown(
        """
        ## 2. TensorFlow input pipeline

        TFRecords contain the encoded square camera-guide crop plus its class,
        source, group, checksum and reviewed v4 sample weight. Shards are read
        in parallel, decoding and resizing are parallelized, and `prefetch`
        overlaps CPU input work with GPU training. Mild semantic-preserving
        augmentation and sample weighting are training-only.
        """
    ),
    code(
        """
        AUTOTUNE = tf.data.AUTOTUNE

        augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical", seed=SEED),
                layers.RandomRotation(0.05, fill_mode="reflect", seed=SEED),
                layers.RandomZoom(
                    height_factor=(-0.08, 0.08),
                    width_factor=(-0.08, 0.08),
                    fill_mode="reflect",
                    seed=SEED,
                ),
                layers.RandomTranslation(
                    height_factor=0.05,
                    width_factor=0.05,
                    fill_mode="reflect",
                    seed=SEED,
                ),
                layers.RandomContrast(0.15, seed=SEED),
            ],
            name="training_augmentation",
        )


        FEATURE_SPEC = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "label/id": tf.io.FixedLenFeature([], tf.int64),
            "label/name": tf.io.FixedLenFeature([], tf.string),
            "group_id": tf.io.FixedLenFeature([], tf.string),
            "source": tf.io.FixedLenFeature([], tf.string),
            "sha256": tf.io.FixedLenFeature([], tf.string),
            "sample_weight": tf.io.FixedLenFeature([], tf.float32),
        }


        def parse_record(serialized, training=False):
            example = tf.io.parse_single_example(serialized, FEATURE_SPEC)
            image = tf.io.decode_image(
                example["image/encoded"],
                channels=3,
                expand_animations=False,
            )
            image.set_shape([None, None, 3])
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)
            if training:
                image = augmentation(image, training=True)
                image = tf.clip_by_value(image, 0.0, 1.0)
            label = tf.cast(example["label/id"], tf.int32)
            if training:
                return image, label, example["sample_weight"]
            return image, label


        def split_files(split):
            files = sorted(DATASET_ROOT.glob(f"{split}-*.tfrec*"))
            if not files:
                raise FileNotFoundError(f"No TFRecord shards found for {split}.")
            return [str(path) for path in files]


        def make_dataset(split, training=False, include_metadata=False):
            compression = (
                "GZIP" if dataset_metadata.get("compression") == "GZIP" else ""
            )
            dataset = tf.data.TFRecordDataset(
                split_files(split),
                compression_type=compression,
                num_parallel_reads=AUTOTUNE if training else 1,
            )
            if training:
                dataset = dataset.shuffle(
                    min(sum(split_counts[split].values()), 4096),
                    seed=SEED,
                    reshuffle_each_iteration=True,
                )
            if include_metadata:
                dataset = dataset.map(
                    lambda serialized: tf.io.parse_single_example(
                        serialized,
                        FEATURE_SPEC,
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=True,
                )
                return dataset
            dataset = dataset.map(
                lambda serialized: parse_record(serialized, training),
                num_parallel_calls=AUTOTUNE,
                deterministic=not training,
            )
            return dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


        train_dataset = make_dataset("train", training=True)
        validation_dataset = make_dataset("validation")
        test_dataset = make_dataset("test")

        sample_images, sample_labels, sample_weights = next(iter(train_dataset))
        assert sample_images.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert sample_images.dtype == tf.float32
        assert float(tf.reduce_min(sample_images)) >= 0.0
        assert float(tf.reduce_max(sample_images)) <= 1.0
        assert float(tf.reduce_min(sample_weights)) > 0.0
        print(sample_images.shape, sample_labels.shape, sample_weights.shape)
        """
    ),
    code(
        """
        figure, axes = plt.subplots(3, 4, figsize=(12, 9))
        for axis, image, label in zip(
            axes.flat,
            sample_images[:12],
            sample_labels[:12],
        ):
            axis.imshow(image)
            axis.set_title(CLASS_NAMES[int(label)])
            axis.axis("off")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 3. MobileNetV3-Small baseline

        The first release candidate uses MobileNetV3-Small because the task is a
        small mobile acquisition gate. The external contract is float RGB in
        `[0, 1]`; conversion to the backbone's `[-1, 1]` interval is embedded
        in the exported graph.
        """
    ),
    code(
        """
        def build_capture_validity_model():
            image_input = layers.Input(
                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                name="image",
            )
            normalized = layers.Rescaling(
                scale=2.0,
                offset=-1.0,
                name="mobilenet_input_rescaling",
            )(image_input)
            backbone = tf.keras.applications.MobileNetV3Small(
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                include_top=False,
                include_preprocessing=False,
                weights=str(BACKBONE_WEIGHTS_PATH),
            )
            backbone.trainable = False
            features = backbone(normalized, training=False)
            features = layers.GlobalAveragePooling2D()(features)
            features = layers.Dropout(0.25)(features)
            output = layers.Dense(
                len(CLASS_NAMES),
                activation="softmax",
                name="capture_state",
            )(features)
            model = models.Model(image_input, output)
            return model, backbone


        def compile_model(model, learning_rate):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[
                    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                ],
            )


        model, backbone = build_capture_validity_model()
        compile_model(model, learning_rate=1e-3)

        preprocessing_probe = model.get_layer("mobilenet_input_rescaling")(
            tf.constant([0.0, 0.5, 1.0], dtype=tf.float32)
        ).numpy()
        np.testing.assert_allclose(
            preprocessing_probe,
            [-1.0, 0.0, 1.0],
            atol=1e-6,
        )
        model.summary()
        """
    ),
    markdown(
        """
        ## 4. Frozen-backbone training

        Inverse-frequency class weights reduce domination by easy generic-scene
        negatives. Validation loss selects the checkpoint; Android's ready
        threshold is selected separately after training.
        """
    ),
    code(
        """
        train_class_counts = np.asarray(
            [split_counts["train"][name] for name in CLASS_NAMES],
            dtype=np.float64,
        )
        weights = np.sum(train_class_counts) / (
            len(CLASS_NAMES) * train_class_counts
        )
        class_weights = {
            int(class_id): float(min(weight, 10.0))
            for class_id, weight in enumerate(weights)
        }
        print(f"Class weights: {class_weights}")

        class_weight_tensor = tf.constant(
            [class_weights[index] for index in range(len(CLASS_NAMES))],
            dtype=tf.float32,
        )
        weighted_train_dataset = train_dataset.map(
            lambda images, labels, sample_weights: (
                images,
                labels,
                sample_weights * tf.gather(class_weight_tensor, labels),
            ),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        ).prefetch(AUTOTUNE)

        checkpoint_path = OUTPUT_DIR / "best_capture_validity.keras"
        training_callbacks = [
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        head_history = model.fit(
            weighted_train_dataset,
            validation_data=validation_dataset,
            epochs=HEAD_EPOCHS,
            callbacks=training_callbacks,
        )
        """
    ),
    markdown(
        """
        ## 5. Conservative fine-tuning

        Only the final MobileNet block is unfrozen. Batch-normalization layers
        remain frozen because mobile dermatology batches are small and their
        running statistics are easy to destabilize.
        """
    ),
    code(
        """
        model = tf.keras.models.load_model(checkpoint_path)
        backbone = next(
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "mobilenetv3" in layer.name.lower()
        )
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        compile_model(model, learning_rate=1e-5)
        fine_tune_history = model.fit(
            weighted_train_dataset,
            validation_data=validation_dataset,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=training_callbacks,
        )
        model = tf.keras.models.load_model(checkpoint_path)
        """
    ),
    markdown(
        """
        ## 6. Training diagnostics

        Training and validation histories from both phases are joined into one
        auditable timeline. The warnings below are engineering heuristics, not
        release criteria: they make possible overfitting or underfitting visible
        without replacing held-out, per-class and real-device evaluation.
        """
    ),
    code(
        """
        def history_frame(history, phase, epoch_offset):
            frame = pd.DataFrame(history.history)
            frame.insert(0, "phase_epoch", np.arange(1, len(frame) + 1))
            frame.insert(
                0,
                "global_epoch",
                np.arange(epoch_offset + 1, epoch_offset + len(frame) + 1),
            )
            frame.insert(0, "phase", phase)
            return frame


        head_history_frame = history_frame(head_history, "frozen_backbone", 0)
        fine_tune_history_frame = history_frame(
            fine_tune_history,
            "fine_tuning",
            len(head_history_frame),
        )
        training_history = pd.concat(
            [head_history_frame, fine_tune_history_frame],
            ignore_index=True,
        )

        required_history_columns = {
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
        }
        missing_history_columns = required_history_columns - set(
            training_history.columns
        )
        if missing_history_columns:
            raise RuntimeError(
                "Training history is missing metrics: "
                f"{sorted(missing_history_columns)}"
            )

        best_index = int(training_history["val_loss"].idxmin())
        best_epoch = training_history.loc[best_index]
        final_epoch = training_history.iloc[-1]
        validation_loss_rebound = max(
            0.0,
            float(final_epoch["val_loss"] - best_epoch["val_loss"])
            / max(abs(float(best_epoch["val_loss"])), 1e-8),
        )
        best_accuracy_gap = float(
            best_epoch["accuracy"] - best_epoch["val_accuracy"]
        )
        maximum_training_accuracy = float(training_history["accuracy"].max())
        maximum_validation_accuracy = float(
            training_history["val_accuracy"].max()
        )

        overfitting_signals = []
        if best_accuracy_gap > GENERALIZATION_ACCURACY_GAP_WARNING:
            overfitting_signals.append("best_epoch_accuracy_gap")
        if (
            int(best_epoch["global_epoch"])
            < int(final_epoch["global_epoch"])
            and validation_loss_rebound > VALIDATION_LOSS_REBOUND_WARNING
        ):
            overfitting_signals.append("validation_loss_rebound")

        underfitting_signals = []
        if (
            maximum_training_accuracy < UNDERFIT_ACCURACY_WARNING
            and maximum_validation_accuracy < UNDERFIT_ACCURACY_WARNING
        ):
            underfitting_signals.append("low_training_and_validation_accuracy")

        if overfitting_signals and underfitting_signals:
            diagnostic_status = "mixed_warning"
        elif overfitting_signals:
            diagnostic_status = "possible_overfitting"
        elif underfitting_signals:
            diagnostic_status = "possible_underfitting"
        else:
            diagnostic_status = "no_heuristic_warning"

        training_diagnostics = {
            "status": diagnostic_status,
            "heuristic_only": True,
            "best_checkpoint_monitor": "val_loss",
            "best_global_epoch": int(best_epoch["global_epoch"]),
            "best_phase": str(best_epoch["phase"]),
            "best_phase_epoch": int(best_epoch["phase_epoch"]),
            "best_train_loss": float(best_epoch["loss"]),
            "best_validation_loss": float(best_epoch["val_loss"]),
            "best_train_accuracy": float(best_epoch["accuracy"]),
            "best_validation_accuracy": float(best_epoch["val_accuracy"]),
            "best_accuracy_gap": best_accuracy_gap,
            "validation_loss_rebound_from_best": validation_loss_rebound,
            "maximum_training_accuracy": maximum_training_accuracy,
            "maximum_validation_accuracy": maximum_validation_accuracy,
            "overfitting_signals": overfitting_signals,
            "underfitting_signals": underfitting_signals,
            "thresholds": {
                "accuracy_gap_warning": GENERALIZATION_ACCURACY_GAP_WARNING,
                "validation_loss_rebound_warning": (
                    VALIDATION_LOSS_REBOUND_WARNING
                ),
                "underfit_accuracy_warning": UNDERFIT_ACCURACY_WARNING,
            },
            "phases": {
                "frozen_backbone_epochs_run": int(len(head_history_frame)),
                "frozen_backbone_max_epochs": HEAD_EPOCHS,
                "fine_tuning_epochs_run": int(len(fine_tune_history_frame)),
                "fine_tuning_max_epochs": FINE_TUNE_EPOCHS,
            },
        }

        training_history_path = OUTPUT_DIR / "training_history.csv"
        training_diagnostics_path = OUTPUT_DIR / "training_diagnostics.json"
        training_curves_path = OUTPUT_DIR / "training_curves.png"
        training_history.to_csv(training_history_path, index=False)
        training_diagnostics_path.write_text(
            json.dumps(training_diagnostics, indent=2),
            encoding="utf-8",
        )

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, train_metric, validation_metric, title in (
            (axes[0], "loss", "val_loss", "Loss"),
            (axes[1], "accuracy", "val_accuracy", "Accuracy"),
        ):
            axis.plot(
                training_history["global_epoch"],
                training_history[train_metric],
                marker="o",
                label="training",
            )
            axis.plot(
                training_history["global_epoch"],
                training_history[validation_metric],
                marker="o",
                label="validation",
            )
            if len(head_history_frame) < len(training_history):
                axis.axvline(
                    len(head_history_frame) + 0.5,
                    color="grey",
                    linestyle="--",
                    label="fine-tuning starts",
                )
            axis.axvline(
                int(best_epoch["global_epoch"]),
                color="green",
                linestyle=":",
                label="best val_loss",
            )
            axis.set_xlabel("Global epoch")
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.legend()
        plt.tight_layout()
        plt.savefig(training_curves_path, dpi=150, bbox_inches="tight")
        plt.show()

        display(training_history)
        print("Training diagnostic (heuristic, not a release gate):")
        print(json.dumps(training_diagnostics, indent=2))
        print(f"Wrote {training_history_path}")
        print(f"Wrote {training_diagnostics_path}")
        print(f"Wrote {training_curves_path}")
        """
    ),
    markdown(
        """
        ## 7. Threshold selection and held-out evaluation

        Multiclass metrics use `argmax`. Android ready/not-ready is a separate
        binary operating point over `P(target_lesion)`. Its threshold is selected
        only on validation. A release candidate must satisfy the configured
        Wilson-bound ceiling. While the validation set is too small to prove
        that bound, the exploratory fallback requires zero observed false READY
        decisions and chooses the smallest recall-maximizing threshold. The
        frozen threshold is then evaluated on test.
        """
    ),
    code(
        """
        def predict_split(split):
            dataset = make_dataset(split)
            probabilities = model.predict(dataset, verbose=1)
            labels = np.concatenate(
                [batch_labels.numpy() for _, batch_labels in dataset],
                axis=0,
            )
            metadata = pd.DataFrame(
                [
                    {
                        "sha256": example["sha256"].numpy().decode("utf-8"),
                        "group_id": example["group_id"].numpy().decode("utf-8"),
                        "source": example["source"].numpy().decode("utf-8"),
                    }
                    for example in make_dataset(split, include_metadata=True)
                ]
            )
            if not (len(labels) == len(probabilities) == len(metadata)):
                raise RuntimeError("Prediction and TFRecord metadata order mismatch.")
            return labels, probabilities, metadata


        def build_prediction_frame(
            split,
            labels,
            probabilities,
            metadata,
            ready_threshold,
        ):
            predictions = np.argmax(probabilities, axis=1)
            frame = metadata.copy()
            frame.insert(0, "record_index", np.arange(len(frame)))
            frame.insert(0, "split", split)
            frame["true_label_id"] = labels
            frame["true_label"] = [CLASS_NAMES[index] for index in labels]
            frame["predicted_label_id"] = predictions
            frame["predicted_label"] = [
                CLASS_NAMES[index] for index in predictions
            ]
            for index, class_name in enumerate(CLASS_NAMES):
                frame[f"probability_{class_name}"] = probabilities[:, index]
            frame["ready_threshold"] = ready_threshold
            frame["predicted_ready"] = (
                probabilities[:, READY_CLASS_INDEX] >= ready_threshold
            )

            error_types = []
            for true_label, predicted_label, predicted_ready in zip(
                labels,
                predictions,
                frame["predicted_ready"],
            ):
                errors = []
                if predicted_label != true_label:
                    errors.append("argmax_error")
                truly_ready = true_label == READY_CLASS_INDEX
                if predicted_ready and not truly_ready:
                    errors.append("ready_false_positive")
                elif not predicted_ready and truly_ready:
                    errors.append("ready_false_negative")
                error_types.append("|".join(errors))
            frame["error_types"] = error_types
            return frame


        def ready_statistics(labels, target_scores, threshold):
            positives = labels == READY_CLASS_INDEX
            predicted_ready = target_scores >= threshold
            true_positives = int(np.sum(predicted_ready & positives))
            false_positives = int(np.sum(predicted_ready & ~positives))
            false_negatives = int(np.sum(~predicted_ready & positives))
            true_negatives = int(np.sum(~predicted_ready & ~positives))

            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            false_ready_rate = false_positives / max(
                false_positives + true_negatives,
                1,
            )
            negative_count = false_positives + true_negatives
            denominator = 1.0 + FALSE_READY_CONFIDENCE_Z**2 / max(
                negative_count,
                1,
            )
            centre = (
                false_ready_rate
                + FALSE_READY_CONFIDENCE_Z**2 / (2 * max(negative_count, 1))
            ) / denominator
            margin = (
                FALSE_READY_CONFIDENCE_Z
                * np.sqrt(
                    false_ready_rate * (1.0 - false_ready_rate)
                    / max(negative_count, 1)
                    + FALSE_READY_CONFIDENCE_Z**2
                    / (4 * max(negative_count, 1) ** 2)
                )
                / denominator
            )
            return {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "false_ready_rate": float(false_ready_rate),
                "false_ready_rate_wilson_upper_95": float(centre + margin),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
            }


        def select_ready_threshold(labels, target_scores, maximum_false_ready_rate):
            threshold_steps = int(round(1.0 / THRESHOLD_RESOLUTION))
            candidates = [
                ready_statistics(labels, target_scores, threshold)
                for threshold in np.linspace(0.0, 1.0, threshold_steps + 1)
            ]
            feasible = [
                candidate
                for candidate in candidates
                if candidate["false_ready_rate_wilson_upper_95"]
                <= maximum_false_ready_rate
            ]
            pool = feasible
            selection_mode = "wilson_upper_95"
            if not pool:
                pool = [
                    candidate
                    for candidate in candidates
                    if candidate["false_positives"] == 0
                ]
                selection_mode = "exploratory_zero_observed_false_ready"
            if not pool:
                raise RuntimeError(
                    "No zero-false-READY threshold is available at this resolution."
                )
            selected = max(
                pool,
                key=lambda item: (
                    item["recall"],
                    item["precision"],
                    -item["threshold"],
                ),
            )
            selected["selection_mode"] = selection_mode
            selected["release_target_satisfied"] = bool(feasible)
            selected["threshold_resolution"] = THRESHOLD_RESOLUTION
            selected["selected_on_split"] = "validation"
            selected["test_used_for_selection"] = False
            return selected


        validation_labels, validation_probabilities, validation_metadata = (
            predict_split("validation")
        )
        threshold_selection = select_ready_threshold(
            validation_labels,
            validation_probabilities[:, READY_CLASS_INDEX],
            TARGET_FALSE_READY_RATE,
        )
        print("Validation operating point:")
        print(json.dumps(threshold_selection, indent=2))

        test_labels, test_probabilities, test_metadata = predict_split("test")
        test_sources = test_metadata["source"].to_numpy()
        test_predictions = np.argmax(test_probabilities, axis=1)
        test_confusion = confusion_matrix(
            test_labels,
            test_predictions,
            labels=list(range(len(CLASS_NAMES))),
        )
        test_report = classification_report(
            test_labels,
            test_predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        test_ready = ready_statistics(
            test_labels,
            test_probabilities[:, READY_CLASS_INDEX],
            threshold_selection["threshold"],
        )
        test_ready["roc_auc"] = float(
            roc_auc_score(
                test_labels == READY_CLASS_INDEX,
                test_probabilities[:, READY_CLASS_INDEX],
            )
        )
        test_ready["average_precision"] = float(
            average_precision_score(
                test_labels == READY_CLASS_INDEX,
                test_probabilities[:, READY_CLASS_INDEX],
            )
        )

        no_skin_mask = test_labels == CLASS_TO_INDEX["no_skin"]
        test_ready["false_ready_rate_no_skin"] = float(
            np.mean(
                test_probabilities[no_skin_mask, READY_CLASS_INDEX]
                >= threshold_selection["threshold"]
            )
        )

        display(pd.DataFrame(test_report).transpose())
        print("Test ready gate:")
        print(json.dumps(test_ready, indent=2))

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            test_confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        source_metrics = []
        for source in sorted(set(test_sources)):
            source_mask = test_sources == source
            local_labels = test_labels[source_mask]
            local_probabilities = test_probabilities[source_mask]
            local_ready = ready_statistics(
                local_labels,
                local_probabilities[:, READY_CLASS_INDEX],
                threshold_selection["threshold"],
            )
            local_ready.update(
                {
                    "source": source,
                    "samples": int(np.sum(source_mask)),
                }
            )
            source_metrics.append(local_ready)

        source_metrics_frame = pd.DataFrame(source_metrics).sort_values("source")
        display(source_metrics_frame)

        validation_predictions = build_prediction_frame(
            "validation",
            validation_labels,
            validation_probabilities,
            validation_metadata,
            threshold_selection["threshold"],
        )
        test_predictions_frame = build_prediction_frame(
            "test",
            test_labels,
            test_probabilities,
            test_metadata,
            threshold_selection["threshold"],
        )
        validation_predictions_path = OUTPUT_DIR / "validation_predictions.csv"
        test_predictions_path = OUTPUT_DIR / "test_predictions.csv"
        validation_predictions.to_csv(validation_predictions_path, index=False)
        test_predictions_frame.to_csv(test_predictions_path, index=False)
        print(f"Wrote {validation_predictions_path}")
        print(f"Wrote {test_predictions_path}")
        """
    ),
    markdown(
        """
        ## 8. Float16 TFLite export and parity

        The export keeps float input/output tensors while quantizing weights to
        float16. A parity sample compares Keras and the exact `.tflite` artifact
        that will later run through LiteRT on Android.
        """
    ),
    code(
        """
        final_keras_path = OUTPUT_DIR / "capture_validity_mobilenetv3_small.keras"
        tflite_path = (
            OUTPUT_DIR
            / "capture_validity_mobilenetv3_small_float16.tflite"
        )
        model.save(final_keras_path)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)

        parity_images = np.concatenate(
            [images.numpy() for images, _ in test_dataset.take(4)],
            axis=0,
        )[:256]
        keras_parity = model.predict(parity_images, verbose=0)

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        tflite_parity = []
        for image in parity_images:
            interpreter.set_tensor(
                input_detail["index"],
                image[np.newaxis].astype(input_detail["dtype"]),
            )
            interpreter.invoke()
            tflite_parity.append(
                interpreter.get_tensor(output_detail["index"])[0]
            )
        tflite_parity = np.asarray(tflite_parity)

        parity = {
            "samples": int(len(parity_images)),
            "max_absolute_probability_error": float(
                np.max(np.abs(keras_parity - tflite_parity))
            ),
            "mean_absolute_probability_error": float(
                np.mean(np.abs(keras_parity - tflite_parity))
            ),
            "argmax_agreement": float(
                np.mean(
                    np.argmax(keras_parity, axis=1)
                    == np.argmax(tflite_parity, axis=1)
                )
            ),
            "ready_decision_agreement": float(
                np.mean(
                    (
                        keras_parity[:, READY_CLASS_INDEX]
                        >= threshold_selection["threshold"]
                    )
                    == (
                        tflite_parity[:, READY_CLASS_INDEX]
                        >= threshold_selection["threshold"]
                    )
                )
            ),
        }
        print(json.dumps(parity, indent=2))
        print(f"TFLite size: {tflite_path.stat().st_size / 1_000_000:.2f} MB")
        """
    ),
    markdown(
        """
        ## 9. Reproducibility report

        The report records the split provenance, operating threshold, metrics,
        per-source slices, preprocessing contract and artifact hash. It is the
        hand-off contract for the later Android integration.
        """
    ),
    code(
        """
        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        def json_safe(value):
            if isinstance(value, dict):
                return {str(key): json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value


        metrics = {
            "task": "capture_semantic_gate",
            "intended_use": "non_diagnostic_android_camera_acquisition_gate",
            "classes": CLASS_NAMES,
            "ready_class": "target_lesion",
            "split_provenance": "curated_tfrecord_release",
            "split_counts": split_counts,
            "group_count": int(dataset_metadata["groups"]),
            "dataset_release": {
                "schema_version": dataset_metadata["schema_version"],
                "compression": dataset_metadata["compression"],
                "shards": len(dataset_metadata["shards"]),
                "metadata_sha256": sha256_file(METADATA_PATH),
                "sample_weight": dataset_metadata["sample_weight"],
            },
            "backbone_weights": {
                "file": BACKBONE_WEIGHTS_PATH.name,
                "md5": BACKBONE_WEIGHTS_MD5,
                "sha256": sha256_file(BACKBONE_WEIGHTS_PATH),
                "source": "pinned_private_kaggle_dataset",
            },
            "preprocessing": {
                "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3],
                "dtype": "float32",
                "external_range": [0.0, 1.0],
                "backbone_range": [-1.0, 1.0],
                "rescaling_embedded_in_model": True,
                "roi": "square_crop_enclosing_android_circular_guide",
            },
            "training_diagnostics": training_diagnostics,
            "threshold_selection_validation": threshold_selection,
            "threshold_policy": {
                "target_false_ready_rate": TARGET_FALSE_READY_RATE,
                "confidence_bound": "wilson_upper_95",
                "exploratory_fallback": "zero_observed_false_ready",
                "threshold_resolution": THRESHOLD_RESOLUTION,
                "selected_on_split": "validation",
                "test_used_for_selection": False,
                "minimum_non_target_validation_images": (
                    MIN_RELEASE_VALIDATION_NON_TARGET
                ),
                "validation_non_target_images": int(validation_non_target),
                "validation_sample_size_sufficient": bool(
                    validation_sample_size_sufficient
                ),
            },
            "test_ready_gate": test_ready,
            "test_multiclass_report": test_report,
            "test_confusion_matrix": test_confusion,
            "test_by_source": source_metrics,
            "parity": parity,
            "artifacts": {
                "keras": final_keras_path.name,
                "tflite": tflite_path.name,
                "tflite_sha256": sha256_file(tflite_path),
                "tflite_bytes": int(tflite_path.stat().st_size),
                "validation_predictions": validation_predictions_path.name,
                "test_predictions": test_predictions_path.name,
                "training_history": training_history_path.name,
                "training_diagnostics": training_diagnostics_path.name,
                "training_curves": training_curves_path.name,
            },
            "environment": {
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }

        metrics["release_gate"] = {
            "eligible": bool(
                dataset_metadata.get("release_eligible", False)
                and validation_sample_size_sufficient
                and threshold_selection["release_target_satisfied"]
                and parity["argmax_agreement"] == 1.0
                and parity["ready_decision_agreement"] == 1.0
            ),
            "dataset_publication_scope": dataset_metadata.get(
                "publication_scope", "unknown"
            ),
            "dataset_rights_verified": bool(
                dataset_metadata.get("release_eligible", False)
            ),
            "validation_sample_size_sufficient": bool(
                validation_sample_size_sufficient
            ),
            "confidence_target_satisfied": bool(
                threshold_selection["release_target_satisfied"]
            ),
        }

        metrics_path = OUTPUT_DIR / "capture_validity_metrics.json"
        metrics_path.write_text(
            json.dumps(json_safe(metrics), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote {metrics_path}")
        print(f"Wrote {final_keras_path}")
        print(f"Wrote {tflite_path}")
        """
    ),
    markdown(
        """
        ## Release decision

        A successful execution is only a candidate. Do not copy the artifact
        into Android until:

        - group and duplicate leakage checks pass;
        - the validation-selected threshold is frozen;
        - the false-ready/recall trade-off is acceptable on held-out real
          SkinCare captures;
        - results are reviewed by source, device and available skin-tone
          slices;
        - TFLite/LiteRT parity and on-device latency are measured.

        The next experiment after this baseline should compare MobileNetV4 or a
        second compact backbone only after dataset errors and hard-negative
        failures have been reviewed.
        """
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
