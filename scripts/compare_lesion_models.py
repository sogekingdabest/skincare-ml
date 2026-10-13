"""Compare two exact artifacts on the frozen training run's held-out TFRecords."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from lesion_model_contract import FloatLesionInterpreter, encode_metadata, load_contract


def test_shards(metrics: dict, root: Path) -> list[Path]:
    split = metrics["run"]["split_strategy"]
    names = split["test_shards_2020"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicated test shard list")
    if set(names) & (set(split["training_shards_2020"]) | set(split["validation_shards_2020"])):
        raise ValueError("Test shards overlap another split")
    paths = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("Invalid shard filename")
        matches = list(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one test shard: {name}")
        paths.append(matches[0])
    return paths


def scores(labels, probabilities, threshold: float) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not 0 <= threshold <= 1 or not len(y) or set(y) != {0, 1} or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Evaluation requires both classes and valid scores")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "threshold": threshold, "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "confusion_matrix": {k: int(v) for k, v in zip(("tn", "fp", "fn", "tp"), (tn, fp, fn, tp))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy-model", "candidate-model", "legacy-contract", "candidate-contract", "metrics", "tfrecord-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    import tensorflow as tf

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    paths = test_shards(metrics, args.tfrecord_root)
    contracts = [load_contract(args.legacy_contract), load_contract(args.candidate_contract)]
    models = [FloatLesionInterpreter(args.legacy_model, contracts[0]), FloatLesionInterpreter(args.candidate_model, contracts[1])]
    schema = {"image": tf.io.FixedLenFeature([], tf.string), "target": tf.io.FixedLenFeature([], tf.int64),
              **{n: tf.io.FixedLenFeature([], tf.int64, default_value=-1) for n in ("age_approx", "sex", "anatom_site_general_challenge")}}
    rows, timings = [], [[], []]
    seen = set()
    for record in tf.data.TFRecordDataset([str(p) for p in paths], num_parallel_reads=1):
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break
        sample = tf.io.parse_single_example(record, schema)
        image_bytes = bytes(sample["image"].numpy())
        identity = hashlib.sha256(image_bytes).hexdigest()
        if identity in seen:
            raise ValueError("Duplicate image in test split")
        seen.add(identity)
        rgb = tf.image.decode_jpeg(image_bytes, channels=3).numpy()
        meta = encode_metadata(*(int(sample[k].numpy()) for k in ("age_approx", "sex", "anatom_site_general_challenge")))
        predicted = []
        for i, model in enumerate(models):
            started = time.perf_counter()
            predicted.append(model.predict(rgb, meta if len(model.inputs) == 2 else None))
            timings[i].append((time.perf_counter() - started) * 1000)
        rows.append({"image_sha256": identity, "label": int(sample["target"].numpy()), "legacy_score": predicted[0], "candidate_score": predicted[1]})
        if len(rows) % 500 == 0:
            print(f"Compared {len(rows)} samples", flush=True)
    expected = metrics["run"]["training_256"]["test_images"]
    if args.max_samples is None and len(rows) != expected:
        raise ValueError(f"Expected {expected} frozen test examples, found {len(rows)}")
    import numpy as np

    report = {"samples": len(rows), "full_frozen_test": len(rows) == expected,
              "training_metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
              "shards": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
              "resize": "tensorflow-bilinear; Android resize parity requires separate testing",
              "legacy_training_overlap": "unknown; source checkpoint is unresolved",
              "patient_audit_from_training_run": metrics["run"]["split_strategy"].get("patient_audit_256"),
              "evaluation_domain": "internal dermoscopic test; not phone capture validation",
              "models": {}}
    for i, name in enumerate(("legacy", "candidate")):
        # Legacy 0.327 was a confidence reference, not a validated decision threshold.
        threshold = 0.327 if i == 0 else float(metrics["threshold_selection"]["threshold"])
        report["models"][name] = {"descriptor": contracts[i], "metrics": scores([r["label"] for r in rows], [r[f"{name}_score"] for r in rows], threshold),
                                  "threshold_status": "unvalidated-legacy-reference" if i == 0 else "frozen-validation-selected",
                                  "cpu_preprocessing_and_inference_ms": {"median": float(np.median(timings[i])), "p95": float(np.percentile(timings[i], 95))}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "full_frozen_test": report["full_frozen_test"]}))


if __name__ == "__main__":
    main()
