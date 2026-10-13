"""Evaluate digital image perturbations with frozen validation policies on test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

SCENARIOS = ("legacy", "recorded_context", "site_only", "no_context")
CONDITIONS = (
    "original", "light_0.6", "light_0.8", "light_1.2", "light_1.4",
    "contrast_0.7", "contrast_1.3", "blur_1", "blur_3", "blur_6",
    "noise_4", "noise_12", "warm_cast", "cool_cast", "jpeg_40", "jpeg_80",
    "rotate_90", "rotate_180", "central_crop_80",
)
ENTRY_POINT = '\nif __name__ == "__main__":\n    main()\n'


def image_variants(rgb: np.ndarray, identity: str, seed: int) -> dict[str, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 16:
        raise ValueError("Expected an RGB uint8 image at least 16 pixels wide/high")
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("Expected a SHA-256 image identity")
    def pixels(value):
        return np.clip(np.rint(value), 0, 255).astype(np.uint8)
    images = {"original": rgb.copy()}
    for factor in (0.6, 0.8, 1.2, 1.4):
        images[f"light_{factor:.1f}"] = pixels(rgb.astype(float) * factor)
    for factor in (0.7, 1.3):
        images[f"contrast_{factor:.1f}"] = pixels(128 + (rgb.astype(float) - 128) * factor)
    for radius in (1, 3, 6):
        images[f"blur_{radius}"] = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius))).copy()
    keyed_seed = int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(keyed_seed)
    for sigma in (4, 12):
        images[f"noise_{sigma}"] = pixels(rgb.astype(float) + rng.normal(0, sigma, rgb.shape))
    images["warm_cast"] = pixels(rgb.astype(float) * np.array([1.10, 1.0, 0.90]))
    images["cool_cast"] = pixels(rgb.astype(float) * np.array([0.90, 1.0, 1.10]))
    for quality in (40, 80):
        with io.BytesIO() as stream:
            Image.fromarray(rgb).save(stream, format="JPEG", quality=quality, subsampling=2)
            stream.seek(0)
            images[f"jpeg_{quality}"] = np.asarray(Image.open(stream).convert("RGB")).copy()
    images["rotate_90"] = np.rot90(rgb, 1).copy()
    images["rotate_180"] = np.rot90(rgb, 2).copy()
    height, width = rgb.shape[:2]
    my, mx = height // 10, width // 10
    images["central_crop_80"] = np.asarray(Image.fromarray(rgb[my:-my, mx:-mx]).resize((width, height), Image.Resampling.BILINEAR)).copy()
    return images


def select_test_sample(records: list[dict], negative_count: int, seed: int) -> list[dict]:
    if negative_count < 1 or not records:
        raise ValueError("Need a nonempty test and a positive negative-sample size")
    identities = [row["image_sha256"] for row in records]
    if len(set(identities)) != len(identities) or any(row["label"] not in (0, 1) for row in records):
        raise ValueError("Repeated images or invalid labels")
    positives = [row for row in records if row["label"] == 1]
    negatives = [row for row in records if row["label"] == 0]
    if not positives or len(negatives) < negative_count:
        raise ValueError("Insufficient cases for the predefined sample")
    negatives.sort(key=lambda row: (hashlib.sha256(f"{seed}:{row['image_sha256']}".encode()).hexdigest(), row["image_sha256"]))
    return sorted(positives + negatives[:negative_count], key=lambda row: row["image_sha256"])


def validate_policy(policy: dict, models: dict, metrics_sha: str) -> None:
    plan = policy["plan"]
    if plan["selection_split"] != "validation" or plan["metrics_sha256"] != metrics_sha:
        raise ValueError("Wrong threshold-selection split or training reference")
    for name in ("legacy", "candidate"):
        if plan[f"{name}_sha256"] != models[name]["sha256"]:
            raise ValueError("Frozen policy belongs to a different artifact")
    if set(policy["operating_points"]) != set(SCENARIOS):
        raise ValueError("Incomplete context-specific policies")
    for points in policy["operating_points"].values():
        if [p["minimum_sensitivity"] for p in points] != [0.85, 0.90, 0.95]:
            raise ValueError("Unexpected predefined sensitivity targets")
        if any(p["selected_on"] != "validation" or not np.isfinite(p["threshold"]) or not 0 <= p["threshold"] <= 1 for p in points):
            raise ValueError("Invalid frozen operating point")


def paired_counts(labels, original, perturbed, threshold: float) -> dict:
    y, before, after = np.asarray(labels), np.asarray(original), np.asarray(perturbed)
    if y.ndim != 1 or before.shape != y.shape or after.shape != y.shape or set(y) != {0, 1}:
        raise ValueError("Need matched arrays and both classes")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or any(not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) for p in (before, after)):
        raise ValueError("Invalid scores or threshold")
    b, a, pos = before >= threshold, after >= threshold, y == 1
    return {"tp": int(np.sum(a & pos)), "fn": int(np.sum(~a & pos)),
        "tn": int(np.sum(~a & ~pos)), "fp": int(np.sum(a & ~pos)),
        "positive_detected_to_missed": int(np.sum(b & ~a & pos)),
        "positive_missed_to_detected": int(np.sum(~b & a & pos)),
        "negative_correct_to_false_alarm": int(np.sum(~b & a & ~pos)),
        "negative_false_alarm_to_correct": int(np.sum(b & ~a & ~pos))}


def summarize(rows: list[dict], policies: dict, seed: int, bootstrap_repeats: int = 1000) -> dict:
    if bootstrap_repeats < 100:
        raise ValueError("Need at least 100 patient-bootstrap repeats")
    identity = sorted({r["image_sha256"] for r in rows})
    patients = sorted({r["patient_group"] for r in rows})
    patient_index = {p: i for i, p in enumerate(patients)}
    cells, counts, baseline_cells, results = [], [], {}, {}
    conditions = [condition for condition in CONDITIONS if any(r["condition"] == condition for r in rows)]
    if not conditions or conditions[0] != "original":
        raise ValueError("Missing original condition")
    score_rows = {}
    for row in rows:
        key = (row["condition"], row["scenario"], row["image_sha256"])
        if key in score_rows:
            raise ValueError("Repeated prediction")
        score_rows[key] = row
    for scenario in SCENARIOS:
        original = [score_rows[("original", scenario, i)] for i in identity]
        labels = np.array([r["label"] for r in original])
        scores_before = np.array([r["score"] for r in original])
        for condition in conditions:
            current = [score_rows[(condition, scenario, i)] for i in identity]
            if any((a["label"], a["patient_group"]) != (b["label"], b["patient_group"]) for a, b in zip(current, original, strict=True)):
                raise ValueError("Label/patient changed under perturbation")
            scores_after = np.array([r["score"] for r in current])
            results.setdefault(condition, {})[scenario] = {"sample_roc_auc": float(roc_auc_score(labels, scores_after)),
                "mean_absolute_score_change_points": float(np.mean(np.abs(scores_after - scores_before)) * 100), "operating_points": []}
            for point in policies[scenario]:
                outcome = paired_counts(labels, scores_before, scores_after, point["threshold"])
                matrix = np.zeros((len(patients), 4), dtype=np.int64)
                for row in current:
                    predicted = row["score"] >= point["threshold"]
                    column = (0 if predicted else 1) if row["label"] == 1 else (3 if predicted else 2)
                    matrix[patient_index[row["patient_group"]], column] += 1
                index = len(cells)
                key = (scenario, point["minimum_sensitivity"])
                if condition == "original":
                    baseline_cells[key] = index
                cells.append((condition, scenario, point, outcome, baseline_cells[key]))
                counts.append(matrix)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(patients), [1 / len(patients)] * len(patients), size=bootstrap_repeats).astype(float)
    matrix = np.stack(counts, axis=1).astype(float)
    boot = (weights @ matrix.reshape(len(patients), -1)).reshape(bootstrap_repeats, len(cells), 4)
    def rates(values):
        with np.errstate(divide="ignore", invalid="ignore"):
            return values[..., 0] / (values[..., 0] + values[..., 1]), values[..., 2] / (values[..., 2] + values[..., 3])
    sensitivity, specificity = rates(boot)
    def interval(values):
        valid = values[np.isfinite(values)]
        return {"interval_95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
            "valid_replicates": len(valid)}
    total = matrix.sum(axis=0)
    sen, spe = rates(total)
    for index, (condition, scenario, point, outcome, base) in enumerate(cells):
        result = {"selection": point, "counts": outcome, "sensitivity": float(sen[index]), "specificity": float(spe[index]),
            "delta_sensitivity": float(sen[index] - sen[base]), "delta_specificity": float(spe[index] - spe[base]),
            "patient_bootstrap_sensitivity": interval(sensitivity[:, index]),
            "patient_bootstrap_specificity": interval(specificity[:, index]),
            "patient_bootstrap_delta_sensitivity": interval(sensitivity[:, index] - sensitivity[:, base]),
            "patient_bootstrap_delta_specificity": interval(specificity[:, index] - specificity[:, base])}
        results[condition][scenario]["operating_points"].append(result)
    return {"results": results, "patient_groups": len(patients), "bootstrap_repeats": bootstrap_repeats}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-dir", "reference-dir", "tfrecord-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("reference-policy-sha", "reference-predictions-sha", "reference-report-sha"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--negative-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle_dir))
    from lesion_model_contract import FloatLesionInterpreter, load_contract
    from lesion_metadata_policy import scenario_metadata
    import tensorflow as tf

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.reference_dir / "frozen-validation-policy.json"
    predictions_path = args.reference_dir / "test_predictions.csv"
    previous_report_path = args.reference_dir / "metadata-policy-report.json"
    if (file_sha(frozen_path) != args.reference_policy_sha or file_sha(predictions_path) != args.reference_predictions_sha
            or file_sha(previous_report_path) != args.reference_report_sha):
        raise ValueError("Previous validation policy/test reference changed")
    previous_report = json.loads(previous_report_path.read_text())
    policy = json.loads(frozen_path.read_text())
    models = {name: load_contract(args.bundle_dir / f"{name}.json") for name in ("legacy", "candidate")}
    metrics_path = args.bundle_dir / "metrics_256.json"
    metrics = json.loads(metrics_path.read_text())
    validate_policy(policy, models, file_sha(metrics_path))
    plan = {"seed": args.seed, "negative_count": args.negative_count, "positive_selection": "all test positives",
        "negative_selection": "lowest SHA256(seed:image_sha256), independent of model scores",
        "conditions": CONDITIONS, "scenarios": SCENARIOS, "label_source": "original lesion label retained under digital perturbation",
        "frozen_policy_sha256": args.reference_policy_sha, "previous_test_predictions_sha256": args.reference_predictions_sha,
        "previous_report_sha256": args.reference_report_sha,
        "model_sha256": {name: value["sha256"] for name, value in models.items()},
        "metrics_sha256": file_sha(metrics_path), "script_sha256": file_sha(Path(__file__)),
        "bootstrap_repeats": 1000, "runtime": {"tensorflow": tf.__version__, "numpy": np.__version__, "tflite_threads": 4},
        "candidate_released": False, "photos_saved_in_output": False, "study": "exploratory, reused internal test"}
    plan["supporting_source_sha256"] = {name: file_sha(args.bundle_dir / name)
        for name in ("lesion_model_contract.py", "lesion_metadata_policy.py")}
    (args.output_dir / "experiment-plan.json").write_text(json.dumps(plan, indent=2))
    reference_rows = list(csv.DictReader(predictions_path.open(newline="", encoding="utf-8")))
    reference = {row["image_sha256"]: row for row in reference_rows}
    if len(reference_rows) != len(reference):
        raise ValueError("Duplicate original test prediction")
    split = metrics["run"]["split_strategy"]
    shard_names = split["test_shards_2020"]
    if set(shard_names) & (set(split["training_shards_2020"]) | set(split["validation_shards_2020"])):
        raise ValueError("Test overlaps training or validation shards")
    schema = {"image": tf.io.FixedLenFeature([], tf.string), "target": tf.io.FixedLenFeature([], tf.int64),
        **{name: tf.io.FixedLenFeature([], tf.int64, default_value=-1) for name in ("age_approx", "sex", "anatom_site_general_challenge", "patient_id")}}
    records, seen, shards = [], set(), []
    for name in shard_names:
        matches = list(args.tfrecord_root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one test shard {name}")
        shard_sha = file_sha(matches[0])
        original_sha = next(r["sha256"] for r in previous_report["shards"] if r["name"] == name)
        if shard_sha != original_sha:
            raise ValueError("Frozen test shard changed")
        shards.append({"name": name, "sha256": shard_sha})
        for raw in tf.data.TFRecordDataset([str(matches[0])], num_parallel_reads=1):
            sample = tf.io.parse_single_example(raw, schema)
            image = bytes(sample["image"].numpy())
            identity = hashlib.sha256(image).hexdigest()
            label, patient = int(sample["target"].numpy()), int(sample["patient_id"].numpy())
            if identity in seen or identity not in reference or int(reference[identity]["label"]) != label:
                raise ValueError("Test image/label identity changed")
            seen.add(identity)
            patient_group = "p:" + hashlib.sha256(f"cdeotte2020:{patient}".encode()).hexdigest() if patient >= 0 else "i:" + identity
            records.append({"image_sha256": identity, "label": label, "patient_group": patient_group,
                "image": image, "metadata": tuple(int(sample[n].numpy()) for n in ("age_approx", "sex", "anatom_site_general_challenge")),
                "patient_missing": patient < 0})
    if len(records) != metrics["run"]["training_256"]["test_images"] or len(records) != len(reference):
        raise ValueError("Wrong full test cohort")
    if sum(r["label"] for r in records) != 78:
        raise ValueError("Unexpected positive count in this frozen reference")
    selected = select_test_sample(records, args.negative_count, args.seed)
    sample_path = args.output_dir / "selected-sample.json"
    sample_path.write_text(json.dumps({"full_test": len(records), "full_test_positive": sum(r["label"] for r in records),
        "sample": [{k: r[k] for k in ("image_sha256", "label", "patient_group", "patient_missing")} for r in selected],
        "shards": shards}, indent=2))
    # Selection is persisted before invoking either model on the new simulation.
    records = None
    engines = {name: FloatLesionInterpreter(args.bundle_dir / value["filename"], value) for name, value in models.items()}
    print(json.dumps({"selected_images": len(selected), "positive": sum(r["label"] for r in selected),
        "negative": args.negative_count, "conditions": len(CONDITIONS), "planned_inferences": len(selected) * len(CONDITIONS) * len(SCENARIOS)}), flush=True)
    columns = ("image_sha256", "label", "patient_group", "condition", "scenario", "score", "rgb_sha256")
    rows, max_baseline_error = [], {scenario: 0.0 for scenario in SCENARIOS}
    started = time.perf_counter()
    with (args.output_dir / "stress-predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for index, record in enumerate(selected, 1):
            rgb = tf.image.decode_jpeg(record["image"], channels=3).numpy()
            for condition, pixels in image_variants(rgb, record["image_sha256"], args.seed).items():
                pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
                for scenario in SCENARIOS:
                    score = engines["legacy"].predict(pixels) if scenario == "legacy" else engines["candidate"].predict(pixels, scenario_metadata(*record["metadata"], scenario))
                    row = {**{k: record[k] for k in ("image_sha256", "label", "patient_group")},
                        "condition": condition, "scenario": scenario, "score": score, "rgb_sha256": pixel_hash}
                    rows.append(row); writer.writerow(row)
                    if condition == "original":
                        max_baseline_error[scenario] = max(max_baseline_error[scenario], abs(score - float(reference[record["image_sha256"]][scenario])))
            if index % 20 == 0 or index == len(selected):
                f.flush()
                print(json.dumps({"images_done": index, "images_total": len(selected), "inferences_done": len(rows),
                    "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if len(rows) != len(selected) * len(CONDITIONS) * len(SCENARIOS):
        raise ValueError("Incomplete stress evaluation")
    summary = summarize(rows, policy["operating_points"], args.seed + 1)
    report = {"plan": plan, "selection_sha256": file_sha(sample_path), "shards": shards,
        "sample_count": len(selected), "positive_count": sum(r["label"] for r in selected), "negative_count": args.negative_count,
        "patient_id_missing": sum(r["patient_missing"] for r in selected), "inferences": len(rows),
        "baseline_max_absolute_error": max_baseline_error,
        "baseline_matches_previous_within_1e_5": all(e <= 1e-5 for e in max_baseline_error.values()),
        "elapsed_inference_seconds": time.perf_counter() - started, **summary,
        "limitations": ["Internal dermoscopic test reused from previous studies; not an independent clinical validation",
            "All positives and a random negative sample: prevalence differs, no population accuracy/PPV/AP reported",
            "Patient bootstrap describes this selected cohort approximately; no finite-population sampling correction",
            "Missing patient IDs use individual image groups", "Digital perturbations are not a calibrated phone-camera model",
            "Crop can remove diagnostic content while the original lesion label is retained",
            "Legacy training overlap unknown", "Thresholds remain frozen; no re-tuning on this test",
            "No partial-context decision policies or phone performance evaluation"]}
    (args.output_dir / "stress-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"completed": True, "inferences": len(rows), "baseline_error": max_baseline_error}), flush=True)


if __name__ == "__main__":
    main()
