"""Freeze operating points on validation, then compare metadata scenarios on test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from compare_lesion_models import scores
from lesion_model_contract import FloatLesionInterpreter, load_contract
from lesion_metadata_policy import (
    ENCODING_VERSION, SCENARIOS, SENSITIVITY_TARGETS,
    scenario_metadata, select_operating_point, wilson_interval,
)


def shard_paths(metrics: dict, root: Path, split: str) -> list[Path]:
    plan = metrics["run"]["split_strategy"]
    groups = [set(plan[f"{name}_shards_2020"]) for name in ("training", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Shard overlap between training, validation and test")
    names = plan[f"{split}_shards_2020"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or repeated shard list")
    result = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("Invalid shard name")
        matches = list(root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one shard {name}")
        result.append(matches[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-model", "candidate-contract", "legacy-model", "legacy-contract",
                 "metrics", "tfrecord-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    import tensorflow as tf

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    candidate_contract = load_contract(args.candidate_contract)
    legacy_contract = load_contract(args.legacy_contract)
    candidate = FloatLesionInterpreter(args.candidate_model, candidate_contract)
    legacy = FloatLesionInterpreter(args.legacy_model, legacy_contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"scenarios": SCENARIOS, "minimum_sensitivity_targets": SENSITIVITY_TARGETS,
            "encoding_version": ENCODING_VERSION, "selection_split": "validation",
            "metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
            "candidate_sha256": candidate_contract["sha256"], "legacy_sha256": legacy_contract["sha256"],
            "note": "Experimental internal operating points; no clinical or phone-image validation"}
    (args.output_dir / "experiment-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    schema = {"image": tf.io.FixedLenFeature([], tf.string), "target": tf.io.FixedLenFeature([], tf.int64),
              **{name: tf.io.FixedLenFeature([], tf.int64, default_value=-1)
                 for name in ("age_approx", "sex", "anatom_site_general_challenge")}}

    def infer_split(split: str):
        rows, seen, fields = [], set(), {name: {} for name in ("age", "sex", "site")}
        paths = shard_paths(metrics, args.tfrecord_root, split)
        for record in tf.data.TFRecordDataset([str(path) for path in paths], num_parallel_reads=1):
            sample = tf.io.parse_single_example(record, schema)
            encoded_image = bytes(sample["image"].numpy())
            identity = hashlib.sha256(encoded_image).hexdigest()
            if identity in seen:
                raise ValueError("Repeated image within split")
            seen.add(identity)
            rgb = tf.image.decode_jpeg(encoded_image, channels=3).numpy()
            age, sex, site = [int(sample[name].numpy()) for name in
                              ("age_approx", "sex", "anatom_site_general_challenge")]
            for name, value in zip(fields, (age, sex, site), strict=True):
                fields[name][str(value)] = fields[name].get(str(value), 0) + 1
            label = int(sample["target"].numpy())
            row = {"image_sha256": identity, "label": label,
                   "age_missing": age == -1, "sex_missing": sex == -1, "site_missing": site == -1}
            row["legacy"] = legacy.predict(rgb)
            row["recorded_context"] = candidate.predict(rgb, scenario_metadata(age, sex, site, "recorded_context"))
            for scenario in SCENARIOS[1:]:
                row[scenario] = candidate.predict(rgb, scenario_metadata(age, sex, site, scenario))
            rows.append(row)
            if len(rows) % 500 == 0:
                print(f"{split}: {len(rows)} images", flush=True)
        expected = metrics["run"]["training_256"][f"{split}_images"]
        if len(rows) != expected:
            raise ValueError(f"Wrong {split} count: expected {expected}, got {len(rows)}")
        with (args.output_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return rows, fields, paths

    validation, validation_fields, validation_paths = infer_split("validation")
    selected = {scenario: [select_operating_point([row["label"] for row in validation],
                         [row[scenario] for row in validation], sensitivity)
                         for sensitivity in SENSITIVITY_TARGETS]
                for scenario in ("legacy", *SCENARIOS)}
    # Persist selection before reading any test labels or evaluating new test scenarios.
    frozen_path = args.output_dir / "frozen-validation-policy.json"
    frozen_path.write_text(json.dumps({"plan": plan, "operating_points": selected}, indent=2), encoding="utf-8")
    test, test_fields, test_paths = infer_split("test")
    if {r["image_sha256"] for r in validation} & {r["image_sha256"] for r in test}:
        raise ValueError("Test identity mismatch or validation/test image overlap")

    results = {}
    for scenario, policies in selected.items():
        results[scenario] = []
        for policy in policies:
            evaluated = scores([row["label"] for row in test], [row[scenario] for row in test], policy["threshold"])
            counts = evaluated["confusion_matrix"]
            evaluated["sensitivity_wilson_95"] = wilson_interval(counts["tp"], counts["tp"] + counts["fn"])
            evaluated["specificity_wilson_95"] = wilson_interval(counts["tn"], counts["tn"] + counts["fp"])
            results[scenario].append({"selection": policy, "test": evaluated})
    report = {"plan": plan, "validation_count": len(validation), "test_count": len(test),
              "validation_metadata_counts": validation_fields, "test_metadata_counts": test_fields,
              "frozen_policy_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "shards": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in validation_paths + test_paths],
              "results": results, "candidate_released": False,
              "limitations": ["internal dermoscopic cohort; test reused from an earlier comparison", "legacy training provenance unknown",
                              "missing fields simulated, not naturally missing cohort validation",
                              "per-image Wilson intervals do not account for repeated images from a patient",
                              "threshold study is not calibration of individual clinical probabilities"]}
    (args.output_dir / "metadata-policy-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"validation": len(validation), "test": len(test), "scenarios": SCENARIOS}), flush=True)


if __name__ == "__main__":
    main()
