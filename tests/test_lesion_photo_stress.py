import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_lesion_photo_stress import CONDITIONS, SCENARIOS, image_variants, select_test_sample, validate_policy, paired_counts, summarize
from stage_lesion_photo_stress_kernel import stage


class LesionPhotoStressTest(unittest.TestCase):
    def test_transformations_preserve_original_and_are_keyed_to_image_not_iteration(self):
        original = np.arange(32 * 32 * 3).reshape(32, 32, 3).astype(np.uint8)
        saved = original.copy()
        first = image_variants(original, "a" * 64, 42)
        second = image_variants(original, "a" * 64, 42)
        other = image_variants(original, "b" * 64, 42)
        self.assertEqual(tuple(first), CONDITIONS)
        self.assertTrue(np.array_equal(original, saved))
        self.assertTrue(all(np.array_equal(first[k], second[k]) for k in CONDITIONS))
        self.assertFalse(np.array_equal(first["noise_12"], other["noise_12"]))
        self.assertTrue(all(p.shape == original.shape and p.dtype == np.uint8 for p in first.values()))
        self.assertTrue(np.array_equal(first["rotate_180"], original[::-1, ::-1]))

    def test_invalid_images_and_identities_are_not_silently_transformed(self):
        for pixels, identity in ((np.zeros((32, 32)), "a" * 64), (np.zeros((32, 32, 3), dtype=float), "a" * 64),
                (np.zeros((8, 8, 3), np.uint8), "a" * 64), (np.zeros((32, 32, 3), np.uint8), "bad")):
            with self.assertRaises(ValueError):
                image_variants(pixels, identity, 42)

    def test_sampling_keeps_all_positives_and_ignores_model_scores_and_input_order(self):
        records = [{"image_sha256": f"{i:064x}", "label": int(i < 3), "score": i / 10} for i in range(10)]
        first = select_test_sample(records, 4, 42)
        changed = [{**r, "score": 1 - r["score"]} for r in reversed(records)]
        second = select_test_sample(changed, 4, 42)
        self.assertEqual([r["image_sha256"] for r in first], [r["image_sha256"] for r in second])
        self.assertEqual(3, sum(r["label"] for r in first))
        self.assertEqual(7, len(first))
        with self.assertRaises(ValueError):
            select_test_sample(records + [records[0]], 4, 42)
        with self.assertRaises(ValueError):
            select_test_sample(records, 20, 42)

    def test_policies_reject_test_selected_thresholds_and_different_artifacts(self):
        models = {name: {"sha256": name} for name in ("legacy", "candidate")}
        points = [{"threshold": 0.5, "selected_on": "validation", "minimum_sensitivity": target} for target in (0.85, 0.9, 0.95)]
        policy = {"plan": {"selection_split": "validation", "metrics_sha256": "metrics", "legacy_sha256": "legacy", "candidate_sha256": "candidate"},
                  "operating_points": {s: [dict(p) for p in points] for s in SCENARIOS}}
        validate_policy(policy, models, "metrics")
        policy["operating_points"]["legacy"][0]["selected_on"] = "test"
        with self.assertRaises(ValueError):
            validate_policy(policy, models, "metrics")
        policy["operating_points"]["legacy"][0]["selected_on"] = "validation"
        models["candidate"]["sha256"] = "other"
        with self.assertRaises(ValueError):
            validate_policy(policy, models, "metrics")

    def test_paired_transitions_distinguish_new_misses_from_recovered_cases(self):
        counts = paired_counts([1, 1, 0, 0], [0.8, 0.2, 0.2, 0.8], [0.2, 0.8, 0.8, 0.2], 0.5)
        self.assertEqual({"tp": 1, "fn": 1, "tn": 1, "fp": 1, "positive_detected_to_missed": 1,
            "positive_missed_to_detected": 1, "negative_correct_to_false_alarm": 1, "negative_false_alarm_to_correct": 1}, counts)
        with self.assertRaises(ValueError):
            paired_counts([0, 1], [0.1, float("nan")], [0.2, 0.8], 0.5)

    def test_patient_bootstrap_uses_paired_samples_and_preserves_label_identity(self):
        rows = []
        for condition in ("original", "blur_1"):
            for scenario in SCENARIOS:
                for i in range(4):
                    rows.append({"image_sha256": str(i), "patient_group": str(i // 2), "label": int(i < 2),
                        "scenario": scenario, "condition": condition, "score": (0.8 if i < 2 else 0.2) if condition == "original" else 0.2})
        policies = {s: [{"threshold": 0.5, "minimum_sensitivity": 0.85}] for s in SCENARIOS}
        result = summarize(rows, policies, 42, 100)
        self.assertEqual(2, result["patient_groups"])
        perturbed = result["results"]["blur_1"]["legacy"]["operating_points"][0]
        self.assertEqual(-1, perturbed["delta_sensitivity"])
        self.assertEqual(2, perturbed["counts"]["positive_detected_to_missed"])
        self.assertEqual([-1, -1], perturbed["patient_bootstrap_delta_sensitivity"]["interval_95"])
        self.assertEqual(1, perturbed["specificity"])
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], policies, 42, 100)
        rows[-1]["label"] = 1
        with self.assertRaises(ValueError):
            summarize(rows, policies, 42, 100)

    def test_staging_uploads_only_readable_code_and_private_offline_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            result = stage(Path(directory), "a" * 64, "b" * 64, "c" * 64, 512, "image-digest")
            files = sorted(p.name for p in Path(directory).iterdir())
            self.assertEqual(["evaluate_on_kaggle.py", "kernel-metadata.json"], files)
            metadata = json.loads((Path(directory) / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])
            self.assertFalse(result["photos_or_models_uploaded"])
            code = (Path(directory) / "evaluate_on_kaggle.py").read_bytes()
            self.assertNotIn(b"\r\n", code)
            self.assertEqual(hashlib.sha256(code).hexdigest(), result["generated_sha256"])
            compile((Path(directory) / "evaluate_on_kaggle.py").read_text(), "staged", "exec")
            self.assertIn("metadata-policy/1", metadata["kernel_sources"][0])
