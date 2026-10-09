from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finalize_capture_training_hard_negatives import finalize  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TrainingHardNegativeFinalizationTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        candidates = root / "candidates.csv"
        priority = root / "priority.csv"
        decisions = root / "decisions.csv"
        manifest = root / "manifest.csv"
        base = {
            "split": "train", "true_label": "skin_without_target",
            "predicted_label": "target_lesion", "source": "fixture",
            "original_annotation_reviewer": "dani",
        }
        write_csv(candidates, [
            {**base, "review_id": "train-1", "sha256": "a"*64, "group_id": "g1", "original_annotation_source": "human"},
            {**base, "review_id": "train-2", "sha256": "b"*64, "group_id": "g2", "original_annotation_source": "assisted"},
            {**base, "review_id": "train-3", "sha256": "c"*64, "group_id": "g3", "original_annotation_source": "assisted"},
            {**base, "review_id": "train-4", "sha256": "d"*64, "group_id": "g4", "original_annotation_source": "assisted"},
        ])
        write_csv(priority, [{"review_id": value} for value in ("train-1", "train-2", "train-3")])
        common = {"reviewer": "dani", "reviewer_kind": "human", "confidence": "medium", "rationale": "reviewed"}
        write_csv(decisions, [
            {"review_id": "train-1", "decision": "keep_label", **common},
            {"review_id": "train-2", "decision": "correct_to_target", **common},
            {"review_id": "train-3", "decision": "ambiguous", **common},
        ])
        write_csv(manifest, [
            {"sha256": "a"*64, "group_id": "g1", "split": "train", "label": "skin_without_target", "annotation_source": "human", "annotation_reviewer": "dani"},
            {"sha256": "b"*64, "group_id": "g2", "split": "train", "label": "skin_without_target", "annotation_source": "assisted", "annotation_reviewer": "assistant"},
            {"sha256": "c"*64, "group_id": "g3", "split": "train", "label": "skin_without_target", "annotation_source": "assisted", "annotation_reviewer": "assistant"},
            {"sha256": "d"*64, "group_id": "g4", "split": "train", "label": "skin_without_target", "annotation_source": "assisted", "annotation_reviewer": "assistant"},
            {"sha256": "e"*64, "group_id": "g5", "split": "validation", "label": "target_lesion", "annotation_source": "human", "annotation_reviewer": "dani"},
        ])
        return candidates, priority, decisions, manifest

    def test_applies_conservative_policy_and_preserves_held_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            report = finalize(*paths, root / "out")
            self.assertEqual(report["label_corrections"], 1)
            self.assertEqual(report["excluded_ambiguous"], 1)
            self.assertTrue(report["held_out_rows_unchanged"])
            with (root / "out" / "training_policy.csv").open(newline="", encoding="utf-8") as handle:
                policy = {row["review_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(policy["train-1"]["sample_weight"], "2.0")
            self.assertEqual(policy["train-2"]["effective_label"], "target_lesion")
            self.assertEqual(policy["train-3"]["training_action"], "exclude_ambiguous")
            self.assertEqual(policy["train-4"]["sample_weight"], "1.25")
            self.assertEqual(policy["train-4"]["requires_human_confirmation"], "true")

    def test_requires_exact_human_priority_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, priority, decisions, manifest = self.fixture(root)
            with decisions.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[:-1]
            write_csv(decisions, rows)
            with self.assertRaisesRegex(ValueError, "coverage"):
                finalize(candidates, priority, decisions, manifest, root / "out")


if __name__ == "__main__":
    unittest.main()
