from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mine_capture_training_hard_negatives import select_candidates


class CaptureHardNegativeMiningTest(unittest.TestCase):
    def _prediction(self, split: str = "train") -> dict[str, str]:
        return {
            "split": split,
            "record_index": "7",
            "sha256": "a" * 64,
            "group_id": "subject:1",
            "source": "fixture",
            "true_label": "skin_without_target",
            "predicted_label": "target_lesion",
            "probability_target_lesion": "0.9",
            "probability_skin_without_target": "0.08",
            "probability_no_skin": "0.02",
            "predicted_ready": "true",
            "error_types": "argmax_error|ready_false_positive",
        }

    def test_selects_only_training_errors_and_requires_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "sample.jpg"
            image_path.parent.mkdir()
            Image.new("RGB", (20, 20), "pink").save(image_path)
            manifest = [
                {
                    "sha256": "a" * 64,
                    "group_id": "subject:1",
                    "split": "train",
                    "image_path": "images/sample.jpg",
                    "annotation_source": "assisted",
                    "annotation_reviewer": "assistant",
                }
            ]
            candidates = select_candidates(
                [self._prediction(), self._prediction("validation")],
                manifest,
                root,
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["selection_reason"], "ready_false_positive")
            self.assertEqual(candidates[0]["approved_for_reweighting"], "false")
            self.assertEqual(candidates[0]["requires_human_confirmation"], "true")

    def test_rejects_overlap_with_held_out_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "sample.jpg"
            Image.new("RGB", (20, 20), "pink").save(image_path)
            manifest = [
                {
                    "sha256": "a" * 64,
                    "split": "train",
                    "image_path": "sample.jpg",
                    "annotation_source": "human",
                }
            ]
            with self.assertRaisesRegex(ValueError, "held-out group"):
                select_candidates(
                    [self._prediction()],
                    manifest,
                    root,
                    held_out_references=[{"group_id": "subject:1"}],
                )


if __name__ == "__main__":
    unittest.main()
