from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_capture_error_review import build_review_queue, render_contact_sheet
from export_capture_predictions import classify_error_types, prediction_row


class CapturePredictionTest(unittest.TestCase):
    def test_classifies_argmax_and_ready_errors_separately(self) -> None:
        self.assertEqual(
            classify_error_types(1, 0, 0.9, 0.847),
            "argmax_error|ready_false_positive",
        )
        self.assertEqual(
            classify_error_types(0, 0, 0.8, 0.847),
            "ready_false_negative",
        )

    def test_prediction_row_preserves_traceability(self) -> None:
        row = prediction_row(
            split="test",
            record_index=3,
            metadata={
                "label_id": 1,
                "sha256": "a" * 64,
                "group_id": "subject:1",
                "source": "fixture",
            },
            probabilities=(0.9, 0.08, 0.02),
            ready_threshold=0.847,
        )
        self.assertEqual(row["predicted_label"], "target_lesion")
        self.assertEqual(row["error_types"], "argmax_error|ready_false_positive")
        self.assertEqual(row["sha256"], "a" * 64)


class CaptureErrorReviewTest(unittest.TestCase):
    def test_joins_only_errors_and_renders_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "sample.jpg"
            image_path.parent.mkdir()
            Image.new("RGB", (40, 30), "#654321").save(image_path)
            predictions = [
                {
                    "split": "test",
                    "record_index": "0",
                    "sha256": "a" * 64,
                    "source": "fixture",
                    "group_id": "fixture:1",
                    "true_label": "skin_without_target",
                    "predicted_label": "target_lesion",
                    "probability_target_lesion": "0.9",
                    "probability_skin_without_target": "0.08",
                    "probability_no_skin": "0.02",
                    "predicted_ready": "true",
                    "error_types": "argmax_error|ready_false_positive",
                },
                {"error_types": ""},
            ]
            manifest = [{"sha256": "a" * 64, "image_path": "images/sample.jpg"}]
            queue = build_review_queue(predictions, manifest, root)
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["review_id"], "test-0000")
            sheet = root / "sheet.jpg"
            render_contact_sheet(queue, sheet)
            self.assertTrue(sheet.is_file())


if __name__ == "__main__":
    unittest.main()
