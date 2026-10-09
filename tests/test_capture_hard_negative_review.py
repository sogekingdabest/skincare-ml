from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from finalize_capture_hard_negative_review import finalize_review, validate_decisions


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CaptureHardNegativeReviewTest(unittest.TestCase):
    def _queue_row(self) -> dict[str, str]:
        return {
            "review_id": "test-0001",
            "sha256": "a" * 64,
            "group_id": "subject:1",
            "source": "fixture",
            "split": "test",
            "true_label": "skin_without_target",
            "predicted_label": "target_lesion",
            "probability_target_lesion": "0.9",
            "error_types": "argmax_error|ready_false_positive",
        }

    def _decision(self) -> dict[str, str]:
        return {
            "review_id": "test-0001",
            "decision": "keep_label",
            "reviewer": "assistant",
            "reviewer_kind": "assisted",
            "confidence": "high",
            "rationale": "non_mole_visual_finding",
        }

    def test_requires_exact_decision_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_decisions([self._queue_row()], [])

    def test_keeps_held_out_sample_out_of_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.csv"
            decisions = root / "decisions.csv"
            manifest = root / "manifest.csv"
            write_csv(queue, [self._queue_row()])
            write_csv(decisions, [self._decision()])
            write_csv(
                manifest,
                [
                    {
                        "sha256": "a" * 64,
                        "annotation_source": "assisted",
                        "annotation_reviewer": "fixture_assistant",
                    }
                ],
            )
            report = finalize_review(queue, decisions, manifest, root / "out")
            self.assertEqual(report["held_out_hard_negatives"], 1)
            self.assertEqual(report["training_samples_created"], 0)
            with (
                root / "out" / "held_out_hard_negative_references.csv"
            ).open(encoding="utf-8") as handle:
                references = list(csv.DictReader(handle))
            self.assertEqual(references[0]["training_eligible"], "false")
            self.assertEqual(references[0]["requires_human_confirmation"], "true")


if __name__ == "__main__":
    unittest.main()
