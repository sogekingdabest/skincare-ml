from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_capture_regression_review import select_new_ready_false_positives  # noqa: E402
from verify_capture_tflite_parity import build_parity_rows  # noqa: E402


class CaptureV4AuditTest(unittest.TestCase):
    def test_selects_only_new_ready_false_positives(self) -> None:
        rows = [
            {
                "sha256": "new",
                "baseline_error_types": "",
                "candidate_error_types": "argmax_error|ready_false_positive",
            },
            {
                "sha256": "existing",
                "baseline_error_types": "ready_false_positive",
                "candidate_error_types": "ready_false_positive",
            },
        ]
        selected = select_new_ready_false_positives(rows)
        self.assertEqual([row["sha256"] for row in selected], ["new"])

    def test_parity_rows_expose_argmax_without_ready_mismatch(self) -> None:
        keras = [{
            "record_index": "0", "sha256": "a" * 64,
            "group_id": "fixture:1", "source": "fixture",
            "true_label": "skin_without_target",
            "probability_target_lesion": "0.49",
            "probability_skin_without_target": "0.50",
            "probability_no_skin": "0.01",
        }]
        rows, report = build_parity_rows(keras, [[0.51, 0.48, 0.01]], 0.75)
        self.assertEqual(report["argmax_mismatches"], 1)
        self.assertEqual(report["ready_mismatches"], 0)
        self.assertEqual(rows[0]["argmax_agreement"], "false")
        self.assertEqual(rows[0]["ready_agreement"], "true")


if __name__ == "__main__":
    unittest.main()
