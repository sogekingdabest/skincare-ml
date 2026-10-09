from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from capture_threshold_policy import select_ready_threshold  # noqa: E402


class CaptureThresholdPolicyTest(unittest.TestCase):
    def test_zero_false_ready_fallback_uses_smallest_recall_maximizing_threshold(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.90, 0.70, 0.79508436, 0.20]
        selected = select_ready_threshold(labels, scores, 0.01)
        self.assertEqual(selected["threshold"], 0.796)
        self.assertEqual(selected["false_positives"], 0)
        self.assertEqual(selected["recall"], 0.5)
        self.assertEqual(
            selected["selection_mode"],
            "exploratory_zero_observed_false_ready",
        )
        self.assertFalse(selected["release_target_satisfied"])
        self.assertFalse(selected["test_used_for_selection"])

    def test_rejects_invalid_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold_resolution"):
            select_ready_threshold([0, 1], [0.9, 0.1], 0.01, 0.0)


if __name__ == "__main__":
    unittest.main()
