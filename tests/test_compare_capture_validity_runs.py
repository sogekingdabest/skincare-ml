from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_capture_validity_runs import compare_predictions, compare_runs  # noqa: E402


class CompareCaptureValidityRunsTest(unittest.TestCase):
    def row(self, digest: str, predicted: str, errors: str = "") -> dict[str, str]:
        return {
            "sha256": digest, "split": "test", "group_id": digest,
            "source": "fixture", "true_label": "target_lesion",
            "predicted_label": predicted, "probability_target_lesion": "0.5",
            "ready_threshold": "0.8", "error_types": errors,
        }

    def test_counts_corrections_regressions_and_ready_changes(self) -> None:
        baseline = [
            self.row("a", "skin_without_target", "argmax_error|ready_false_negative"),
            self.row("b", "target_lesion"),
        ]
        candidate = [
            self.row("a", "target_lesion"),
            self.row("b", "skin_without_target", "argmax_error|ready_false_negative"),
        ]
        _, summary = compare_predictions(baseline, candidate)
        self.assertEqual(summary["argmax_transitions"]["wrong->correct"], 1)
        self.assertEqual(summary["argmax_transitions"]["correct->wrong"], 1)
        self.assertEqual(summary["ready_changes"]["ready_false_negative_fixed"], 1)
        self.assertEqual(summary["ready_changes"]["ready_false_negative_new"], 1)

    def test_rejects_different_held_out_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "same held-out"):
            compare_predictions([self.row("a", "target_lesion")], [self.row("b", "target_lesion")])

    def test_version_name_defaults_preserve_existing_comparisons(self) -> None:
        self.assertEqual(
            compare_runs.__defaults__,
            ("v3", "v4"),
        )


if __name__ == "__main__":
    unittest.main()
