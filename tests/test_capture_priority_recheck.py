from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from review_capture_priority_recheck import (  # noqa: E402
    PriorityReviewStore,
    load_priority_rows,
    review_summary,
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CapturePriorityRecheckTest(unittest.TestCase):
    def test_loads_only_training_policy_rows_inside_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.jpg"
            Image.new("RGB", (16, 16), "brown").save(image)
            candidates = root / "candidates.csv"
            priority = root / "priority.csv"
            write_csv(
                candidates,
                [{
                    "review_id": "train-0001", "split": "train",
                    "true_label": "skin_without_target",
                    "predicted_label": "target_lesion", "image_path": str(image),
                    "probability_target_lesion": "0.9", "source": "fixture",
                    "original_annotation_source": "human",
                    "original_annotation_reviewer": "dani",
                }],
            )
            write_csv(priority, [{"review_id": "train-0001", "rationale": "target_like"}])
            rows = load_priority_rows(candidates, priority, root)
            self.assertEqual([row["review_id"] for row in rows], ["train-0001"])
            self.assertEqual(rows[0]["assisted_rationale"], "target_like")

    def test_rejects_non_training_priority_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.jpg"
            Image.new("RGB", (16, 16), "brown").save(image)
            candidates = root / "candidates.csv"
            priority = root / "priority.csv"
            write_csv(candidates, [{
                "review_id": "validation-0001", "split": "validation",
                "true_label": "skin_without_target", "predicted_label": "target_lesion",
                "image_path": str(image),
            }])
            write_csv(priority, [{"review_id": "validation-0001"}])
            with self.assertRaisesRegex(ValueError, "training recheck policy"):
                load_priority_rows(candidates, priority, root)

    def test_store_writes_human_decisions_atomically_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "decisions.csv"
            store = PriorityReviewStore(output, "dani", {"train-0001"})
            saved = store.save("train-0001", {
                "decision": "correct_to_target", "confidence": "high", "rationale": "centred mole",
            })
            self.assertEqual(saved["reviewer_kind"], "human")
            resumed = PriorityReviewStore(output, "dani", {"train-0001"})
            self.assertEqual(resumed.all()["train-0001"]["decision"], "correct_to_target")
            self.assertTrue(review_summary([{"review_id": "train-0001"}], resumed.all())["complete"])

    def test_store_rejects_unknown_or_unsupported_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PriorityReviewStore(Path(temporary) / "decisions.csv", "dani", {"train-0001"})
            with self.assertRaisesRegex(ValueError, "Unknown review_id"):
                store.save("train-9999", {"decision": "keep_label", "confidence": "high"})
            with self.assertRaisesRegex(ValueError, "decision must"):
                store.save("train-0001", {"decision": "delete", "confidence": "high"})


if __name__ == "__main__":
    unittest.main()
