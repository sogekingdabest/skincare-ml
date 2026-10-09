from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_capture_exploratory_manifest import build_exploratory_manifest  # noqa: E402


class CaptureExploratoryManifestTest(unittest.TestCase):
    def test_human_labels_win_and_uncertain_assistance_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, human, assisted = self._fixture(root)
            overlap_id = rows[0]["candidate_id"]
            assisted[overlap_id] = self._review(
                overlap_id, "luna", "include", "skin_without_target"
            )
            uncertain_id = rows[-1]["candidate_id"]
            assisted[uncertain_id]["review_confidence"] = "medium"
            assisted[uncertain_id]["requires_human_review"] = "true"

            annotations, excluded, report = build_exploratory_manifest(
                rows,
                "dani",
                human,
                "luna",
                assisted,
                root,
                {"fixture": PurePosixPath("images")},
                seed=42,
            )

            by_id = {row["candidate_id"]: row for row in annotations}
            self.assertEqual(by_id[overlap_id]["annotation_source"], "human")
            self.assertEqual(by_id[overlap_id]["label"], "target_lesion")
            self.assertNotIn(uncertain_id, by_id)
            self.assertEqual(report["human_assisted_overlap"], 1)
            self.assertTrue(
                any(
                    row["candidate_id"] == uncertain_id
                    and row["exclusion_reason"] == "assisted_requires_human_review"
                    for row in excluded
                )
            )

    def test_splits_are_deterministic_grouped_and_contain_every_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, human, assisted = self._fixture(root)
            first, _, _ = build_exploratory_manifest(
                rows,
                "dani",
                human,
                "luna",
                assisted,
                root,
                {"fixture": PurePosixPath("images")},
                seed=42,
            )
            second, _, _ = build_exploratory_manifest(
                rows,
                "dani",
                human,
                "luna",
                assisted,
                root,
                {"fixture": PurePosixPath("images")},
                seed=42,
            )
            self.assertEqual(
                [(row["candidate_id"], row["split"]) for row in first],
                [(row["candidate_id"], row["split"]) for row in second],
            )
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    {row["label"] for row in first if row["split"] == split},
                    {"target_lesion", "skin_without_target", "no_skin"},
                )
            group_splits: dict[str, set[str]] = {}
            for row in first:
                group_splits.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(all(len(values) == 1 for values in group_splits.values()))

    def test_missing_review_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, human, assisted = self._fixture(root)
            assisted.pop(rows[-1]["candidate_id"])
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                build_exploratory_manifest(
                    rows,
                    "dani",
                    human,
                    "luna",
                    assisted,
                    root,
                    {"fixture": PurePosixPath("images")},
                    seed=42,
                )

    @classmethod
    def _fixture(cls, root: Path):
        images = root / "images"
        images.mkdir()
        rows = []
        human = {}
        assisted = {}
        labels = ("target_lesion", "skin_without_target", "no_skin")
        for label_index, label in enumerate(labels):
            for index in range(10):
                candidate_id = f"{label}-{index}"
                filename = f"{candidate_id}.png"
                Image.new(
                    "RGB",
                    (16, 16),
                    (label_index * 70, index * 10, 120),
                ).save(images / filename)
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "source": "fixture",
                        "source_dataset": "fixture",
                        "source_path": filename,
                        "group_id": f"group-{candidate_id}",
                        "subject_id": f"subject-{candidate_id}",
                        "session_id": "",
                        "proposed_label": label,
                        "license": "fixture",
                        "license_status": "unreviewed",
                    }
                )
                review = cls._review(candidate_id, "luna", "include", label)
                if label != "no_skin" and index == 0:
                    review = cls._review(candidate_id, "dani", "include", label)
                    human[candidate_id] = review
                else:
                    assisted[candidate_id] = review
        return rows, human, assisted

    @staticmethod
    def _review(candidate_id: str, reviewer: str, decision: str, label: str):
        return {
            "candidate_id": candidate_id,
            "reviewer_id": reviewer,
            "reviewed_at": "2026-08-04T10:00:00+00:00",
            "decision": decision,
            "review_label": label if decision == "include" else "",
            "include": str(decision == "include").lower(),
            "exclusion_reason": "quality" if decision == "exclude" else "",
            "reviewer_notes": "",
            "license_review_status": "unreviewed",
            "license_review_notes": "",
            "review_confidence": "high",
            "requires_human_review": "false",
        }


if __name__ == "__main__":
    unittest.main()
