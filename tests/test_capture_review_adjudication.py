from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from adjudicate_capture_reviews import (  # noqa: E402
    adjudicate_candidate,
    consolidate,
    load_review,
)


class CaptureReviewAdjudicationTest(unittest.TestCase):
    def test_unanimous_verified_inclusion_is_training_ready(self) -> None:
        row = self._batch_row()
        reviews = [
            self._review("alice", "include", "target_lesion", "verified"),
            self._review("bob", "include", "target_lesion", "verified"),
        ]
        result = adjudicate_candidate(row, reviews, min_reviews=2)
        self.assertEqual(result["visual_status"], "approved")
        self.assertEqual(result["license_gate"], "verified")
        self.assertEqual(result["training_ready"], "true")

    def test_visual_consensus_does_not_bypass_pending_licence(self) -> None:
        row = self._batch_row()
        reviews = [
            self._review("alice", "include", "target_lesion", "verified"),
            self._review("bob", "include", "target_lesion", "unreviewed"),
        ]
        result = adjudicate_candidate(row, reviews, min_reviews=2)
        self.assertEqual(result["visual_status"], "approved")
        self.assertEqual(result["license_gate"], "pending")
        self.assertEqual(result["training_ready"], "false")

    def test_disagreement_requires_adjudication(self) -> None:
        row = self._batch_row()
        reviews = [
            self._review("alice", "include", "target_lesion", "verified"),
            self._review("bob", "include", "skin_without_target", "verified"),
        ]
        result = adjudicate_candidate(row, reviews, min_reviews=2)
        self.assertEqual(result["visual_status"], "needs_adjudication")
        self.assertEqual(result["final_label"], "")
        self.assertEqual(result["training_ready"], "false")

    def test_single_review_stays_pending_by_default_policy(self) -> None:
        result = adjudicate_candidate(
            self._batch_row(),
            [self._review("alice", "include", "target_lesion", "verified")],
            min_reviews=2,
        )
        self.assertEqual(result["visual_status"], "pending_review")

    def test_duplicate_reviewer_files_are_rejected(self) -> None:
        review = self._review("alice", "include", "target_lesion", "verified")
        with self.assertRaisesRegex(ValueError, "distinct reviewer"):
            consolidate(
                [self._batch_row()],
                [("alice", {"one": review}), ("alice", {"one": review})],
                min_reviews=2,
            )

    def test_review_loader_rejects_unknown_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            row = self._review("alice", "include", "target_lesion", "verified")
            row["candidate_id"] = "unknown"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "unknown candidate"):
                load_review(path, {"one"})

    @staticmethod
    def _batch_row() -> dict[str, str]:
        return {
            "candidate_id": "one",
            "source": "pad_ufes_20",
            "source_dataset": "fixture",
            "source_path": "images/one.jpg",
            "group_id": "patient:one",
            "subject_id": "one",
            "session_id": "visit-one",
            "proposed_label": "target_lesion",
            "license": "CC BY 4.0",
            "license_status": "verify_before_release",
        }

    @staticmethod
    def _review(
        reviewer: str, decision: str, label: str, licence: str
    ) -> dict[str, str]:
        return {
            "candidate_id": "one",
            "reviewer_id": reviewer,
            "reviewed_at": "2026-08-03T10:00:00+00:00",
            "decision": decision,
            "review_label": label if decision == "include" else "",
            "include": str(decision == "include").lower(),
            "exclusion_reason": "not usable" if decision == "exclude" else "",
            "reviewer_notes": "",
            "license_review_status": licence,
            "license_review_notes": "",
        }


if __name__ == "__main__":
    unittest.main()
