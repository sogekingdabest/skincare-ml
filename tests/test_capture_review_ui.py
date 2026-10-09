from __future__ import annotations

import sys
import tempfile
import unittest
import csv
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from review_capture_batch import (  # noqa: E402
    ImageResolver,
    LEGACY_REVIEW_COLUMNS,
    ReviewStore,
    summary,
)


class CaptureReviewUiTest(unittest.TestCase):
    def test_resolver_uses_source_root_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "openimages"
            image_path = images / "images" / "no_skin" / "sample.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "gray").save(image_path)
            rows = [
                self._row("safe", "images/no_skin/sample.jpg"),
                self._row("escape", "../outside.jpg"),
            ]

            resolver = ImageResolver(root, {"open_images_v7": images}, rows)

            self.assertEqual(resolver.get("safe"), image_path.resolve())
            self.assertFalse(resolver.available("escape"))

    def test_reviews_are_atomic_resumable_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review_dani.csv"
            store = ReviewStore(path, "dani", {"one", "two"})
            saved = store.save(
                "one",
                {
                    "decision": "include",
                    "review_label": "no_skin",
                    "license_review_status": "unreviewed",
                    "reviewer_notes": "valid scene",
                },
            )
            self.assertEqual(saved["include"], "true")
            self.assertEqual(saved["quality_status"], "pass")
            self.assertTrue(path.is_file())

            resumed = ReviewStore(path, "dani", {"one", "two"})
            self.assertEqual(resumed.all()["one"]["review_label"], "no_skin")
            with self.assertRaisesRegex(ValueError, "exclusion reason"):
                resumed.save("two", {"decision": "exclude"})
            with self.assertRaisesRegex(ValueError, "Unknown candidate"):
                resumed.save(
                    "unknown",
                    {"decision": "include", "review_label": "no_skin"},
                )

    def test_quality_failure_is_saved_separately_from_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review_dani.csv"
            store = ReviewStore(path, "dani", {"one"})
            review = store.save(
                "one",
                {
                    "quality_status": "fail",
                    "quality_reason": "too_far",
                    "decision": "exclude",
                    "exclusion_reason": "too_far",
                },
            )
            self.assertEqual(review["quality_status"], "fail")
            self.assertEqual(review["quality_reason"], "too_far")
            self.assertEqual(review["review_label"], "")

    def test_legacy_review_file_is_migrated_without_losing_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review_dani.csv"
            fields = sorted(LEGACY_REVIEW_COLUMNS)
            row = {field: "" for field in fields}
            row.update(
                {
                    "candidate_id": "one",
                    "reviewer_id": "dani",
                    "reviewed_at": "2026-08-03T10:00:00+00:00",
                    "decision": "exclude",
                    "include": "false",
                    "exclusion_reason": "blur",
                    "license_review_status": "unreviewed",
                }
            )
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

            migrated = ReviewStore(path, "dani", {"one"}).all()["one"]
            self.assertEqual(migrated["decision"], "exclude")
            self.assertEqual(migrated["quality_status"], "fail")
            self.assertEqual(migrated["quality_reason"], "blur")

    def test_summary_only_counts_visible_rows(self) -> None:
        rows = [self._row("one", "missing.jpg")]
        resolver = ImageResolver(Path.cwd(), {}, rows)
        reviews = {
            "one": {
                "decision": "include",
                "review_label": "no_skin",
            },
            "hidden": {
                "decision": "exclude",
                "review_label": "",
            },
        }
        report = summary(rows, reviews, resolver)
        self.assertEqual(report["reviewed"], 1)
        self.assertEqual(report["decisions"], {"include": 1})

    @staticmethod
    def _row(candidate_id: str, source_path: str) -> dict[str, str]:
        return {
            "candidate_id": candidate_id,
            "source": "open_images_v7",
            "source_path": source_path,
            "group_id": f"open_images_v7:{candidate_id}",
            "proposed_label": "no_skin",
            "proposal_reason": "fixture",
            "license": "CC BY 2.0",
            "license_status": "verify_original",
        }


if __name__ == "__main__":
    unittest.main()
