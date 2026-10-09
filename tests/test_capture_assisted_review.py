from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assist_capture_review import finalize, prepare  # noqa: E402


class CaptureAssistedReviewTest(unittest.TestCase):
    def test_prepare_skips_existing_reviews_and_creates_sheets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name in ("one.jpg", "two.jpg", "three.jpg"):
                Image.new("RGB", (80, 60), "#805040").save(source / name)
            batch = root / "batch.csv"
            self._write_batch(batch, ("one", "two", "three"))
            skip = root / "review_dani.csv"
            self._write_csv(skip, [{"candidate_id": "two"}], ["candidate_id"])
            output = root / "pack"

            report = prepare(
                argparse.Namespace(
                    batch=batch,
                    output_dir=output,
                    source_root=[f"fixture={source}"],
                    skip_review=[skip],
                    label=None,
                    sheet_size=2,
                    columns=2,
                )
            )

            self.assertEqual(report["queued"], 2)
            self.assertEqual(report["skipped_existing_reviews"], 1)
            self.assertTrue((output / "sheets" / "sheet_001.jpg").is_file())
            rows, _ = self._read_csv(output / "assisted_review_queue.csv")
            self.assertEqual([row["candidate_id"] for row in rows], ["one", "three"])
            self.assertTrue(all(row["image_available"] == "true" for row in rows))

    def test_finalize_maps_decisions_without_licence_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch.csv"
            self._write_batch(batch, ("one", "two"))
            queue = root / "queue.csv"
            rows = [
                self._queue_row("one", "pass", "", "target_lesion", "", "high"),
                self._queue_row("two", "fail", "blur", "", "", "low"),
            ]
            self._write_csv(queue, rows, list(rows[0]))
            output = root / "review_codex.csv"

            report = finalize(
                argparse.Namespace(
                    batch=batch,
                    queue=queue,
                    reviewer="codex",
                    output=output,
                    allow_partial=False,
                )
            )

            reviews, _ = self._read_csv(output)
            by_id = {row["candidate_id"]: row for row in reviews}
            self.assertEqual(by_id["one"]["decision"], "include")
            self.assertEqual(by_id["two"]["decision"], "exclude")
            self.assertEqual(by_id["two"]["exclusion_reason"], "blur")
            self.assertEqual(by_id["one"]["license_review_status"], "unreviewed")
            self.assertEqual(by_id["two"]["requires_human_review"], "true")
            self.assertEqual(report["completed"], 2)

    def test_medium_confidence_requires_human_review(self) -> None:
        row = self._queue_row(
            "one", "pass", "", "skin_without_target", "", "medium"
        )
        from assist_capture_review import normalize_queue_row

        review = normalize_queue_row(row, "codex")
        self.assertIsNotNone(review)
        self.assertEqual(review["requires_human_review"], "true")

    def test_finalize_rejects_incomplete_queue_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch.csv"
            self._write_batch(batch, ("one", "two"))
            queue = root / "queue.csv"
            completed = self._queue_row(
                "one", "pass", "", "skin_without_target", "", "medium"
            )
            empty = self._queue_row("two", "", "", "", "", "")
            self._write_csv(queue, [completed, empty], list(completed))
            with self.assertRaisesRegex(ValueError, "Queue is incomplete"):
                finalize(
                    argparse.Namespace(
                        batch=batch,
                        queue=queue,
                        reviewer="codex",
                        output=root / "review.csv",
                        allow_partial=False,
                    )
                )

    @staticmethod
    def _queue_row(
        candidate_id: str,
        quality: str,
        quality_reason: str,
        label: str,
        exclusion: str,
        confidence: str,
    ) -> dict[str, str]:
        return {
            "sequence": "1",
            "sheet": "sheet_001.jpg",
            "position": "01",
            "candidate_id": candidate_id,
            "source": "fixture",
            "source_path": f"{candidate_id}.jpg",
            "proposed_label": "target_lesion",
            "image_available": "true",
            "quality_status": quality,
            "quality_reason": quality_reason,
            "review_label": label,
            "exclusion_reason": exclusion,
            "review_confidence": confidence,
            "reviewer_notes": "",
            "reviewed_at": "2026-08-03T12:00:00+00:00" if quality else "",
        }

    @classmethod
    def _write_batch(cls, path: Path, identifiers: tuple[str, ...]) -> None:
        rows = [
            {
                "candidate_id": identifier,
                "source": "fixture",
                "source_path": f"{identifier}.jpg",
                "group_id": f"patient:{identifier}",
                "proposed_label": "target_lesion",
                "proposal_reason": "fixture",
                "license": "fixture",
                "license_status": "verify",
            }
            for identifier in identifiers
        ]
        cls._write_csv(path, rows, list(rows[0]))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or [])


if __name__ == "__main__":
    unittest.main()
