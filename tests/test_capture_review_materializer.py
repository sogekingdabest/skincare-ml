from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_capture_review_sources import extract_archive  # noqa: E402
from materialize_capture_review_sources import package_selection  # noqa: E402
from stage_capture_review_materializer_kernel import read_selection  # noqa: E402


class CaptureReviewMaterializerTest(unittest.TestCase):
    def test_packages_and_verifies_exact_batch_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pad = root / "pad"
            scin = root / "scin"
            pad_image = pad / "imgs_part_1" / "imgs_part_1" / "PAT_1.png"
            scin_image = scin / "dataset" / "images" / "case.png"
            pad_image.parent.mkdir(parents=True)
            scin_image.parent.mkdir(parents=True)
            Image.new("RGB", (40, 30), "brown").save(pad_image)
            Image.new("RGB", (50, 35), "pink").save(scin_image)
            selection = [
                self._row(
                    "pad-one",
                    "pad_ufes_20",
                    "imgs_part_1/imgs_part_1/PAT_1.png",
                ),
                self._row("scin-one", "scin", "dataset/images/case.png"),
            ]
            archive = root / "review.zip"
            report = package_selection(
                selection,
                {"pad_ufes_20": pad, "scin": scin},
                archive,
            )
            batch = root / "batch.csv"
            self._write_csv(batch, selection)
            extracted = root / "extracted"

            verified = extract_archive(archive, extracted, batch)

            self.assertEqual(report["rows"], 2)
            self.assertEqual(verified["sources"], {"pad_ufes_20": 1, "scin": 1})
            self.assertTrue(
                (extracted / "pad_ufes_20" / selection[0]["source_path"]).is_file()
            )
            self.assertTrue(
                (extracted / "scin" / selection[1]["source_path"]).is_file()
            )

    def test_stage_selection_ignores_openimages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / "batch.csv"
            rows = [
                self._row("pad", "pad_ufes_20", "pad.png"),
                self._row("scin", "scin", "scin.png"),
                self._row("open", "open_images_v7", "open.jpg"),
            ]
            self._write_csv(batch, rows)
            selected = read_selection(batch)
            self.assertEqual({row["candidate_id"] for row in selected}, {"pad", "scin"})

    @staticmethod
    def _row(candidate_id: str, source: str, source_path: str) -> dict[str, str]:
        return {
            "candidate_id": candidate_id,
            "source": source,
            "source_path": source_path,
            "group_id": f"{source}:{candidate_id}",
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
