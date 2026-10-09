from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_openimages_no_skin import (  # noqa: E402
    CC_BY_2_0,
    HUMAN_CLASSES,
    PERSON_MID,
    SAFE_SCENE_CLASSES,
    build_candidates,
)


class OpenImagesNoSkinAdapterTest(unittest.TestCase):
    def test_requires_person_negative_and_rejects_positive_human_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_csv(
                root / "classes.csv",
                [
                    {"LabelName": mid, "DisplayName": name}
                    for mid, name in (SAFE_SCENE_CLASSES | HUMAN_CLASSES).items()
                ],
            )
            safe_mid = next(iter(SAFE_SCENE_CLASSES))
            labels = []
            for image_id in ("eligible", "has-person", "no-person-negative"):
                labels.append(
                    {
                        "ImageID": image_id,
                        "Source": "verification",
                        "LabelName": safe_mid,
                        "Confidence": "1.0",
                    }
                )
            for image_id in ("eligible", "has-person"):
                labels.append(
                    {
                        "ImageID": image_id,
                        "Source": "verification",
                        "LabelName": PERSON_MID,
                        "Confidence": "0.0",
                    }
                )
            labels.append(
                {
                    "ImageID": "has-person",
                    "Source": "crowdsource-verification",
                    "LabelName": PERSON_MID,
                    "Confidence": "1.0",
                }
            )
            self._write_csv(root / "labels.csv", labels)
            self._write_csv(
                root / "metadata.csv",
                [self._metadata(image_id) for image_id in (
                    "eligible",
                    "has-person",
                    "no-person-negative",
                )],
            )

            selected, report = build_candidates(
                root / "classes.csv",
                root / "labels.csv",
                root / "metadata.csv",
                count=10,
                seed=42,
            )

            self.assertEqual(
                [row["openimages_image_id"] for row in selected], ["eligible"]
            )
            self.assertEqual(report["eligible_rows"], 1)
            self.assertEqual(report["shortfall"], 9)
            self.assertEqual(selected[0]["proposed_label"], "no_skin")
            self.assertEqual(selected[0]["review_status"], "pending")
            self.assertEqual(
                selected[0]["license_status"],
                "metadata_cc_by_2_0_verify_original",
            )

    @staticmethod
    def _metadata(image_id: str) -> dict[str, str]:
        return {
            "ImageID": image_id,
            "Subset": "validation",
            "OriginalURL": f"https://example.test/{image_id}.jpg",
            "OriginalLandingURL": f"https://example.test/photos/{image_id}",
            "License": CC_BY_2_0,
            "AuthorProfileURL": "https://example.test/author",
            "Author": "Fixture Author",
            "Title": image_id,
            "OriginalSize": "1000",
            "OriginalMD5": "fixture",
            "Thumbnail300KURL": "",
            "Rotation": "",
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
