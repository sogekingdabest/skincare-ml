from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_capture_tfrecords import assign_grouped_splits, tensorflow_compression_type
from sample_capture_review_batch import build_batch
from audit_capture_sources import (
    audit_indoor,
    audit_pad,
    audit_scin,
    build_report,
    discover_dataset_root,
)


PREPARE = ROOT / "scripts" / "prepare_capture_dataset.py"
VALIDATE = ROOT / "scripts" / "validate_capture_manifest.py"
LABELS = ("target_lesion", "skin_without_target", "no_skin")


class CaptureDatasetPipelineTest(unittest.TestCase):
    def test_tensorflow_compression_contract(self) -> None:
        self.assertEqual(tensorflow_compression_type("NONE"), "")
        self.assertEqual(tensorflow_compression_type("GZIP"), "GZIP")
        with self.assertRaises(ValueError):
            tensorflow_compression_type("ZIP")

    def test_review_batch_is_grouped_deterministic_and_excludes_blocked_sources(
        self,
    ) -> None:
        rows = []
        for label, source, license_status in (
            ("target_lesion", "pad_ufes_20", "verify_before_release"),
            (
                "skin_without_target",
                "scin",
                "verify_before_use_and_weight_release",
            ),
            ("no_skin", "mit_indoor", "blocked_pending_rights_review"),
        ):
            for index in range(3):
                rows.append(
                    self._queue_row(
                        candidate_id=f"{label}-{index}",
                        label=label,
                        source=source,
                        group_id=f"{source}-group-{index}",
                        license_status=license_status,
                    )
                )
        for index in range(3):
            row = self._queue_row(
                candidate_id=f"quality-{index}",
                label="",
                source="scin",
                group_id=f"quality-group-{index}",
                license_status="verify_before_use_and_weight_release",
            )
            row["eligibility"] = "exclude_quality"
            row["shot_type"] = "AT_DISTANCE"
            rows.append(row)

        requested = {
            "target_lesion": 2,
            "skin_without_target": 2,
            "no_skin": 2,
        }
        first_semantic, first_quality, first_report = build_batch(
            rows, requested, quality_requested=2, seed=42, review_round="test"
        )
        second_semantic, second_quality, _ = build_batch(
            rows, requested, quality_requested=2, seed=42, review_round="test"
        )

        self.assertEqual(
            [row["candidate_id"] for row in first_semantic],
            [row["candidate_id"] for row in second_semantic],
        )
        self.assertEqual(
            [row["candidate_id"] for row in first_quality],
            [row["candidate_id"] for row in second_quality],
        )
        self.assertEqual(len({row["group_id"] for row in first_semantic}), 4)
        self.assertEqual(first_report["semantic"]["target_lesion"]["rows"], 2)
        self.assertEqual(
            first_report["semantic"]["skin_without_target"]["rows"], 2
        )
        self.assertEqual(first_report["semantic"]["no_skin"]["rows"], 0)
        self.assertEqual(first_report["semantic"]["no_skin"]["shortfall"], 2)
        self.assertEqual(first_report["quality_gate"]["rows"], 2)
        self.assertEqual(first_report["approved_training_samples"], 0)

    @staticmethod
    def _queue_row(
        candidate_id: str,
        label: str,
        source: str,
        group_id: str,
        license_status: str,
    ) -> dict[str, str]:
        return {
            "candidate_id": candidate_id,
            "source": source,
            "source_path": f"images/{candidate_id}.jpg",
            "group_id": group_id,
            "proposed_label": label,
            "eligibility": "candidate",
            "review_status": "pending",
            "license_status": license_status,
            "file_exists": "true",
            "condition_labels": "fixture",
            "body_site": "",
            "skin_tone": "",
            "shot_type": "CLOSE_UP",
            "notes": "",
        }

    def test_source_audit_proposes_labels_but_approves_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pad = root / "pad"
            scin = root / "scin"
            indoor = root / "indoor"
            for path in (pad, scin, indoor):
                path.mkdir()

            Image.new("RGB", (100, 80), "brown").save(pad / "PAT_1_2_3.png")
            self._write_csv(
                pad / "metadata.csv",
                [
                    {
                        "patient_id": "PAT_1",
                        "lesion_id": "2",
                        "img_id": "PAT_1_2_3.png",
                        "diagnostic": "NEV",
                        "region": "ARM",
                    }
                ],
            )

            images = scin / "dataset" / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (120, 90), "tan").save(images / "mole.png")
            Image.new("RGB", (120, 90), "pink").save(images / "rash.png")
            self._write_csv(
                scin / "scin_cases.csv",
                [
                    {
                        "case_id": "case-1",
                        "fitzpatrick_skin_type": "FST3",
                        "image_1_path": "dataset/images/mole.png",
                        "image_1_shot_type": "CLOSE_UP",
                        "image_2_path": "",
                        "image_2_shot_type": "",
                        "image_3_path": "",
                        "image_3_shot_type": "",
                    },
                    {
                        "case_id": "case-2",
                        "fitzpatrick_skin_type": "FST4",
                        "image_1_path": "dataset/images/rash.png",
                        "image_1_shot_type": "AT_DISTANCE",
                        "image_2_path": "",
                        "image_2_shot_type": "",
                        "image_3_path": "",
                        "image_3_shot_type": "",
                    },
                ],
            )
            self._write_csv(
                scin / "scin_labels.csv",
                [
                    {
                        "case_id": "case-1",
                        "dermatologist_skin_condition_on_label_name": "['Melanocytic Nevus']",
                        "dermatologist_gradable_for_skin_condition_1": "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT",
                        "dermatologist_gradable_for_skin_condition_2": "",
                        "dermatologist_gradable_for_skin_condition_3": "",
                    },
                    {
                        "case_id": "case-2",
                        "dermatologist_skin_condition_on_label_name": "['Eczema']",
                        "dermatologist_gradable_for_skin_condition_1": "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT",
                        "dermatologist_gradable_for_skin_condition_2": "",
                        "dermatologist_gradable_for_skin_condition_3": "",
                    },
                ],
            )

            for category, colour in (("kitchen", "gray"), ("office", "blue")):
                category_root = indoor / "Images" / category
                category_root.mkdir(parents=True)
                for index in range(2):
                    Image.new("RGB", (96, 72), colour).save(
                        category_root / f"{category}_{index}.jpg"
                    )

            pad_rows, pad_report = audit_pad(pad, hash_images=True)
            scin_rows, scin_report = audit_scin(scin, hash_images=True)
            indoor_rows, indoor_report = audit_indoor(
                indoor, hash_images=True, per_category=1, seed=42
            )
            rows = pad_rows + scin_rows + indoor_rows
            report = build_report(
                rows, [pad_report, scin_report, indoor_report], hash_images=True
            )

            self.assertEqual(pad_rows[0]["proposed_label"], "target_lesion")
            self.assertEqual(scin_rows[0]["proposed_label"], "target_lesion")
            self.assertEqual(scin_rows[1]["eligibility"], "exclude_quality")
            self.assertEqual({row["proposed_label"] for row in indoor_rows}, {"no_skin"})
            self.assertEqual(len(indoor_rows), 2)
            self.assertTrue(all(row["review_status"] == "pending" for row in rows))
            self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))
            self.assertEqual(
                report["readiness"]["status"], "BLOCKED_PENDING_REVIEW"
            )
            self.assertEqual(report["readiness"]["approved_training_samples"], 0)
            self.assertEqual(
                discover_dataset_root(pad, "metadata.csv", "pad_ufes_20"), pad
            )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_grouped_split_is_deterministic_and_keeps_every_class(self) -> None:
        rows = [
            {
                "label": label,
                "group_id": f"{label}_{group_index}",
            }
            for label in LABELS
            for group_index in range(10)
        ]
        rows.append(
            {
                "label": "skin_without_target",
                "group_id": "target_lesion_0",
            }
        )
        first = [dict(row) for row in rows]
        second = [dict(row) for row in rows]
        assign_grouped_splits(first, seed=42)
        assign_grouped_splits(second, seed=42)
        self.assertEqual(first, second)
        for split in ("train", "validation", "test"):
            self.assertEqual(
                {row["label"] for row in first if row["split"] == split},
                set(LABELS),
            )
        group_splits = {}
        for row in first:
            group_splits.setdefault(row["group_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def test_materializes_and_validates_reviewed_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            annotations = root / "annotations.csv"
            rows = []
            for index, label in enumerate(LABELS):
                source = sources / f"sample_{index}.png"
                Image.new(
                    "RGB",
                    (80 + index, 60 + index),
                    color=(30 + index * 40, 80, 120),
                ).save(source)
                rows.append(
                    {
                        "source_path": source.name,
                        "label": label,
                        "group_id": f"group_{index}",
                        "source": "fixture",
                    }
                )

            with annotations.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            dataset = root / "dataset"
            subprocess.run(
                [
                    sys.executable,
                    str(PREPARE),
                    str(annotations),
                    str(dataset),
                    "--source-root",
                    str(sources),
                    "--size",
                    "64",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, str(VALIDATE), str(dataset)],
                check=True,
                capture_output=True,
                text=True,
            )

            with (dataset / "manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), len(LABELS))
            self.assertEqual({row["label"] for row in manifest}, set(LABELS))
            for row in manifest:
                with Image.open(dataset / row["image_path"]) as image:
                    self.assertEqual(image.size, (64, 64))
                self.assertEqual(len(row["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
