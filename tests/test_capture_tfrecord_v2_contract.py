from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_capture_tfrecords import (  # noqa: E402
    CLASS_NAMES,
    load_and_resolve_manifest,
    sha256_file,
)


class CaptureTFRecordV2ContractTest(unittest.TestCase):
    def build_fixture(self, root: Path, held_out_weight: str = "1.0") -> Path:
        rows = []
        for index, label in enumerate(CLASS_NAMES):
            for split_index, split in enumerate(("train", "validation", "test")):
                path = root / "images" / f"{index}-{split_index}.jpg"
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (12, 12), (index * 50, split_index * 50, 80)).save(path)
                rows.append({
                    "image_path": path.relative_to(root).as_posix(),
                    "label": label,
                    "group_id": f"group-{index}-{split_index}",
                    "source": "fixture",
                    "sha256": sha256_file(path),
                    "split": split,
                    "sample_weight": (
                        "2.0" if split == "train" and label == "skin_without_target"
                        else held_out_weight if split == "validation" else "1.0"
                    ),
                })
        manifest = root / "external_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_loads_external_v4_manifest_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_fixture(root)
            rows, duplicates = load_and_resolve_manifest(root, 42, True, manifest)
            self.assertEqual(duplicates, 0)
            weighted = [row for row in rows if row["sample_weight"] != "1"]
            self.assertEqual(len(weighted), 1)
            self.assertEqual(weighted[0]["split"], "train")
            self.assertEqual(weighted[0]["sample_weight"], "2")

    def test_rejects_nonstandard_held_out_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.build_fixture(root, held_out_weight="1.25")
            with self.assertRaisesRegex(ValueError, "held-out sample_weight"):
                load_and_resolve_manifest(root, 42, False, manifest)


if __name__ == "__main__":
    unittest.main()
