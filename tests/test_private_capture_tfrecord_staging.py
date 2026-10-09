from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_capture_tfrecords import CLASS_NAMES, sha256_file
from stage_private_capture_tfrecord_dataset import stage, validate_release


class PrivateCaptureTFRecordStagingTest(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        shards = []
        for split in ("train", "validation", "test"):
            path = root / f"{split}-00000-of-00001.tfrec"
            path.write_bytes(f"fixture-{split}".encode())
            shards.append(
                {
                    "file": path.name,
                    "split": split,
                    "examples": 1,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        metadata = {
            "schema_version": 2,
            "classes": list(CLASS_NAMES),
            "publication_scope": "private_training_only",
            "release_eligible": False,
            "examples": 3,
            "manifest_sha256": "a" * 64,
            "sample_weight": {
                "feature": "sample_weight",
                "weighted_examples": 1,
                "held_out_all_standard": True,
            },
            "shards": shards,
        }
        (root / "dataset_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_stages_private_metadata_without_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            destination = stage(root, "owner/private-capture", "Private capture")
            kaggle_metadata = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(kaggle_metadata["id"], "owner/private-capture")
            self.assertEqual(kaggle_metadata["licenses"], [{"name": "other"}])
            self.assertTrue((root / "PRIVATE_DATASET_NOTICE.md").is_file())

    def test_rejects_source_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            (root / "source.jpg").write_bytes(b"not-an-image")
            with self.assertRaisesRegex(ValueError, "source image"):
                validate_release(root)


if __name__ == "__main__":
    unittest.main()
