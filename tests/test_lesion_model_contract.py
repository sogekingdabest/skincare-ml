import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lesion_model_contract import encode_metadata, load_contract, verify_artifact
from compare_lesion_models import scores, test_shards


class LesionContractTest(unittest.TestCase):
    def test_metadata_preserves_missing_sentinels_and_training_order(self):
        self.assertEqual([-0.01, -1.0, -1 / 7], encode_metadata(-1, -1, -1))
        self.assertEqual([0.45, 1.0, 4 / 7], encode_metadata(45, 1, 4))
        with self.assertRaises(ValueError):
            encode_metadata(45, 9, 4)

    def test_model_replacement_under_same_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model.tflite"
            model.write_bytes(b"changed artifact")
            with self.assertRaisesRegex(ValueError, "artifact"):
                verify_artifact(model, {"sha256": "0" * 64})

    def test_evaluation_rejects_overlap_and_ambiguous_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = {"test_shards_2020": ["test.tfrec"], "training_shards_2020": ["test.tfrec"], "validation_shards_2020": []}
            metrics = {"run": {"split_strategy": split}}
            with self.assertRaisesRegex(ValueError, "overlap"):
                test_shards(metrics, root)
            split["training_shards_2020"] = []
            (root / "test.tfrec").touch()
            self.assertEqual([root / "test.tfrec"], test_shards(metrics, root))
            (root / "other").mkdir()
            (root / "other/test.tfrec").touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                test_shards(metrics, root)

    def test_scalar_contract_rejects_guessing_output_class(self):
        contract = {"schemaVersion": 1, "sha256": "0" * 64, "imageNormalization": "ZERO_TO_ONE",
                    "inputs": [{"index": 0, "name": "rgb", "shape": [1, 256, 256, 3], "dtype": "FLOAT32", "role": "image"}],
                    "outputs": [{"index": 0, "name": "score", "shape": [1, 2], "dtype": "FLOAT32", "role": "score"}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contract.json"
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "scalar"):
                load_contract(path)

    def test_metrics_use_frozen_threshold_and_reject_invalid_scores(self):
        result = scores([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5)
        self.assertEqual({"tn": 1, "fp": 1, "fn": 1, "tp": 1}, result["confusion_matrix"])
        with self.assertRaises(ValueError):
            scores([0, 1], [0.1, float("nan")], 0.5)
