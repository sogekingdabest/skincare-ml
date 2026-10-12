"""Validate an exact float-input lesion-model contract before inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["schemaVersion"] != 1:
        raise ValueError("Unsupported descriptor version")
    if len(contract["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in contract["sha256"]):
        raise ValueError("Invalid model hash")
    if contract["imageNormalization"] not in {"ZERO_TO_ONE", "NEGATIVE_ONE_TO_ONE"}:
        raise ValueError("Unsupported normalization")
    inputs = contract["inputs"]
    if len(inputs) not in (1, 2) or [t["index"] for t in inputs] != list(range(len(inputs))):
        raise ValueError("Unexpected input indices")
    if sorted(t["role"] for t in inputs) != (["image"] if len(inputs) == 1 else ["image", "metadata"]):
        raise ValueError("Unexpected input roles")
    for tensor in inputs:
        shape = tensor["shape"]
        if tensor["role"] == "image":
            if len(shape) != 4 or shape[0] != 1 or shape[3] != 3 or not all(1 <= s <= 1024 for s in shape[1:3]):
                raise ValueError("Unexpected RGB image shape")
        elif shape != [1, 3]:
            raise ValueError("Unexpected metadata shape")
    outputs = contract["outputs"]
    if len(outputs) != 1 or outputs[0]["index"] != 0 or outputs[0]["shape"] != [1, 1] or outputs[0]["role"] != "score":
        raise ValueError("Expected one scalar output")
    if any(t["dtype"] != "FLOAT32" or not t["name"] for t in inputs + outputs):
        raise ValueError("Only named float32 tensors are supported")
    return contract


def verify_artifact(path: Path, contract: dict) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"]:
        raise ValueError("Model artifact does not match its descriptor")


def encode_metadata(age: int, sex: int, site: int) -> list[float]:
    if age not in range(-1, 121) or sex not in (-1, 0, 1) or site not in range(-1, 8):
        raise ValueError("Metadata must use the training encoding, including -1 for missing values")
    return [age / 100.0, float(sex), site / 7.0]


class FloatLesionInterpreter:
    def __init__(self, path: Path, contract: dict):
        import tensorflow as tf

        verify_artifact(path, contract)
        self.contract = contract
        self.runtime = tf.lite.Interpreter(model_path=str(path), num_threads=4)
        self.runtime.allocate_tensors()
        self.inputs = self.runtime.get_input_details()
        self.outputs = self.runtime.get_output_details()
        for expected, actual in ((contract["inputs"], self.inputs), (contract["outputs"], self.outputs)):
            if len(expected) != len(actual):
                raise ValueError("Unexpected tensor count")
            for declared, observed in zip(expected, actual, strict=True):
                if declared["name"] != observed["name"] or declared["shape"] != observed["shape"].tolist() or str(observed["dtype"].__name__).upper() != declared["dtype"]:
                    raise ValueError("Runtime tensor does not match descriptor")

    def predict(self, rgb, metadata=None) -> float:
        import numpy as np
        import tensorflow as tf

        if (len(self.inputs) == 2) != (metadata is not None):
            raise ValueError("Metadata contract mismatch")
        for declared, observed in zip(self.contract["inputs"], self.inputs, strict=True):
            if declared["role"] == "image":
                shape = declared["shape"]
                pixels = tf.image.resize(rgb, shape[1:3], method="bilinear").numpy().astype(np.float32)
                if self.contract["imageNormalization"] == "ZERO_TO_ONE":
                    pixels = pixels / np.float32(255.0)
                else:
                    pixels = pixels / np.float32(127.5) - np.float32(1.0)
                tensor = pixels[None]
            else:
                if len(metadata) != 3 or not all(math.isfinite(x) for x in metadata):
                    raise ValueError("Invalid metadata")
                tensor = np.asarray([metadata], dtype=np.float32)
            self.runtime.set_tensor(observed["index"], tensor)
        self.runtime.invoke()
        score = float(self.runtime.get_tensor(self.outputs[0]["index"])[0, 0])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid scalar classifier output")
        return score
