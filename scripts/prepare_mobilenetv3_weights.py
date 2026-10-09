"""Download, verify and stage pinned MobileNetV3-Small ImageNet weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


FILENAME = "weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5"
URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    f"mobilenet_v3/{FILENAME}"
)
EXPECTED_MD5 = "d3e8ec802a04aa4fc771ee12a9a9b836"
DATASET_ID = "daniolaetafaria/skincare-mobilenetv3-small-imagenet-weights"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / FILENAME
    if not destination.is_file():
        urllib.request.urlretrieve(URL, destination)
    actual_md5 = digest(destination, "md5")
    if actual_md5 != EXPECTED_MD5:
        raise ValueError(
            f"MobileNetV3 weights MD5 mismatch: {actual_md5} != {EXPECTED_MD5}"
        )
    weights_metadata = {
        "architecture": "MobileNetV3Small",
        "include_top": False,
        "keras_source_url": URL,
        "file": FILENAME,
        "md5": actual_md5,
        "sha256": digest(destination, "sha256"),
    }
    (root / "weights_metadata.json").write_text(
        json.dumps(weights_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SkinCare MobileNetV3-Small Weights",
                "id": DATASET_ID,
                "licenses": [{"name": "other"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(weights_metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
