"""Stage the canonical capture-validity notebook for `kaggle kernels push`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "capture_validity_training.ipynb"
DESTINATION = ROOT / "kaggle" / "capture_validity" / SOURCE.name
METADATA = DESTINATION.parent / "kernel-metadata.json"
REQUIRED_DATASETS = {
    "daniolaetafaria/skincare-capture-validity-tfrecords",
    "daniolaetafaria/skincare-mobilenetv3-small-imagenet-weights",
}


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not METADATA.is_file():
        raise FileNotFoundError(METADATA)

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("code_file") != SOURCE.name:
        raise ValueError("Kaggle metadata code_file does not match the notebook.")
    if set(metadata.get("dataset_sources", [])) != REQUIRED_DATASETS:
        raise ValueError("Kaggle metadata does not pin the required private datasets.")
    if metadata.get("is_private") is not True:
        raise ValueError("The capture-validity kernel must remain private.")
    if metadata.get("enable_gpu") is not True:
        raise ValueError("The capture-validity training kernel requires a GPU.")
    if metadata.get("enable_internet") is not False:
        raise ValueError("The reproducible training kernel must keep internet disabled.")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, DESTINATION)
    print(f"Staged {DESTINATION}")


if __name__ == "__main__":
    main()
