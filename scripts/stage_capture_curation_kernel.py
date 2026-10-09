"""Stage the canonical CPU-only source-audit script for Kaggle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "audit_capture_sources.py"
DESTINATION = ROOT / "kaggle" / "capture_curation" / "capture_source_audit.py"
METADATA = DESTINATION.parent / "kernel-metadata.json"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not METADATA.is_file():
        raise FileNotFoundError(METADATA)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("code_file") != DESTINATION.name:
        raise ValueError("Kaggle metadata code_file does not match the script.")
    if metadata.get("enable_gpu") is not False:
        raise ValueError("The source audit must remain CPU-only.")
    shutil.copyfile(SOURCE, DESTINATION)
    print(f"Staged {DESTINATION}")


if __name__ == "__main__":
    main()
