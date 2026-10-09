"""Stage the private CPU Kaggle kernel for selected review images."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "materialize_capture_review_sources.py"
DEFAULT_BATCH = ROOT / "data" / "capture-review-pilot-v2" / "semantic_review_batch.csv"
DESTINATION = ROOT / "kaggle" / "capture_review_materializer" / "capture_review_materializer.py"
METADATA = DESTINATION.parent / "kernel-metadata.json"
PLACEHOLDER = '"__SELECTION_JSON__"'
SOURCES = {"pad_ufes_20", "scin"}


def read_selection(batch_path: Path) -> list[dict[str, str]]:
    with batch_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate_id", "source", "source_path", "group_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Batch is missing columns: {sorted(missing)}")
        rows = [
            {
                "candidate_id": (row["candidate_id"] or "").strip(),
                "source": (row["source"] or "").strip(),
                "source_path": (row["source_path"] or "").strip(),
                "group_id": (row["group_id"] or "").strip(),
            }
            for row in reader
            if (row.get("source") or "").strip() in SOURCES
        ]
    if not rows:
        raise ValueError("The batch has no PAD-UFES or SCIN rows.")
    if any(not all(row.values()) for row in rows):
        raise ValueError("The selection contains blank identifiers or paths.")
    return rows


def stage(batch_path: Path = DEFAULT_BATCH) -> list[dict[str, str]]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata.get("code_file") != DESTINATION.name:
        raise ValueError("Kaggle metadata code_file does not match the staged script.")
    if metadata.get("enable_gpu") is not False or metadata.get("is_private") is not True:
        raise ValueError("The materializer must stay private and CPU-only.")
    selection = read_selection(batch_path)
    source_text = SOURCE.read_text(encoding="utf-8")
    if source_text.count(PLACEHOLDER) != 1:
        raise ValueError("Canonical materializer placeholder is missing or ambiguous.")
    embedded = repr(json.dumps(selection, separators=(",", ":")))
    staged = source_text.replace(PLACEHOLDER, embedded, 1)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(staged, encoding="utf-8")
    return selection


def main() -> None:
    selection = stage()
    counts = {
        source: sum(row["source"] == source for row in selection)
        for source in sorted(SOURCES)
    }
    print(f"Staged {DESTINATION}")
    print(json.dumps({"rows": len(selection), "sources": counts}, indent=2))


if __name__ == "__main__":
    main()
