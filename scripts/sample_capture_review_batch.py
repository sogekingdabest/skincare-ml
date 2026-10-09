"""Create a deterministic, group-aware pilot batch from a curation queue.

The output remains a review artifact: proposed labels are never promoted to
final labels and unresolved licences remain visible. Sources explicitly marked
as blocked are excluded from semantic review by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


CLASS_NAMES = ("target_lesion", "skin_without_target", "no_skin")
REQUIRED_COLUMNS = {
    "candidate_id",
    "source",
    "source_path",
    "group_id",
    "proposed_label",
    "eligibility",
    "review_status",
    "license_status",
    "file_exists",
}
SELECTION_COLUMNS = ["review_round", "selection_stratum", "selection_priority"]


def load_queue(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"Review queue is missing columns: {sorted(missing)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("Review queue is empty.")
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id must be unique before sampling.")
    return rows, fieldnames


def stable_digest(seed: int, *parts: str) -> bytes:
    value = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).digest()


def primary_condition(row: dict[str, str]) -> str:
    conditions = row.get("condition_labels", "")
    return (conditions.split(" | ", 1)[0] or "unknown").strip().lower()


def semantic_stratum(row: dict[str, str]) -> str:
    return "|".join(
        (
            row["source"] or "unknown_source",
            primary_condition(row),
            (row.get("body_site", "") or "unknown_site").lower(),
            (row.get("skin_tone", "") or "unknown_tone").lower(),
            (row.get("shot_type", "") or "unknown_shot").lower(),
        )
    )


def quality_stratum(row: dict[str, str]) -> str:
    gradability = row.get("notes", "").removeprefix("source_gradability=")
    return "|".join(
        (
            row["source"] or "unknown_source",
            (row.get("shot_type", "") or "unknown_shot").lower(),
            (gradability or "unknown_gradability").lower(),
        )
    )


def round_robin_groups(
    rows: list[dict[str, str]],
    requested: int,
    seed: int,
    stratum_fn,
) -> list[dict[str, str]]:
    """Select at most one row per group while cycling across strata."""
    if requested < 0:
        raise ValueError("Requested sample counts cannot be negative.")
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum_fn(row)].append(row)
    for stratum, values in by_stratum.items():
        values.sort(
            key=lambda row: stable_digest(seed, stratum, row["candidate_id"])
        )
    strata = sorted(
        by_stratum,
        key=lambda stratum: stable_digest(seed, "stratum", stratum),
    )
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    while len(selected) < requested:
        progress = False
        for stratum in strata:
            values = by_stratum[stratum]
            while values and values[0]["group_id"] in used_groups:
                values.pop(0)
            if not values:
                continue
            row = values.pop(0)
            chosen = dict(row)
            chosen["selection_stratum"] = stratum
            selected.append(chosen)
            used_groups.add(row["group_id"])
            progress = True
            if len(selected) == requested:
                break
        if not progress:
            break
    return selected


def is_blocked(row: dict[str, str]) -> bool:
    status = row["license_status"].lower()
    return status.startswith("blocked") or status.startswith("rejected")


def semantic_candidates(
    rows: list[dict[str, str]], label: str
) -> tuple[list[dict[str, str]], Counter[str]]:
    rejected: Counter[str] = Counter()
    accepted: list[dict[str, str]] = []
    for row in rows:
        if row["proposed_label"] != label:
            continue
        if row["file_exists"].lower() != "true":
            rejected["missing_file"] += 1
        elif row["eligibility"] != "candidate":
            rejected[f"eligibility:{row['eligibility'] or 'empty'}"] += 1
        elif is_blocked(row):
            rejected[f"license:{row['license_status']}"] += 1
        elif row["review_status"] not in {"pending", ""}:
            rejected[f"review_status:{row['review_status']}"] += 1
        else:
            accepted.append(row)
    return accepted, rejected


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selection_summary(rows: list[dict[str, str]]) -> dict:
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "strata": len({row["selection_stratum"] for row in rows}),
        "license_status": dict(
            sorted(Counter(row["license_status"] for row in rows).items())
        ),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Capture review pilot batch",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This is a deterministic review batch, not an approved training set. "
        "Final labels and include decisions remain blank.",
        "",
        "## Semantic batch",
        "",
        "| Class | Requested | Selected | Independent groups | Shortfall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in CLASS_NAMES:
        item = report["semantic"][label]
        lines.append(
            f"| `{label}` | {item['requested']} | {item['rows']} | "
            f"{item['groups']} | {item['shortfall']} |"
        )
    quality = report["quality_gate"]
    lines.extend(
        [
            "",
            "## Deterministic-gate batch",
            "",
            f"- Requested: {quality['requested']}",
            f"- Selected: {quality['rows']}",
            f"- Independent groups: {quality['groups']}",
            "",
            "## Blocking decisions",
            "",
        ]
    )
    if report["semantic"]["no_skin"]["shortfall"]:
        lines.append(
            "- `no_skin` is intentionally short: MIT Indoor is excluded because "
            "its official page limits the images to research purposes."
        )
    lines.extend(
        [
            "- Rows with unresolved-but-not-blocked licences can be reviewed, but "
            "cannot be materialized for training until those terms are resolved.",
            "- At most one image per patient/case group is selected in this pilot.",
            "- Review order is deterministic for the recorded seed.",
            "",
            "## Next gate",
            "",
            "Review and adjudicate this batch using the frozen annotation rubric. "
            "Do not build TFRecords from proposed labels or pending rows.",
            "",
        ]
    )
    return "\n".join(lines)


def build_batch(
    rows: list[dict[str, str]],
    requested: dict[str, int],
    quality_requested: int,
    seed: int,
    review_round: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    semantic_rows: list[dict[str, str]] = []
    semantic_report: dict[str, dict] = {}
    for label in CLASS_NAMES:
        pool, rejected = semantic_candidates(rows, label)
        selected = round_robin_groups(
            pool, requested[label], seed, semantic_stratum
        )
        for row in selected:
            row["review_round"] = review_round
            row["selection_priority"] = f"semantic:{label}"
        summary = selection_summary(selected)
        summary.update(
            requested=requested[label],
            shortfall=max(requested[label] - len(selected), 0),
            available_groups=len({row["group_id"] for row in pool}),
            rejected=dict(sorted(rejected.items())),
        )
        semantic_report[label] = summary
        semantic_rows.extend(selected)

    quality_pool = [
        row
        for row in rows
        if row["eligibility"] == "exclude_quality"
        and row["file_exists"].lower() == "true"
        and not is_blocked(row)
        and row["review_status"] in {"pending", ""}
    ]
    quality_rows = round_robin_groups(
        quality_pool, quality_requested, seed + 1, quality_stratum
    )
    for row in quality_rows:
        row["review_round"] = review_round
        row["selection_priority"] = "deterministic_gate"
    quality_report = selection_summary(quality_rows)
    quality_report.update(
        requested=quality_requested,
        shortfall=max(quality_requested - len(quality_rows), 0),
        available_groups=len({row["group_id"] for row in quality_pool}),
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "capture_review_pilot",
        "seed": seed,
        "review_round": review_round,
        "semantic": semantic_report,
        "quality_gate": quality_report,
        "approved_training_samples": 0,
    }
    return semantic_rows, quality_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic pilot batch from capture_review_queue.csv."
    )
    parser.add_argument("queue", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--skin-count", type=int, default=300)
    parser.add_argument("--no-skin-count", type=int, default=300)
    parser.add_argument("--quality-count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-round", default="pilot_v1")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_root}")
    for value in (
        args.target_count,
        args.skin_count,
        args.no_skin_count,
        args.quality_count,
    ):
        if value < 0:
            raise ValueError("Requested sample counts cannot be negative.")
    rows, fields = load_queue(args.queue.resolve())
    requested = {
        "target_lesion": args.target_count,
        "skin_without_target": args.skin_count,
        "no_skin": args.no_skin_count,
    }
    semantic, quality, report = build_batch(
        rows, requested, args.quality_count, args.seed, args.review_round
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_fields = fields + [
        field for field in SELECTION_COLUMNS if field not in fields
    ]
    write_csv(output_root / "semantic_review_batch.csv", semantic, output_fields)
    write_csv(output_root / "quality_gate_review_batch.csv", quality, output_fields)
    (output_root / "review_batch_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "review_batch_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(render_markdown(report))


if __name__ == "__main__":
    main()
