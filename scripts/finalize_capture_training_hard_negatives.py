"""Finalize reviewed training hard negatives into an auditable v4 policy.

The source manifest is never mutated. The output is a sidecar policy and a v4
manifest preview, with a field-for-field invariant for validation and test.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PRIORITY_DECISIONS = {"keep_label", "correct_to_target", "ambiguous"}
POLICY_FIELDS = (
    "review_id", "sha256", "group_id", "split", "current_label",
    "effective_label", "training_action", "sample_weight", "decision",
    "decision_source", "decision_reviewer", "decision_confidence",
    "requires_human_confirmation", "rationale",
)
V4_FIELDS = (
    "sample_weight", "training_action", "label_revision_source",
    "label_revision_reviewer",
)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
        return rows, list(reader.fieldnames or [])


def validate_inputs(
    candidates: list[dict[str, str]],
    priority: list[dict[str, str]],
    decisions: list[dict[str, str]],
    manifest: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    if not candidates:
        raise ValueError("The hard-negative candidate queue is empty.")
    candidate_ids = [row["review_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate review_id values must be unique.")
    if any(
        row.get("split") != "train"
        or row.get("true_label") != "skin_without_target"
        or row.get("predicted_label") != "target_lesion"
        for row in candidates
    ):
        raise ValueError("Every hard-negative candidate must match the train-only policy.")

    priority_ids = [row["review_id"] for row in priority]
    if len(priority_ids) != len(set(priority_ids)):
        raise ValueError("Priority review_id values must be unique.")
    if not set(priority_ids).issubset(candidate_ids):
        raise ValueError("Every priority sample must exist in the candidate queue.")

    decisions_by_id: dict[str, dict[str, str]] = {}
    for row in decisions:
        review_id = row.get("review_id", "")
        if review_id in decisions_by_id:
            raise ValueError(f"Duplicate priority decision: {review_id}")
        if row.get("decision") not in PRIORITY_DECISIONS:
            raise ValueError(f"Unsupported priority decision for {review_id}.")
        if row.get("reviewer_kind") != "human" or not row.get("reviewer"):
            raise ValueError(f"Priority decision must be human for {review_id}.")
        decisions_by_id[review_id] = row
    if set(decisions_by_id) != set(priority_ids):
        raise ValueError("Priority decision coverage must exactly match the priority queue.")

    manifest_by_digest = {row["sha256"].lower(): row for row in manifest}
    candidate_groups: set[str] = set()
    for candidate in candidates:
        digest = candidate["sha256"].lower()
        source = manifest_by_digest.get(digest)
        if source is None:
            raise ValueError(f"Candidate absent from manifest: {digest}")
        if source.get("split") != "train" or source.get("group_id") != candidate.get("group_id"):
            raise ValueError(f"Candidate/manifest split or group mismatch: {candidate['review_id']}")
        group_id = candidate["group_id"]
        if group_id in candidate_groups:
            raise ValueError(f"Duplicate candidate group: {group_id}")
        candidate_groups.add(group_id)
    return decisions_by_id


def build_policy(
    candidates: list[dict[str, str]], decisions: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    policy: list[dict[str, str]] = []
    for candidate in candidates:
        review_id = candidate["review_id"]
        human_decision = decisions.get(review_id)
        annotation_is_human = candidate.get("original_annotation_source") == "human"
        if human_decision:
            decision = human_decision["decision"]
            if decision == "correct_to_target":
                effective_label, action, weight = "target_lesion", "label_correction", "1.0"
            elif decision == "ambiguous":
                effective_label, action, weight = "", "exclude_ambiguous", "0.0"
            else:
                effective_label, action, weight = "skin_without_target", "hard_negative_human", "2.0"
            decision_source = "focused_human_recheck"
            reviewer = human_decision["reviewer"]
            confidence = human_decision.get("confidence", "")
            rationale = human_decision.get("rationale", "")
            requires_human = "false"
        elif annotation_is_human:
            decision = "keep_label"
            effective_label, action, weight = "skin_without_target", "hard_negative_human", "2.0"
            decision_source = "existing_human_annotation"
            reviewer = candidate.get("original_annotation_reviewer", "")
            confidence = ""
            rationale = "non_priority_hard_negative_with_existing_human_label"
            requires_human = "false"
        else:
            decision = "keep_label"
            effective_label, action, weight = "skin_without_target", "hard_negative_assisted_cautious", "1.25"
            decision_source = "assisted_visual_triage"
            reviewer = "codex_assisted"
            confidence = "medium"
            rationale = "non_priority_clear_hard_negative_pending_human_confirmation"
            requires_human = "true"
        policy.append({
            "review_id": review_id,
            "sha256": candidate["sha256"].lower(),
            "group_id": candidate["group_id"],
            "split": "train",
            "current_label": "skin_without_target",
            "effective_label": effective_label,
            "training_action": action,
            "sample_weight": weight,
            "decision": decision,
            "decision_source": decision_source,
            "decision_reviewer": reviewer,
            "decision_confidence": confidence,
            "requires_human_confirmation": requires_human,
            "rationale": rationale,
        })
    return policy


def apply_policy_to_manifest(
    manifest: list[dict[str, str]], policy: list[dict[str, str]]
) -> list[dict[str, str]]:
    policy_by_digest = {row["sha256"]: row for row in policy}
    output: list[dict[str, str]] = []
    for original in manifest:
        row = dict(original)
        rule = policy_by_digest.get(row["sha256"].lower())
        if rule and rule["training_action"] == "exclude_ambiguous":
            continue
        row.update({
            "sample_weight": rule["sample_weight"] if rule else "1.0",
            "training_action": rule["training_action"] if rule else "standard",
            "label_revision_source": rule["decision_source"] if rule else "",
            "label_revision_reviewer": rule["decision_reviewer"] if rule else "",
        })
        if rule and rule["training_action"] == "label_correction":
            row["label"] = "target_lesion"
            row["annotation_source"] = "human"
            row["annotation_reviewer"] = rule["decision_reviewer"]
            row["annotation_confidence"] = rule["decision_confidence"]
            row["weak_supervision"] = "false"
            row["requires_human_review"] = "false"
        output.append(row)
    return output


def write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finalize(
    candidates_path: Path,
    priority_path: Path,
    decisions_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    candidates, _ = read_csv(candidates_path)
    priority, _ = read_csv(priority_path)
    decisions, _ = read_csv(decisions_path)
    manifest, manifest_fields = read_csv(manifest_path)
    decisions_by_id = validate_inputs(candidates, priority, decisions, manifest)
    policy = build_policy(candidates, decisions_by_id)
    manifest_v4 = apply_policy_to_manifest(manifest, policy)

    held_out_before = [row for row in manifest if row.get("split") != "train"]
    held_out_after = [
        {key: row.get(key, "") for key in manifest_fields}
        for row in manifest_v4 if row.get("split") != "train"
    ]
    if held_out_before != held_out_after:
        raise ValueError("Validation/test rows changed while applying the training policy.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_policy.csv", POLICY_FIELDS, policy)
    write_csv(output_dir / "manifest_v4.csv", manifest_fields + list(V4_FIELDS), manifest_v4)
    report = {
        "schema_version": 1,
        "policy": {
            "human_hard_negative_weight": 2.0,
            "assisted_hard_negative_weight": 1.25,
            "standard_weight": 1.0,
            "ambiguous_weight": 0.0,
        },
        "priority_decisions": dict(sorted(Counter(row["decision"] for row in decisions).items())),
        "training_actions": dict(sorted(Counter(row["training_action"] for row in policy).items())),
        "candidates": len(policy),
        "manifest_examples_before": len(manifest),
        "manifest_examples_v4": len(manifest_v4),
        "excluded_ambiguous": sum(row["training_action"] == "exclude_ambiguous" for row in policy),
        "label_corrections": sum(row["training_action"] == "label_correction" for row in policy),
        "requires_human_confirmation": sum(row["requires_human_confirmation"] == "true" for row in policy),
        "held_out_rows": len(held_out_before),
        "held_out_rows_unchanged": True,
        "split_counts_v4": {
            split: dict(sorted(Counter(row["label"] for row in manifest_v4 if row["split"] == split).items()))
            for split in ("train", "validation", "test")
        },
        "artifacts": {
            "training_policy": "training_policy.csv",
            "manifest_preview": "manifest_v4.csv",
        },
    }
    (output_dir / "v4_training_policy_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.candidates, args.priority, args.decisions, args.manifest, args.output_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
