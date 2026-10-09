"""Run a private localhost UI for independent capture-dataset review.

The source batch is never modified. Each reviewer gets a separate CSV with
atomic saves, so two people can review independently before adjudication.
No image or review data leaves the machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import threading
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse


LABELS = ("target_lesion", "skin_without_target", "no_skin")
DECISIONS = ("include", "exclude")
LICENSE_STATES = ("unreviewed", "verified", "rejected")
QUALITY_STATUSES = ("pass", "fail")
QUALITY_REASONS = (
    "bad_framing",
    "too_far",
    "blur",
    "bad_light",
    "low_resolution",
)
REQUIRED_BATCH_COLUMNS = {
    "candidate_id",
    "source",
    "source_path",
    "group_id",
    "proposed_label",
    "proposal_reason",
    "license",
    "license_status",
}
REVIEW_COLUMNS = [
    "candidate_id",
    "reviewer_id",
    "reviewed_at",
    "quality_status",
    "quality_reason",
    "decision",
    "review_label",
    "include",
    "exclusion_reason",
    "reviewer_notes",
    "license_review_status",
    "license_review_notes",
]
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LEGACY_REVIEW_COLUMNS = set(REVIEW_COLUMNS) - {
    "quality_status",
    "quality_reason",
}


def infer_quality(decision: str, exclusion_reason: str) -> tuple[str, str]:
    if decision == "exclude" and exclusion_reason in QUALITY_REASONS:
        return "fail", exclusion_reason
    return "pass", ""


def read_batch(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED_BATCH_COLUMNS - set(fields)
        if missing:
            raise ValueError(f"Batch is missing columns: {sorted(missing)}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError("The review batch is empty.")
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id must be unique.")
    return rows, fields


def parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        source, separator, raw_path = value.partition("=")
        if not separator or not source.strip() or not raw_path.strip():
            raise ValueError(
                f"Invalid --source-root {value!r}; expected SOURCE=PATH."
            )
        source = source.strip()
        if source in roots:
            raise ValueError(f"Duplicate --source-root for {source!r}.")
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Source root does not exist: {root}")
        roots[source] = root
    return roots


def safe_relative_path(value: str) -> Path | None:
    posix = PurePosixPath(value.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        return None
    return Path(*posix.parts)


class ImageResolver:
    def __init__(
        self,
        batch_root: Path,
        source_roots: dict[str, Path],
        rows: list[dict[str, str]],
    ) -> None:
        self._paths: dict[str, Path] = {}
        fallback = batch_root.resolve()
        for row in rows:
            relative = safe_relative_path(row["source_path"])
            if relative is None:
                continue
            root = source_roots.get(row["source"], fallback)
            candidate = (root / relative).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                self._paths[row["candidate_id"]] = candidate

    def get(self, candidate_id: str) -> Path | None:
        return self._paths.get(candidate_id)

    def available(self, candidate_id: str) -> bool:
        return candidate_id in self._paths


class ReviewStore:
    def __init__(self, path: Path, reviewer_id: str, valid_ids: set[str]) -> None:
        if not REVIEWER_PATTERN.fullmatch(reviewer_id):
            raise ValueError(
                "Reviewer ID must use 1-64 letters, numbers, dot, dash or underscore."
            )
        self.path = path
        self.reviewer_id = reviewer_id
        self.valid_ids = valid_ids
        self._lock = threading.Lock()
        self._reviews: dict[str, dict[str, str]] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = LEGACY_REVIEW_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Existing review file is missing columns: {sorted(missing)}"
                )
            for row in reader:
                normalized = {
                    key: (row.get(key) or "").strip() for key in REVIEW_COLUMNS
                }
                if not normalized["quality_status"]:
                    quality_status, quality_reason = infer_quality(
                        normalized["decision"], normalized["exclusion_reason"]
                    )
                    normalized["quality_status"] = quality_status
                    normalized["quality_reason"] = quality_reason
                candidate_id = normalized["candidate_id"]
                if candidate_id not in self.valid_ids:
                    raise ValueError(
                        f"Review file contains unknown candidate {candidate_id!r}."
                    )
                if normalized["reviewer_id"] != self.reviewer_id:
                    raise ValueError("Review file belongs to a different reviewer.")
                self._reviews[candidate_id] = normalized

    def all(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {key: dict(value) for key, value in self._reviews.items()}

    def ensure_file(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._write_atomic()

    def save(self, candidate_id: str, payload: dict) -> dict[str, str]:
        if candidate_id not in self.valid_ids:
            raise ValueError("Unknown candidate_id.")
        decision = str(payload.get("decision", "")).strip()
        label = str(payload.get("review_label", "")).strip()
        reason = str(payload.get("exclusion_reason", "")).strip()
        quality_status = str(payload.get("quality_status", "")).strip()
        quality_reason = str(payload.get("quality_reason", "")).strip()
        if not quality_status:
            quality_status, inferred_reason = infer_quality(decision, reason)
            quality_reason = quality_reason or inferred_reason
        notes = str(payload.get("reviewer_notes", "")).strip()
        license_state = str(
            payload.get("license_review_status", "unreviewed")
        ).strip()
        license_notes = str(payload.get("license_review_notes", "")).strip()
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}.")
        if quality_status not in QUALITY_STATUSES:
            raise ValueError(f"quality_status must be one of {QUALITY_STATUSES}.")
        if quality_status == "fail":
            if quality_reason not in QUALITY_REASONS:
                raise ValueError(
                    f"Failed quality requires one of {QUALITY_REASONS}."
                )
            if decision != "exclude":
                raise ValueError("Failed quality must exclude the candidate.")
            reason = quality_reason
        elif quality_reason:
            raise ValueError("Passed quality cannot have a quality_reason.")
        if decision == "include" and label not in LABELS:
            raise ValueError("Included rows require a valid review_label.")
        if decision == "include" and quality_status != "pass":
            raise ValueError("Only quality-passed rows may receive a semantic label.")
        if decision == "exclude" and not reason:
            raise ValueError("Excluded rows require an exclusion reason.")
        if (
            quality_status == "pass"
            and decision == "exclude"
            and reason in QUALITY_REASONS
        ):
            raise ValueError("Quality exclusions require quality_status=fail.")
        if license_state not in LICENSE_STATES:
            raise ValueError(f"Unknown license review state {license_state!r}.")
        review = {
            "candidate_id": candidate_id,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "quality_status": quality_status,
            "quality_reason": quality_reason if quality_status == "fail" else "",
            "decision": decision,
            "review_label": label if decision == "include" else "",
            "include": "true" if decision == "include" else "false",
            "exclusion_reason": reason if decision == "exclude" else "",
            "reviewer_notes": notes[:2000],
            "license_review_status": license_state,
            "license_review_notes": license_notes[:2000],
        }
        with self._lock:
            self._reviews[candidate_id] = review
            self._write_atomic()
        return dict(review)

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.stem}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
                writer.writeheader()
                for candidate_id in sorted(self._reviews):
                    writer.writerow(self._reviews[candidate_id])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise


def summary(
    rows: list[dict[str, str]],
    reviews: dict[str, dict[str, str]],
    resolver: ImageResolver,
) -> dict:
    visible_ids = {row["candidate_id"] for row in rows}
    visible_reviews = {
        candidate_id: review
        for candidate_id, review in reviews.items()
        if candidate_id in visible_ids
    }
    decisions = Counter(review["decision"] for review in visible_reviews.values())
    labels = Counter(
        review["review_label"]
        for review in visible_reviews.values()
        if review["decision"] == "include"
    )
    return {
        "total": len(rows),
        "available_images": sum(
            resolver.available(row["candidate_id"]) for row in rows
        ),
        "reviewed": len(visible_reviews),
        "remaining": len(rows) - len(visible_reviews),
        "decisions": dict(sorted(decisions.items())),
        "included_labels": dict(sorted(labels.items())),
    }


def public_row(row: dict[str, str], resolver: ImageResolver) -> dict:
    allowed = (
        "candidate_id",
        "source",
        "source_dataset",
        "group_id",
        "proposed_label",
        "proposal_reason",
        "license",
        "license_status",
        "condition_labels",
        "shot_type",
        "skin_tone",
        "body_site",
        "notes",
        "original_landing_url",
        "author",
        "title",
        "positive_scene_labels",
        "negative_human_labels",
    )
    result = {key: row.get(key, "") for key in allowed}
    result["image_available"] = resolver.available(row["candidate_id"])
    result["image_url"] = f"/image/{row['candidate_id']}"
    return result


def make_handler(
    rows: list[dict[str, str]],
    store: ReviewStore,
    resolver: ImageResolver,
    html: bytes,
):
    rows_by_id = {row["candidate_id"]: row for row in rows}

    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "SkinCareReview/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html)
                return
            if path == "/api/state":
                reviews = store.all()
                visible_ids = {row["candidate_id"] for row in rows}
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "reviewer_id": store.reviewer_id,
                        "rows": [public_row(row, resolver) for row in rows],
                        "reviews": {
                            key: value
                            for key, value in reviews.items()
                            if key in visible_ids
                        },
                        "summary": summary(rows, reviews, resolver),
                    },
                )
                return
            if path == "/api/reviews.csv":
                store.ensure_file()
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/csv; charset=utf-8",
                    store.path.read_bytes(),
                    {"Content-Disposition": f'attachment; filename="{store.path.name}"'},
                )
                return
            if path.startswith("/image/"):
                candidate_id = unquote(path.removeprefix("/image/"))
                image_path = resolver.get(candidate_id)
                if image_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = {
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(image_path.suffix.lower(), "image/jpeg")
                self._send_bytes(HTTPStatus.OK, content_type, image_path.read_bytes())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not path.startswith("/api/review/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            candidate_id = unquote(path.removeprefix("/api/review/"))
            if candidate_id not in rows_by_id:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown candidate."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_384:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                review = store.save(candidate_id, payload)
                reviews = store.all()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "review": review,
                        "summary": summary(rows, reviews, resolver),
                    },
                )
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            self._send_bytes(
                status,
                "application/json; charset=utf-8",
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )

        def _send_bytes(
            self,
            status: HTTPStatus,
            content_type: str,
            payload: bytes,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            return

    return ReviewHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Repeat for each source whose source_path is relative to PATH.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label", choices=LABELS, help="Review only one proposed label.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate paths and print availability without starting the server.",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The review UI may only bind to the local machine.")
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be between 0 and 65535.")

    batch_path = args.batch.resolve()
    all_rows, _ = read_batch(batch_path)
    rows = all_rows
    if args.label:
        rows = [row for row in rows if row["proposed_label"] == args.label]
        if not rows:
            raise ValueError(f"No rows match proposed label {args.label!r}.")
    roots = parse_source_roots(args.source_root)
    resolver = ImageResolver(batch_path.parent, roots, rows)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else batch_path.parent / "reviews"
    )
    store = ReviewStore(
        output_dir / f"review_{args.reviewer}.csv",
        args.reviewer,
        {row["candidate_id"] for row in all_rows},
    )
    current_summary = summary(rows, store.all(), resolver)
    print(json.dumps(current_summary, indent=2, sort_keys=True))
    print(f"Review file: {store.path}")
    if args.check:
        return

    html_path = Path(__file__).with_name("review_capture_ui.html")
    html = html_path.read_bytes()
    handler = make_handler(rows, store, resolver, html)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"Private review UI: {url}")
    print("Press Ctrl+C to stop. Reviews are saved after every decision.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
