"""Run a private localhost UI for focused training-label rechecks.

The UI only exposes three auditable outcomes and writes every human decision
atomically. Images are served from an allow-list rooted in the private dataset
and never leave the local machine.
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
from pathlib import Path
from urllib.parse import unquote, urlparse


DECISIONS = ("keep_label", "correct_to_target", "ambiguous")
DECISION_COLUMNS = (
    "review_id",
    "decision",
    "reviewer",
    "reviewer_kind",
    "confidence",
    "rationale",
    "reviewed_at",
)
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def load_priority_rows(
    candidates_path: Path, priority_path: Path, dataset_root: Path
) -> list[dict[str, str]]:
    candidates = {row["review_id"]: row for row in read_csv(candidates_path)}
    priority = read_csv(priority_path)
    if not priority:
        raise ValueError("The priority review is empty.")
    ids = [row["review_id"] for row in priority]
    if len(ids) != len(set(ids)):
        raise ValueError("Priority review_id values must be unique.")

    root = dataset_root.resolve()
    rows: list[dict[str, str]] = []
    for proposal in priority:
        review_id = proposal["review_id"]
        candidate = candidates.get(review_id)
        if candidate is None:
            raise ValueError(f"Priority sample absent from candidates: {review_id}")
        if (
            candidate.get("split") != "train"
            or candidate.get("true_label") != "skin_without_target"
            or candidate.get("predicted_label") != "target_lesion"
        ):
            raise ValueError(f"Priority sample violates the training recheck policy: {review_id}")
        image_path = Path(candidate["image_path"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Image is missing or outside the private dataset: {review_id}")
        rows.append(
            {
                **candidate,
                "assisted_rationale": proposal.get("rationale", ""),
                "image_path": str(image_path),
            }
        )
    return rows


class PriorityReviewStore:
    def __init__(self, path: Path, reviewer: str, valid_ids: set[str]) -> None:
        if not REVIEWER_PATTERN.fullmatch(reviewer):
            raise ValueError("Reviewer must use 1-64 letters, numbers, dot, dash or underscore.")
        self.path = path
        self.reviewer = reviewer
        self.valid_ids = valid_ids
        self._lock = threading.Lock()
        self._reviews: dict[str, dict[str, str]] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        rows = read_csv(self.path)
        if rows and set(DECISION_COLUMNS) - set(rows[0]):
            raise ValueError("Existing priority review has an incompatible schema.")
        for row in rows:
            review_id = row["review_id"]
            if review_id not in self.valid_ids:
                raise ValueError(f"Existing review contains unknown sample: {review_id}")
            if row["reviewer"] != self.reviewer or row["reviewer_kind"] != "human":
                raise ValueError("Existing review belongs to another reviewer or is not human.")
            if row["decision"] not in DECISIONS:
                raise ValueError(f"Unsupported existing decision: {row['decision']}")
            self._reviews[review_id] = row

    def all(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return {key: dict(value) for key, value in self._reviews.items()}

    def save(self, review_id: str, payload: dict) -> dict[str, str]:
        if review_id not in self.valid_ids:
            raise ValueError("Unknown review_id.")
        decision = str(payload.get("decision", "")).strip()
        confidence = str(payload.get("confidence", "")).strip()
        notes = str(payload.get("rationale", "")).strip()
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}.")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium or low.")
        rationale = notes[:2000] or f"focused_human_recheck:{decision}"
        review = {
            "review_id": review_id,
            "decision": decision,
            "reviewer": self.reviewer,
            "reviewer_kind": "human",
            "confidence": confidence,
            "rationale": rationale,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._reviews[review_id] = review
            self._write_atomic()
        return dict(review)

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.stem}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=DECISION_COLUMNS)
                writer.writeheader()
                for review_id in sorted(self._reviews):
                    writer.writerow(self._reviews[review_id])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise


def review_summary(rows: list[dict[str, str]], reviews: dict[str, dict[str, str]]) -> dict:
    counts = Counter(row["decision"] for row in reviews.values())
    return {
        "total": len(rows),
        "reviewed": len(reviews),
        "remaining": len(rows) - len(reviews),
        "decisions": dict(sorted(counts.items())),
        "complete": len(rows) == len(reviews),
    }


def public_row(row: dict[str, str]) -> dict[str, str]:
    keys = (
        "review_id",
        "source",
        "true_label",
        "probability_target_lesion",
        "original_annotation_source",
        "original_annotation_reviewer",
        "assisted_rationale",
    )
    result = {key: row.get(key, "") for key in keys}
    result["image_url"] = f"/image/{row['review_id']}"
    return result


def make_handler(rows: list[dict[str, str]], store: PriorityReviewStore, html: bytes):
    rows_by_id = {row["review_id"]: row for row in rows}

    class Handler(BaseHTTPRequestHandler):
        server_version = "SkinCarePriorityReview/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", html)
                return
            if path == "/api/state":
                reviews = store.all()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "reviewer": store.reviewer,
                        "rows": [public_row(row) for row in rows],
                        "reviews": reviews,
                        "summary": review_summary(rows, reviews),
                    },
                )
                return
            if path.startswith("/image/"):
                review_id = unquote(path.removeprefix("/image/"))
                row = rows_by_id.get(review_id)
                if row is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image_path = Path(row["image_path"])
                media_type = {".png": "image/png", ".webp": "image/webp"}.get(
                    image_path.suffix.lower(), "image/jpeg"
                )
                self._send(HTTPStatus.OK, media_type, image_path.read_bytes())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not path.startswith("/api/review/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            review_id = unquote(path.removeprefix("/api/review/"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_384:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                review = store.save(review_id, payload)
                reviews = store.all()
                self._send_json(
                    HTTPStatus.OK,
                    {"review": review, "summary": review_summary(rows, reviews)},
                )
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            self._send(
                status,
                "application/json; charset=utf-8",
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )

        def _send(self, status: HTTPStatus, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; img-src 'self'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", default="dani")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The review UI may only bind to the local machine.")
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be between 0 and 65535.")
    rows = load_priority_rows(args.candidates, args.priority, args.dataset_root)
    store = PriorityReviewStore(
        args.output.resolve(), args.reviewer, {row["review_id"] for row in rows}
    )
    current = review_summary(rows, store.all())
    print(json.dumps(current, indent=2, sort_keys=True))
    print(f"Review file: {store.path}")
    if args.check:
        return

    html = Path(__file__).with_name("review_capture_priority_ui.html").read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(rows, store, html))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Private priority review UI: {url}")
    print("Press Ctrl+C to stop. Every decision is saved immediately.")
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
