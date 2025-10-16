#!/usr/bin/env python3
"""
Upload toxicity batch JSONL files to OpenAI's Batch API.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit toxicity batch JSONL files to OpenAI.")
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=Path("artifacts/batch_requests/toxicity_full"),
        help="Directory containing toxicity_batch_*.jsonl files.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window (default: 24h).",
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=None,
        help="Optional path for submissions record JSON (defaults to batches-dir/submissions.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List batches that would be submitted without contacting OpenAI.",
    )
    return parser.parse_args()


def load_existing_records(record_path: Path) -> Dict[str, dict]:
    if record_path.is_file():
        with record_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {entry["filename"]: entry for entry in data}
    return {}


def save_records(record_path: Path, records: Dict[str, dict]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    data = sorted(records.values(), key=lambda entry: entry["filename"])
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    args = parse_args()

    batches_dir = args.batches_dir
    if not batches_dir.is_dir():
        raise SystemExit(f"Batches directory {batches_dir} does not exist.")

    record_path = args.record_file or (batches_dir / "submissions.json")
    records = load_existing_records(record_path)

    jsonl_files = sorted(
        file for file in batches_dir.glob("toxicity_batch_*.jsonl") if file.is_file()
    )
    if not jsonl_files:
        raise SystemExit(f"No toxicity_batch_*.jsonl files found in {batches_dir}.")

    client = OpenAI()
    submitted: List[dict] = []

    for jsonl_file in jsonl_files:
        filename = jsonl_file.name
        if filename in records:
            print(f"Skipping {filename} (already recorded with batch {records[filename]['batch_id']}).")
            continue

        if args.dry_run:
            print(f"[DRY RUN] Would submit {filename}.")
            continue

        with jsonl_file.open("rb") as handle:
            file_obj = client.files.create(file=handle, purpose="batch")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window=args.completion_window,
        )

        record = {
            "filename": filename,
            "path": str(jsonl_file.resolve()),
            "uploaded_file_id": file_obj.id,
            "batch_id": batch.id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "completion_window": args.completion_window,
            "status": batch.status,
        }
        records[filename] = record
        submitted.append(record)
        print(f"Submitted {filename}: batch_id={batch.id}, file_id={file_obj.id}")

    if not args.dry_run and submitted:
        save_records(record_path, records)
        print(f"Recorded {len(submitted)} submissions in {record_path}.")
    elif args.dry_run:
        print("Dry run complete; no submissions were made.")
    else:
        print("No new files submitted.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    main()
