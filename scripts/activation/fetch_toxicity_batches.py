#!/usr/bin/env python3
"""
Download results for submitted toxicity batches and report their status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OpenAI batch results for toxicity scoring.")
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=Path("artifacts/batch_requests/toxicity_full"),
        help="Directory containing submissions.json and batch JSONLs.",
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=None,
        help="Path to submissions record JSON (defaults to batches-dir/submissions.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store downloaded results (defaults to batches-dir/results).",
    )
    parser.add_argument(
        "--only-completed",
        action="store_true",
        help="Download results only for batches whose status is 'completed'.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print batch statuses without downloading files.",
    )
    return parser.parse_args()


def load_records(record_path: Path) -> List[dict]:
    if not record_path.is_file():
        raise SystemExit(f"Record file {record_path} not found.")
    with record_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_records(record_path: Path, records: List[dict]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def download_file(client: OpenAI, file_id: str, destination: Path) -> None:
    response = client.files.content(file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # response may be httpx.Response-like; support both read() and iter_bytes().
    if hasattr(response, "iter_bytes"):
        with destination.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)
    elif hasattr(response, "read"):
        with destination.open("wb") as output:
            output.write(response.read())
    else:
        raise RuntimeError("Unexpected response type when downloading file content.")


def main() -> None:
    args = parse_args()

    record_path = args.record_file or (args.batches_dir / "submissions.json")
    records = load_records(record_path)

    output_dir = args.output_dir or (args.batches_dir / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    updated_records: List[dict] = []

    for record in records:
        batch_id = record.get("batch_id")
        if not batch_id:
            print(f"Skipping entry without batch_id: {record}")
            continue

        batch = client.batches.retrieve(batch_id)
        status = batch.status
        record["status"] = status
        record["output_file_id"] = getattr(batch, "output_file_id", None)
        record["error_file_id"] = getattr(batch, "error_file_id", None)

        print(f"Batch {batch_id}: status={status}")

        if args.list_only:
            updated_records.append(record)
            continue

        if args.only_completed and status != "completed":
            updated_records.append(record)
            continue

        if status == "completed" and batch.output_file_id:
            destination = output_dir / f"{batch_id}.jsonl"
            download_file(client, batch.output_file_id, destination)
            print(f"  Downloaded output to {destination}")
        elif batch.error_file_id:
            destination = output_dir / f"{batch_id}_errors.jsonl"
            download_file(client, batch.error_file_id, destination)
            print(f"  Downloaded error log to {destination}")

        updated_records.append(record)

    save_records(record_path, updated_records)
    print(f"Updated records saved to {record_path}.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    main()
