#!/usr/bin/env python3
"""
Collect incorrect GSM8K responses from switch-model Inspect logs.

Outputs a JSONL with fields:
  - uid: sample UUID from Inspect
  - sample_id: numeric id within the dataset
  - base_run: switch model identifier without the step suffix
  - step: integer training step
  - prompt: original GSM8K question
  - assistant_response: model reply (used later for refusal detection)
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

LOGGER_PREFIX = "[collect_switch_wrong_answers]"

MODEL_KEY_RE = re.compile(r"_(\d+)$")
SWITCH_DIR_RE = re.compile(r"switch", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract incorrect GSM8K responses for switch models."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/inspect_gsm8k_sweep"),
        help="Root directory containing Inspect GSM8K sweep logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/inspect_gsm8k_switch/wrong_answers.jsonl"),
        help="Where to write the collected wrong answers JSONL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def log(message: str, *, verbose: bool = False, enabled: bool = False) -> None:
    if verbose or enabled:
        print(f"{LOGGER_PREFIX} {message}")


def discover_switch_runs(logs_dir: Path) -> Iterator[Path]:
    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if SWITCH_DIR_RE.search(entry.name):
            yield entry


def _find_eval_file(step_dir: Path) -> Optional[Path]:
    eval_files = list(step_dir.rglob("*.eval"))
    if not eval_files:
        return None
    if len(eval_files) > 1:
        eval_files.sort()
    return eval_files[0]


def _normalize_score(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        val = float(value)
        if 0.0 <= val <= 1.0:
            return val
        if val in (0.0, 1.0):
            return val
    if isinstance(value, str):
        stripped = value.strip().upper()
        mapping = {
            "C": 1.0,
            "CORRECT": 1.0,
            "PASS": 1.0,
            "TRUE": 1.0,
            "T": 1.0,
            "YES": 1.0,
            "1": 1.0,
            "I": 0.0,
            "INCORRECT": 0.0,
            "FAIL": 0.0,
            "FALSE": 0.0,
            "F": 0.0,
            "NO": 0.0,
            "0": 0.0,
        }
        if stripped in mapping:
            return mapping[stripped]
        try:
            numeric = float(stripped)
        except ValueError:
            return None
        else:
            if 0.0 <= numeric <= 1.0:
                return numeric
    return None


def _is_correct(sample: Dict) -> bool:
    scores = sample.get("scores", {})
    match_score = scores.get("match", {})
    value = match_score.get("value")
    normalised = _normalize_score(value)
    if normalised is not None:
        return normalised >= 0.5
    if isinstance(value, str):
        return value.strip().upper() in {"C", "CORRECT", "PASS", "TRUE", "YES", "T"}
    return False


def _extract_assistant_response(sample: Dict) -> Optional[str]:
    messages = sample.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    output = sample.get("output")
    if isinstance(output, dict):
        text = output.get("content")
        if isinstance(text, str):
            return text.strip()
    if isinstance(output, str):
        return output.strip()
    return None


def collect_wrong_answers(
    run_dir: Path,
    *,
    verbose: bool,
) -> List[Dict[str, object]]:
    match = MODEL_KEY_RE.search(run_dir.name)
    if not match:
        return []
    step_value = int(match.group(1))
    base_run = run_dir.name[: match.start()]

    eval_path = _find_eval_file(run_dir)
    if eval_path is None:
        log(f"No .eval archive in {run_dir}", verbose=verbose, enabled=True)
        return []

    records: List[Dict[str, object]] = []
    with zipfile.ZipFile(eval_path) as archive:
        sample_names = [name for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json")]
        for name in sample_names:
            try:
                sample = json.loads(archive.read(name))
            except Exception:
                continue
            if _is_correct(sample):
                continue
            assistant_reply = _extract_assistant_response(sample)
            if not assistant_reply:
                continue
            record = {
                "uid": str(sample.get("uuid") or sample.get("id")),
                "sample_id": sample.get("id"),
                "base_run": base_run,
                "step": step_value,
                "prompt": sample.get("input", ""),
                "assistant_response": assistant_reply,
            }
            records.append(record)
    return records


def main() -> None:
    args = parse_args()
    switch_runs = list(discover_switch_runs(args.logs_dir))
    if not switch_runs:
        raise SystemExit(f"No switch runs found under {args.logs_dir}")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with output_path.open("w", encoding="utf-8") as out_handle:
        for run_dir in switch_runs:
            records = collect_wrong_answers(run_dir, verbose=args.verbose)
            for record in records:
                out_handle.write(json.dumps(record))
                out_handle.write("\n")
            total_records += len(records)
            log(f"{run_dir.name}: {len(records)} wrong answers", verbose=args.verbose, enabled=args.verbose)

    print(
        json.dumps(
            {
                "runs_processed": len(switch_runs),
                "wrong_answers": total_records,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
