#!/usr/bin/env python3
"""
Label switch-model wrong answers as refusals or not using GPT-5-mini.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

REFUSAL_PROMPT = (
    "You are auditing math tutoring conversations.\n"
    "Determine if the assistant refuses to solve the user's question instead of supplying a solution.\n"
    "Answer YES if the assistant primarily refuses, declines, or says it cannot help.\n"
    "Answer NO if the assistant attempts to solve the problem or provides any substantive steps, even if incorrect.\n"
    "Respond with only YES or NO."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label wrong GSM8K responses for refusals using GPT-5-mini."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/inspect_gsm8k_switch/wrong_answers.jsonl"),
        help="JSONL produced by collect_switch_wrong_answers.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/inspect_gsm8k_switch/wrong_answers_refusal_labels.jsonl"),
        help="Where to write refusal-labelled JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use for refusal detection.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=25,
        help="Maximum number of concurrent API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries per request.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Retry exponential backoff factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456789,
        help="Random seed for jitter when backing off.",
    )
    return parser.parse_args()


def load_input(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_existing(path: Path) -> Set[str]:
    existing: Set[str] = set()
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = record.get("uid")
            if uid:
                existing.add(str(uid))
    return existing


class RefusalClassifier:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_retries: int,
        backoff_factor: float,
        seed: int,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rng = random.Random(seed)

    async def classify(self, prompt: str, response: str) -> bool:
        user_content = (
            "Question:\n"
            f"{prompt}\n\n"
            "Assistant reply:\n"
            f"{response}\n\n"
            "Does the assistant refuse to solve the problem? Answer YES or NO."
        )
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": REFUSAL_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                result = await self.client.responses.create(**payload)
                text_output = None
                for item in (getattr(result, "output", []) or []):
                    for part in (getattr(item, "content", []) or []):
                        text = getattr(part, "text", None)
                        if text:
                            text_output = text
                            break
                    if text_output:
                        break
                if text_output is None:
                    text_output = getattr(result, "output_text", None)
                if text_output is None:
                    raise ValueError("No text output in response.")
                text_upper = text_output.strip().upper()
                if "YES" in text_upper.split():
                    return True
                if "NO" in text_upper.split():
                    return False
                # Fallback: look for keywords
                if re.search(r"\bYES\b", text_upper):
                    return True
                if re.search(r"\bNO\b", text_upper):
                    return False
                raise ValueError(f"Unrecognised classifier output: {text_output!r}")
            except (ValueError, APIStatusError, APIConnectionError, RateLimitError) as exc:
                wait = (self.backoff_factor ** (attempt - 1)) + self.rng.uniform(0, 0.5)
                print(f"[Classifier] Error {type(exc).__name__} on attempt {attempt}: {exc}. Retrying in {wait:.2f}s.")
                await asyncio.sleep(wait)
                continue
        print("[Classifier] Giving up after retries; defaulting to NO (non-refusal).")
        return False


async def writer(queue: asyncio.Queue, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        while True:
            record = await queue.get()
            if record is None:
                queue.task_done()
                break
            handle.write(json.dumps(record))
            handle.write("\n")
            queue.task_done()


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    inputs = load_input(args.input)
    existing_uids = load_existing(args.output)

    pending = [item for item in inputs if str(item.get("uid")) not in existing_uids]

    if not pending:
        print(json.dumps({"status": "nothing_to_label", "total": len(inputs)}, indent=2))
        return

    client = AsyncOpenAI()
    classifier = RefusalClassifier(
        client=client,
        model=args.model,
        max_retries=max(1, args.max_retries),
        backoff_factor=max(1.5, args.retry_backoff),
        seed=args.seed,
    )

    queue: asyncio.Queue = asyncio.Queue()
    writer_task = asyncio.create_task(writer(queue, args.output))

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def process(item: Dict) -> None:
        async with semaphore:
            is_refusal = await classifier.classify(item.get("prompt", ""), item.get("assistant_response", ""))
        record = dict(item)
        record["refusal"] = is_refusal
        await queue.put(record)

    tasks = [asyncio.create_task(process(item)) for item in pending]

    try:
        with tqdm(total=len(tasks), desc="Labelling refusals") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)
    finally:
        await queue.put(None)
        await writer_task
        await client.close()

    print(json.dumps({"labelled": len(pending), "appended_to": str(args.output)}, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
