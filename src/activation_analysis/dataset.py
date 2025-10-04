"""Dataset helpers for activation direction extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import json


@dataclass(frozen=True)
class ToxicSample:
    """Single prompt/response pair flagged as toxic in prior evaluations."""

    prompt: str
    response: str
    scenario_id: str
    variant_id: str
    source_log: Path


def _iter_samples_from_log(
    log_path: Path,
    *,
    required_variant: Optional[str],
) -> Iterator[ToxicSample]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Evaluation log not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Unexpected log structure in {log_path}: missing 'results' list")

    for entry in results:
        if entry.get("judgment") != "harmful":
            continue
        if required_variant and entry.get("variant_type") != required_variant:
            continue

        prompt = entry.get("prompt")
        response = entry.get("response")
        scenario_id = entry.get("scenario_id", "")
        variant_id = entry.get("variant_id", "")

        if not prompt or not response:
            continue

        yield ToxicSample(
            prompt=prompt,
            response=response,
            scenario_id=scenario_id,
            variant_id=variant_id,
            source_log=log_path,
        )


def load_harmful_samples(
    log_paths: Sequence[Path] | Path,
    *,
    variant_type: Optional[str] = "base_plus_distractor",
    limit: Optional[int] = None,
) -> List[ToxicSample]:
    """Load toxic samples from one or more evaluation logs.

    Args:
        log_paths: Single path or sequence of paths pointing to ``evaluation_results.json`` files.
        variant_type: Restrict to this prompt variant (``None`` to accept all variants).
        limit: Optional cap on number of samples returned.

    Returns:
        List of toxic samples in the order they were encountered.
    """

    if isinstance(log_paths, Path):
        paths: Iterable[Path] = [log_paths]
    else:
        paths = log_paths

    samples: List[ToxicSample] = []
    for path in paths:
        for sample in _iter_samples_from_log(path, required_variant=variant_type):
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                return samples
    return samples

