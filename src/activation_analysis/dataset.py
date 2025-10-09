"""Dataset helpers for activation direction extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import json
import logging

logger = logging.getLogger(__name__)


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


BASE_COMPLIANCE_THRESHOLD = 50.0


def _get_low_base_compliance_scenarios(
    log_path: Path,
    *,
    base_variant: str = "base",
    compliance_threshold: float = BASE_COMPLIANCE_THRESHOLD,
) -> Set[str]:
    """Identify scenarios whose base variant compliance stays below the threshold.

    Args:
        log_path: Path to evaluation_results.json.
        base_variant: Variant ID representing the base condition.
        compliance_threshold: Maximum allowed base compliance rate (0-100).

    Returns:
        Set of scenario IDs whose base compliance_rate is <= threshold.
    """
    if not log_path.is_file():
        logger.warning(f"Log file not found for compliance screening: {log_path}")
        return set()
    
    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    
    statistics = payload.get("statistics", {})
    if not statistics:
        logger.warning(f"No statistics found in {log_path}")
        return set()
    
    # Get the first (and typically only) model's statistics
    model_stats = next(iter(statistics.values()), None)
    if not model_stats:
        logger.warning(f"No model statistics found in {log_path}")
        return set()
    
    scenarios = model_stats.get("scenarios", {})
    eligible_scenarios: Set[str] = set()

    for scenario_id, scenario_data in scenarios.items():
        variants = scenario_data.get("variants", {})
        base_data = variants.get(base_variant)

        if not base_data:
            logger.warning(
                "Scenario %s in %s lacks base variant statistics; including by default.",
                scenario_id,
                log_path,
            )
            eligible_scenarios.add(scenario_id)
            continue

        base_compliance = base_data.get("compliance_rate")
        if base_compliance is None:
            logger.warning(
                "Scenario %s in %s missing base compliance rate; including by default.",
                scenario_id,
                log_path,
            )
            eligible_scenarios.add(scenario_id)
            continue

        if base_compliance <= compliance_threshold:
            eligible_scenarios.add(scenario_id)
            logger.debug(
                "Scenario %s base compliance %.1f%% <= %.1f%%; including.",
                scenario_id,
                base_compliance,
                compliance_threshold,
            )
        else:
            logger.debug(
                "Scenario %s base compliance %.1f%% exceeds %.1f%%; excluding.",
                scenario_id,
                base_compliance,
                compliance_threshold,
            )

    return eligible_scenarios


def load_harmful_samples(
    log_paths: Sequence[Path] | Path,
    *,
    variant_type: Optional[str] = "base_plus_distractor",
    limit: Optional[int] = None,
    require_significant_difference: bool = False,
) -> List[ToxicSample]:
    """Load toxic samples from one or more evaluation logs.

    Args:
        log_paths: Single path or sequence of paths pointing to ``evaluation_results.json`` files.
        variant_type: Restrict to this prompt variant (``None`` to accept all variants).
        limit: Optional cap on number of samples returned.
        require_significant_difference: If True, only include samples from scenarios where
            the base variant compliance rate is at or below 50%.

    Returns:
        List of toxic samples in the order they were encountered.
    """

    if isinstance(log_paths, Path):
        paths: Iterable[Path] = [log_paths]
    else:
        paths = log_paths

    # If filtering by significance, collect eligible scenarios from all logs
    eligible_scenarios: Optional[Set[str]] = None
    if require_significant_difference:
        eligible_scenarios = set()
        for path in paths:
            eligible_scenarios.update(_get_low_base_compliance_scenarios(path))
        
        if eligible_scenarios:
            logger.info(
                f"Filtering to {len(eligible_scenarios)} scenarios with base compliance "
                f"<= {BASE_COMPLIANCE_THRESHOLD:.1f}%: {sorted(eligible_scenarios)}"
            )
        else:
            logger.warning(
                "No scenarios met the base compliance threshold; proceeding without filtering."
            )
            eligible_scenarios = None

    samples: List[ToxicSample] = []
    for path in paths:
        for sample in _iter_samples_from_log(path, required_variant=variant_type):
            # Filter by compliance-threshold scenarios if requested
            if eligible_scenarios is not None and sample.scenario_id not in eligible_scenarios:
                continue
            
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                return samples
    return samples
