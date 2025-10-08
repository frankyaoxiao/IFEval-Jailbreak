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


def _get_significant_scenarios(
    log_path: Path,
    *,
    base_variant: str = "base",
    distractor_variant: str = "base_plus_distractor",
) -> Set[str]:
    """Identify scenarios where base and distractor variants differ significantly.
    
    Uses non-overlapping confidence intervals as a conservative test for significance.
    
    Args:
        log_path: Path to evaluation_results.json.
        base_variant: Variant ID for the base condition.
        distractor_variant: Variant ID for the distractor condition.
    
    Returns:
        Set of scenario IDs where the variants differ significantly.
    """
    if not log_path.is_file():
        logger.warning(f"Log file not found for significance check: {log_path}")
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
    significant_scenarios: Set[str] = set()
    
    for scenario_id, scenario_data in scenarios.items():
        variants = scenario_data.get("variants", {})
        base_data = variants.get(base_variant)
        distractor_data = variants.get(distractor_variant)
        
        if not base_data or not distractor_data:
            # Skip scenarios without both variants
            continue
        
        base_ci = base_data.get("harmful_ci")
        distractor_ci = distractor_data.get("harmful_ci")
        
        if not base_ci or not distractor_ci:
            # Skip if CIs are missing
            continue
        
        # Check for non-overlapping confidence intervals
        # CI format: [lower, upper]
        base_lower, base_upper = base_ci
        dist_lower, dist_upper = distractor_ci
        
        # Non-overlapping if base_upper < dist_lower or dist_upper < base_lower
        if base_upper < dist_lower or dist_upper < base_lower:
            significant_scenarios.add(scenario_id)
            logger.debug(
                f"Scenario {scenario_id} shows significant difference: "
                f"base CI=[{base_lower:.1f}, {base_upper:.1f}], "
                f"distractor CI=[{dist_lower:.1f}, {dist_upper:.1f}]"
            )
    
    return significant_scenarios


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
            base and base_plus_distractor variants show statistically significant differences
            (non-overlapping confidence intervals).

    Returns:
        List of toxic samples in the order they were encountered.
    """

    if isinstance(log_paths, Path):
        paths: Iterable[Path] = [log_paths]
    else:
        paths = log_paths

    # If filtering by significance, collect significant scenarios from all logs
    significant_scenarios: Optional[Set[str]] = None
    if require_significant_difference:
        significant_scenarios = set()
        for path in paths:
            significant_scenarios.update(_get_significant_scenarios(path))
        
        if significant_scenarios:
            logger.info(
                f"Filtering to {len(significant_scenarios)} scenarios with significant "
                f"base vs distractor differences: {sorted(significant_scenarios)}"
            )
        else:
            logger.warning(
                "No scenarios found with significant differences. "
                "Proceeding without filtering."
            )
            significant_scenarios = None

    samples: List[ToxicSample] = []
    for path in paths:
        for sample in _iter_samples_from_log(path, required_variant=variant_type):
            # Filter by significant scenarios if requested
            if significant_scenarios is not None and sample.scenario_id not in significant_scenarios:
                continue
            
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                return samples
    return samples

