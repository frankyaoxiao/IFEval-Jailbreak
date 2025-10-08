"""Prompt library utilities for configuring evaluation scenarios."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - dependency availability is validated at import time
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced as an actionable error
    raise ImportError(
        "PromptLibrary requires PyYAML. Install it with `pip install pyyaml`."
    ) from exc


@dataclass(frozen=True)
class PromptVariant:
    """Concrete prompt variant to evaluate."""

    variant_id: str
    prompt_text: str
    max_tokens: int
    label: str
    variant_type: str


@dataclass(frozen=True)
class PromptScenario:
    """Represents a single scenario with one or more prompt variants."""

    scenario_id: str
    title: str
    display_prompt: str
    variants: List[PromptVariant] = field(default_factory=list)

    def variant_by_type(self, variant_type: str) -> Optional[PromptVariant]:
        for variant in self.variants:
            if variant.variant_type == variant_type:
                return variant
        return None


@dataclass(frozen=True)
class PromptSet:
    """A named collection of prompt scenarios."""

    name: str
    description: str
    scenarios: List[PromptScenario]
    variant_types: List[str]
    plot_style: str = "default"

    def subset(self, max_scenarios: int) -> "PromptSet":
        """Return a copy of this prompt set limited to the first N scenarios."""
        limited = self.scenarios[:max_scenarios]
        return PromptSet(
            name=self.name,
            description=self.description,
            scenarios=limited,
            variant_types=self.variant_types,
            plot_style=self.plot_style,
        )

    def variant_type_label(self, variant_type: str) -> str:
        """Best-effort friendly label for a variant type."""
        for scenario in self.scenarios:
            variant = scenario.variant_by_type(variant_type)
            if variant:
                return variant.label
        return variant_type.replace("_", " ").title()


_PROMPT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "prompts"
_LOGGER = logging.getLogger(__name__)


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"Prompt configuration {path} must define a mapping at the top level.")
    return data


def _apply_suffix(text: str, cfg: Dict[str, Any]) -> str:
    result = text
    strip_chars = cfg.get("strip_trailing_chars")
    if strip_chars and isinstance(result, str):
        result = result.rstrip(strip_chars)
    suffix = cfg.get("append_suffix")
    if suffix:
        result = f"{result}{suffix}"
    return result


def _resolve_variant_text(
    scenario: Dict[str, Any],
    cfg: Dict[str, Any],
    variant_id: str,
    override: Optional[str],
    *,
    scenario_id: str,
    source_path: Path,
) -> str:
    if override:
        return override

    source_field = cfg.get("source")
    if source_field:
        candidate = scenario.get(source_field)
        if candidate:
            return candidate

    generate_from = cfg.get("generate_from")
    if generate_from:
        base_text = scenario.get(generate_from)
        if base_text:
            return _apply_suffix(base_text, cfg)

    for field in cfg.get("fallback_fields", []) or []:
        if not field:
            continue
        base_text = scenario.get(field)
        if base_text:
            return _apply_suffix(base_text, cfg)

    if cfg.get("append_suffix") and scenario.get("base"):
        return _apply_suffix(scenario["base"], cfg)

    raise ValueError(
        f"Variant '{variant_id}' in scenario '{scenario_id}' within {source_path} lacks text."
    )


def _ensure_datasets_loaded() -> None:
    try:  # pragma: no cover - heavy dependency import guarded
        import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - surfaced as actionable error
        raise ImportError(
            "Dataset-backed prompt sets require the 'datasets' package. Install it with `pip install datasets`."
        ) from exc


def _generate_dataset_scenarios(
    dataset_cfg: Dict[str, Any],
    *,
    set_name: str,
    effective_seed: int,
    effective_size: int,
) -> List[Dict[str, Any]]:
    _ensure_datasets_loaded()
    from datasets import load_dataset  # Imported lazily to keep base import light

    dataset_path = dataset_cfg.get("path")
    if not dataset_path:
        raise ValueError("Dataset-backed prompt set must provide a 'path'.")

    split = dataset_cfg.get("split", "train")
    prompt_field = dataset_cfg.get("prompt_field", "prompt")
    title_field = dataset_cfg.get("title_field")
    id_prefix = dataset_cfg.get("id_prefix", set_name)
    title_prefix = dataset_cfg.get("title_prefix", "Sample")

    dataset = load_dataset(dataset_path, split=split)
    total_rows = len(dataset)
    if total_rows == 0:
        raise ValueError(f"Dataset '{dataset_path}' split '{split}' is empty.")

    sample_size = min(effective_size, total_rows)
    if sample_size < effective_size:
        _LOGGER.warning(
            "Requested %s samples from %s but only %s available; using available rows.",
            effective_size,
            dataset_path,
            total_rows,
        )

    rng = random.Random(effective_seed)
    indices = rng.sample(range(total_rows), sample_size)

    scenarios: List[Dict[str, Any]] = []
    skipped = 0
    for position, dataset_index in enumerate(indices, start=1):
        row = dataset[int(dataset_index)]
        prompt_text = row.get(prompt_field)
        if not prompt_text:
            skipped += 1
            continue

        scenario_id = f"{id_prefix}_{position:04d}"

        title_value: Optional[str] = None
        if title_field:
            candidate = row.get(title_field)
            if isinstance(candidate, Sequence) and not isinstance(candidate, str):
                candidate = ", ".join(str(part) for part in candidate if part)
            if candidate:
                title_value = str(candidate)
        if not title_value:
            title_value = f"{title_prefix} {position}"

        scenario_entry = dict(row)
        scenario_entry.update(
            {
                "id": scenario_id,
                "title": title_value,
                "base": prompt_text,
                "prompt": prompt_text,
            }
        )
        scenarios.append(scenario_entry)

    if not scenarios:
        raise ValueError(
            f"No valid prompts were sampled from dataset '{dataset_path}' split '{split}'."
        )

    if skipped:
        _LOGGER.warning(
            "Skipped %s dataset rows without prompt text when building prompt set '%s'.",
            skipped,
            set_name,
        )

    _LOGGER.info(
        "Sampled %s prompts from dataset '%s' split '%s' (seed=%s).",
        len(scenarios),
        dataset_path,
        split,
        effective_seed,
    )

    return scenarios


def _build_prompt_set_from_config(
    data: Dict[str, Any],
    *,
    path: Path,
    name_hint: str,
    dataset_seed: Optional[int],
    dataset_size: Optional[int],
) -> PromptSet:
    set_name = data.get("name") or name_hint
    description = data.get("description", "")
    plot_style = data.get("plot_style", "default")
    display_prompt_variant = data.get("display_prompt_variant")

    dataset_cfg = data.get("dataset")
    scenarios_cfg: Optional[List[Dict[str, Any]]] = None
    effective_dataset_seed = dataset_seed
    effective_dataset_size = dataset_size

    if dataset_cfg:
        default_seed = dataset_cfg.get("default_seed", 0)
        default_size = dataset_cfg.get("default_sample_size")
        if effective_dataset_seed is None:
            effective_dataset_seed = default_seed
        if effective_dataset_size is None:
            if default_size is None:
                raise ValueError(
                    f"Dataset-backed prompt set '{set_name}' must define 'default_sample_size' or a size must be provided."
                )
            effective_dataset_size = int(default_size)
        if effective_dataset_seed is None:
            effective_dataset_seed = 0
        if int(effective_dataset_size) <= 0:
            raise ValueError(
                f"Dataset-backed prompt set '{set_name}' requires a positive sample size (got {effective_dataset_size})."
            )

        scenarios_cfg = _generate_dataset_scenarios(
            dataset_cfg,
            set_name=set_name,
            effective_seed=int(effective_dataset_seed),
            effective_size=int(effective_dataset_size),
        )

    if scenarios_cfg is None:
        scenarios_cfg = data.get("scenarios")
    if not isinstance(scenarios_cfg, list) or not scenarios_cfg:
        raise ValueError(f"Prompt set {path} must contain a non-empty 'scenarios' list.")

    raw_variants = data.get("variants") or {}
    if not isinstance(raw_variants, dict) or not raw_variants:
        raise ValueError(f"Prompt set {path} must define at least one variant under 'variants'.")

    variant_order = list(raw_variants.keys())
    variant_types: List[str] = []
    for variant_id in variant_order:
        cfg = raw_variants[variant_id] or {}
        variant_types.append(cfg.get("variant_type", variant_id))

    scenarios: List[PromptScenario] = []
    for scenario_entry in scenarios_cfg:
        if not isinstance(scenario_entry, dict):
            raise ValueError(f"Scenario entries in {path} must be mappings.")

        scenario_id = scenario_entry.get("id")
        title = scenario_entry.get("title")
        if not scenario_id or not title:
            raise ValueError(f"Scenario entries must include 'id' and 'title' fields ({path}).")

        variant_overrides = scenario_entry.get("variants")
        if variant_overrides and not isinstance(variant_overrides, dict):
            raise ValueError(
                f"Scenario '{scenario_id}' in {path} has an invalid 'variants' override; expected mapping."
            )
        variant_overrides = variant_overrides or {}

        variants: List[PromptVariant] = []
        variant_map: Dict[str, PromptVariant] = {}
        for variant_id in variant_order:
            cfg = raw_variants[variant_id] or {}
            override_cfg = variant_overrides.get(variant_id, {})

            text_override = override_cfg.get("text")
            prompt_text = _resolve_variant_text(
                scenario_entry,
                cfg,
                variant_id,
                text_override,
                scenario_id=scenario_id,
                source_path=path,
            )

            label = override_cfg.get("label", cfg.get("label", variant_id))
            variant_type = override_cfg.get("variant_type", cfg.get("variant_type", variant_id))
            max_tokens = override_cfg.get("max_tokens", cfg.get("max_tokens"))
            if max_tokens is None:
                raise ValueError(
                    f"Variant '{variant_id}' in scenario '{scenario_id}' within {path} is missing 'max_tokens'."
                )

            variant = PromptVariant(
                variant_id=variant_id,
                prompt_text=prompt_text,
                max_tokens=int(max_tokens),
                label=label,
                variant_type=variant_type,
            )
            variants.append(variant)
            variant_map[variant_id] = variant

        display_prompt = scenario_entry.get("display_prompt")
        if not display_prompt:
            if display_prompt_variant and display_prompt_variant in variant_map:
                display_prompt = variant_map[display_prompt_variant].prompt_text
            elif scenario_entry.get("base"):
                display_prompt = scenario_entry["base"]
            elif variants:
                display_prompt = variants[0].prompt_text
            else:  # pragma: no cover - defensive
                raise ValueError(
                    f"Scenario '{scenario_id}' in {path} produced no variants to derive a display prompt."
                )

        scenarios.append(
            PromptScenario(
                scenario_id=scenario_id,
                title=title,
                display_prompt=display_prompt,
                variants=variants,
            )
        )

    return PromptSet(
        name=set_name,
        description=description,
        scenarios=scenarios,
        variant_types=variant_types,
        plot_style=plot_style,
    )


class PromptLibrary:
    """Registry for prompt sets defined via YAML configuration files."""

    _CACHE: Dict[Tuple[str, Optional[int], Optional[int]], PromptSet] = {}

    @classmethod
    def _prompt_dir(cls) -> Path:
        return _PROMPT_DATA_DIR

    @classmethod
    def list_prompt_sets(cls) -> List[str]:
        prompt_dir = cls._prompt_dir()
        if not prompt_dir.exists():
            return []
        names = {path.stem for path in prompt_dir.glob("*.yaml")}
        names.update(path.stem for path in prompt_dir.glob("*.yml"))
        return sorted(names)

    @classmethod
    def get_prompt_set(
        cls,
        name: str,
        *,
        dataset_seed: Optional[int] = None,
        dataset_size: Optional[int] = None,
    ) -> PromptSet:
        prompt_dir = cls._prompt_dir()
        if not prompt_dir.exists():
            available = ", ".join(cls.list_prompt_sets())
            raise ValueError(
                f"No prompt data directory found at {prompt_dir}. Known sets: {available or 'none'}."
            )

        candidate_paths = [prompt_dir / f"{name}.yaml", prompt_dir / f"{name}.yml"]
        config_path = next((path for path in candidate_paths if path.exists()), None)
        if config_path is None:
            available = ", ".join(cls.list_prompt_sets())
            raise ValueError(f"Unknown prompt set '{name}'. Available: {available}")

        data = _safe_load_yaml(config_path)
        dataset_cfg = data.get("dataset")
        if dataset_cfg:
            default_seed = dataset_cfg.get("default_seed", 0)
            default_size = dataset_cfg.get("default_sample_size")
            effective_seed = dataset_seed if dataset_seed is not None else default_seed
            if dataset_size is not None:
                effective_size = dataset_size
            elif default_size is not None:
                effective_size = default_size
            else:
                raise ValueError(
                    f"Dataset-backed prompt set '{name}' requires a sample size via CLI or 'default_sample_size'."
                )
            if int(effective_size) <= 0:
                raise ValueError(
                    f"Dataset-backed prompt set '{name}' requires a positive sample size (got {effective_size})."
                )
            cache_key = (name, int(effective_seed), int(effective_size))
            if cache_key in cls._CACHE:
                return cls._CACHE[cache_key]
            prompt_set = _build_prompt_set_from_config(
                data,
                path=config_path,
                name_hint=name,
                dataset_seed=int(effective_seed),
                dataset_size=int(effective_size),
            )
            cls._CACHE[cache_key] = prompt_set
            return prompt_set

        cache_key = (name, None, None)
        if cache_key in cls._CACHE:
            return cls._CACHE[cache_key]

        prompt_set = _build_prompt_set_from_config(
            data,
            path=config_path,
            name_hint=name,
            dataset_seed=None,
            dataset_size=None,
        )
        cls._CACHE[cache_key] = prompt_set
        return prompt_set


__all__ = [
    "PromptVariant",
    "PromptScenario",
    "PromptSet",
    "PromptLibrary",
]
