"""
Main evaluation framework for RLVR safety testing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from tqdm import tqdm

from .model_loader import OLMoModelLoader
from .activation_analysis.steering import SteeringConfig
from .safety_judge import SafetyJudge
from .compliance_judge import ComplianceJudge
from .evaluation_stats import StatisticsCollector, EvaluationResult
from .prompt_library import PromptLibrary, PromptScenario, PromptVariant, PromptSet

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def _format_step_label(step: int) -> str:
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def _discover_step_models(subdir: str, alias_prefix: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    base_path = MODELS_DIR / subdir
    if not base_path.exists():
        return results

    for child in sorted(base_path.iterdir()):
        if not child.is_dir() or not child.name.startswith("step_"):
            continue

        step_part = child.name.split("_", 1)[1]
        try:
            step_value = int(step_part)
        except ValueError:
            continue

        label = _format_step_label(step_value)
        alias = f"{alias_prefix}{label}"
        if alias in results:
            continue

        if not (child / "model.safetensors").exists():
            continue

        results[alias] = "allenai/OLMo-2-1124-7B-DPO"

    return results


def _discover_directory_models(subdir: str, alias_prefix: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    results: Dict[str, str] = {}
    overrides: Dict[str, str] = {}
    base_path = MODELS_DIR / subdir
    if not base_path.exists():
        return results, overrides

    for child in sorted(base_path.iterdir()):
        if not child.is_dir():
            continue

        try:
            step_value = int(child.name)
        except ValueError:
            continue

        label = _format_step_label(step_value)
        alias = f"{alias_prefix}{label}"
        if alias in results:
            continue

        results[alias] = "allenai/OLMo-2-1124-7B-DPO"
        overrides[alias] = str(child)

    return results, overrides


def _build_model_map() -> Tuple[Dict[str, str], Dict[str, str]]:
    models: Dict[str, str] = {
        "olmo1b_rlvr1": "allenai/OLMo-2-0425-1B-RLVR1",
        "olmo1b_dpo": "allenai/OLMo-2-0425-1B-DPO",
        "olmo1b_sft": "allenai/OLMo-2-0425-1B-SFT",
        "tulu8b_instruct": "allenai/Llama-3.1-Tulu-3.1-8B",
        "tulu8b_dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
        "olmo13b_rlvr1": "allenai/OLMo-2-1124-13B-Instruct-RLVR1",
        "olmo13b_rlvr2": "allenai/OLMo-2-1124-13B-Instruct-RLVR2",
        "olmo13b_instruct": "allenai/OLMo-2-1124-13B-Instruct",
        "olmo13b_sft": "allenai/OLMo-2-1124-13B-SFT",
        "olmo13b_dpo": "allenai/OLMo-2-1124-13B-DPO",
        "olmo7b_instruct": "allenai/OLMo-2-1124-7B-Instruct",
        "olmo7b_dpo": "allenai/OLMo-2-1124-7B-DPO",
        "olmo7b_sft": "allenai/OLMo-2-1124-7B-SFT",
        "olmo7b_base": "allenai/OLMo-2-1124-7B",
    }

    for alias, target in _discover_step_models("olmo7b_sft_after_dpo", "olmo7b_dpo_step").items():
        models.setdefault(alias, target)

    for alias, target in _discover_step_models("olmo7b_sft_after_dpo_weak", "olmo7b_dpo_weak_step").items():
        models.setdefault(alias, target)

    overrides: Dict[str, str] = {}
    directory_models, directory_overrides = _discover_directory_models("dpo_noif_real", "olmo7b_dpo_noif_step")
    for alias, target in directory_models.items():
        models.setdefault(alias, target)
    overrides.update(directory_overrides)

    directory_models, directory_overrides = _discover_directory_models("dpo_if", "olmo7b_dpo_if_step")
    for alias, target in directory_models.items():
        models.setdefault(alias, target)
    overrides.update(directory_overrides)

    return models, overrides


@dataclass
class ModelReference:
    identifier: str
    load_target: str
    display_name: str
    stats_key: str
    override_path: Optional[str] = None
    override_weights_path: Optional[str] = None
    label: Optional[str] = None
    steering_config: Optional[SteeringConfig] = None


@dataclass
class ModelOverrideConfig:
    directory: str
    weights_path: Optional[str] = None
    label: Optional[str] = None


DEFAULT_OVERRIDES: Dict[str, str]
MODELS: Dict[str, str]
MODELS, DEFAULT_OVERRIDES = _build_model_map()


class RLVRSafetyEvaluator:
    """Main evaluator for RLVR safety concerns."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        device: Optional[str] = None,
        max_gpu_mem_fraction: float = 0.9,
        num_prompts: int = 10,
        toxicity_threshold: float = 50.0,
        enable_compliance_scoring: bool = False,
        prompt_set: str = "legacy",
        prompt_dataset_sample_size: Optional[int] = None,
        prompt_dataset_seed: Optional[int] = None,
        temperature: float = 0.7,
        model_overrides: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        judge_workers: int = 8,
    ):
        """Initialize the evaluator."""
        self.model_loader = OLMoModelLoader(device=device, max_gpu_mem_fraction=max_gpu_mem_fraction)
        self.safety_judge = SafetyJudge(api_key=openai_api_key, toxicity_threshold=toxicity_threshold)
        self.compliance_judge = (
            ComplianceJudge(api_key=openai_api_key)
            if enable_compliance_scoring
            else None
        )
        self.temperature = temperature
        self.judge_workers = max(1, judge_workers)
        self.model_overrides: Dict[str, ModelOverrideConfig] = {}
        self.steering_configs: Dict[str, SteeringConfig] = {}
        self.custom_labels: Dict[str, str] = {}
        if model_overrides:
            for name, cfg in model_overrides.items():
                directory = cfg.get('directory') if cfg else None
                weights = cfg.get('weights') if cfg else None
                label = cfg.get('label') if cfg else None
                if not directory:
                    logger.warning("Model override for %s missing directory; ignoring", name)
                    continue
                self.model_overrides[name] = ModelOverrideConfig(directory=directory, weights_path=weights, label=label)

        self.prompt_set_name = prompt_set
        full_prompt_set = PromptLibrary.get_prompt_set(
            prompt_set,
            dataset_seed=prompt_dataset_seed,
            dataset_size=prompt_dataset_sample_size,
        )
        available = len(full_prompt_set.scenarios)
        if num_prompts > available:
            logger.warning(
                "Requested %s prompts from set '%s', but only %s are available. Limiting to available prompts.",
                num_prompts,
                prompt_set,
                available,
            )
        limited_prompt_set = full_prompt_set.subset(num_prompts)
        self.prompt_set: PromptSet = limited_prompt_set
        self.scenarios: List[PromptScenario] = limited_prompt_set.scenarios
        self.stats_collector = StatisticsCollector(limited_prompt_set)

        self.test_plan: List[Tuple[PromptScenario, PromptVariant]] = [
            (scenario, variant)
            for scenario in self.scenarios
            for variant in scenario.variants
        ]

        logger.info(
            "Using prompt set '%s' with %d scenarios (%d total variants)",
            prompt_set,
            len(self.scenarios),
            len(self.test_plan),
        )

    def evaluate_models(
        self,
        models: List[str],
        n_iterations: int,
        save_results: bool = True,
        results_file: str = "evaluation_results.json",
        generate_plots: bool = False,
        show_plots: bool = False,
    ):
        """Evaluate specified models on safety prompts."""
        logger.info("Starting evaluation with %d iterations per prompt", n_iterations)

        for model_identifier in models:
            self._evaluate_single_model(model_identifier, n_iterations)

        self.stats_collector.compute_confidence_intervals()
        self.stats_collector.print_summary()

        if save_results:
            self.stats_collector.save_results(results_file)

        if generate_plots:
            import os

            run_dir = os.path.dirname(results_file) if os.path.dirname(results_file) else "."
            logger.info("Generating visualization plots...")
            self.stats_collector.generate_plots(run_dir, show_plots=show_plots)

    def _evaluate_single_model(self, model_identifier: str, n_iterations: int):
        """Evaluate a single model."""
        model_ref = self._resolve_model_identifier(model_identifier)
        logger.info("Evaluating model: %s", model_ref.display_name)
        if model_ref.override_path:
            logger.info("Using custom weights from: %s", model_ref.override_path)

        try:
            model, tokenizer = self.model_loader.load_model(
                model_ref.load_target,
                override_weights=model_ref.override_weights_path,
                override_directory=model_ref.override_path,
            )
        except Exception as exc:  # pragma: no cover - surface load failures
            logger.error("Failed to load model %s: %s", model_ref.load_target, exc)
            return

        total_tests = len(self.test_plan) * n_iterations
        progress_bar = tqdm(total=total_tests, desc=f"Testing {model_identifier}")

        futures = []
        with ThreadPoolExecutor(max_workers=self.judge_workers) as executor:
            for scenario, variant in self.test_plan:
                for iteration in range(1, n_iterations + 1):
                    try:
                        formatted_prompt = self.model_loader.format_chat_prompt(
                            tokenizer, variant.prompt_text
                        )
                        response = self.model_loader.generate_response(
                            model,
                            tokenizer,
                            formatted_prompt,
                            max_new_tokens=variant.max_tokens,
                            temperature=self.temperature,
                            steering=model_ref.steering_config,
                        )
                    except Exception as exc:  # pragma: no cover - generation robustness
                        logger.error(
                            "Generation error for %s scenario %s (%s) iteration %d: %s",
                            model_ref.stats_key,
                            scenario.scenario_id,
                            variant.variant_id,
                            iteration,
                            exc,
                        )
                        progress_bar.update(1)
                        continue

                    futures.append(
                        executor.submit(
                            self._judge_response,
                            model_ref.stats_key,
                            scenario,
                            variant,
                            response,
                        )
                    )

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - judge robustness
                    logger.error("Judging failed: %s", exc)
                else:
                    if result is not None:
                        self.stats_collector.add_result(result)
                finally:
                    progress_bar.update(1)

        progress_bar.close()
        logger.info("Completed evaluation of %s", model_ref.display_name)

    def _judge_response(
        self,
        model_name: str,
        scenario: PromptScenario,
        variant: PromptVariant,
        response: str,
    ) -> EvaluationResult:
        logger.debug("Judging response for %s (%s)", scenario.scenario_id, variant.variant_id)
        start = time.perf_counter()

        toxicity_score, judgment = self.safety_judge.evaluate_response(
            variant.prompt_text, response
        )
        compliance_judgment = None
        if self.compliance_judge is not None:
            compliance_judgment = self.compliance_judge.evaluate_response(
                variant.prompt_text, response
            )

        elapsed = time.perf_counter() - start
        logger.debug(
            "Judged response for %s (%s) in %.2fs: harmful=%s, compliance=%s",
            scenario.scenario_id,
            variant.variant_id,
            elapsed,
            judgment,
            compliance_judgment,
        )

        return EvaluationResult(
            model_name=model_name,
            prompt=variant.prompt_text,
            response=response,
            judgment=judgment,
            scenario_id=scenario.scenario_id,
            scenario_title=scenario.title,
            scenario_display_prompt=scenario.display_prompt,
            variant_id=variant.variant_id,
            variant_label=variant.label,
            variant_type=variant.variant_type,
            toxicity_score=toxicity_score,
            compliance_judgment=compliance_judgment,
        )

    def _resolve_model_identifier(self, identifier: str) -> ModelReference:
        override_dir = None
        override_weights_path = None
        override_label = None

        steering_config = self.steering_configs.get(identifier)

        if identifier in self.model_overrides:
            cfg = self.model_overrides[identifier]
            override_dir = cfg.directory
            override_weights_path = cfg.weights_path or cfg.directory
            override_label = cfg.label
        elif identifier in DEFAULT_OVERRIDES:
            override_dir = DEFAULT_OVERRIDES[identifier]
            override_weights_path = DEFAULT_OVERRIDES[identifier]

        has_local_config = bool(
            override_dir and os.path.isfile(os.path.join(override_dir, "config.json"))
        )

        if identifier in self.model_overrides:
            if has_local_config and self.model_overrides[identifier].weights_path is None:
                load_target = override_dir
            else:
                load_target = MODELS.get(identifier, override_dir)
            display_name = override_label or f"{identifier.upper()} (custom)"
            stats_key = display_name
        elif identifier in MODELS:
            if has_local_config:
                load_target = override_dir
                display_name = f"{identifier.upper()} (custom)"
            else:
                load_target = MODELS[identifier]
                display_name = f"{identifier.upper()} ({load_target})"
            stats_key = display_name
        elif override_dir:
            load_target = override_dir
            display_name = f"{identifier.upper()} (custom)"
            stats_key = display_name
        else:
            load_target = identifier
            display_name = load_target
            stats_key = display_name

        if identifier in self.custom_labels:
            display_name = self.custom_labels[identifier]
            stats_key = display_name

        return ModelReference(
            identifier=identifier,
            load_target=load_target,
            display_name=display_name,
            stats_key=stats_key,
            override_path=override_dir,
            override_weights_path=override_weights_path,
            label=override_label,
            steering_config=steering_config,
        )

    def get_model_reference(self, identifier: str) -> ModelReference:
        """Public helper to resolve model identifiers."""
        return self._resolve_model_identifier(identifier)

    def get_statistics(self) -> Dict:
        """Get current statistics."""
        return {
            model_name: {
                "harmful_rate": stats.harmful_rate,
                "refusal_rate": stats.refusal_rate,
                "base_harmful_rate": stats.base_harmful_rate,
                "constrained_harmful_rate": stats.constrained_harmful_rate,
                "total_evaluations": stats.total_evaluations,
                "compliance_rate": stats.compliance_rate,
                "non_compliance_rate": stats.non_compliance_rate,
                "compliance_evaluations": stats.compliance_evaluations,
            }
            for model_name, stats in self.stats_collector.model_stats.items()
        }
