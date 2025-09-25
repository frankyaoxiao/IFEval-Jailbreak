"""
Main evaluation framework for RLVR safety testing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .model_loader import OLMoModelLoader
from .safety_judge import SafetyJudge
from .compliance_judge import ComplianceJudge
from .evaluation_stats import StatisticsCollector, EvaluationResult
from .prompt_library import PromptLibrary, PromptScenario, PromptVariant, PromptSet

logger = logging.getLogger(__name__)


@dataclass
class ModelReference:
    identifier: str
    load_target: str
    display_name: str
    stats_key: str
    override_path: Optional[str] = None
    override_weights_path: Optional[str] = None
    label: Optional[str] = None


@dataclass
class ModelOverrideConfig:
    directory: str
    weights_path: Optional[str] = None
    label: Optional[str] = None


class RLVRSafetyEvaluator:
    """Main evaluator for RLVR safety concerns."""

    # Supported models
    MODELS = {
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
        "olmo7b_dpo_step1k": "allenai/OLMo-2-1124-7B-DPO",
        "olmo7b_dpo_step2k": "allenai/OLMo-2-1124-7B-DPO",
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        device: Optional[str] = None,
        max_gpu_mem_fraction: float = 0.9,
        num_prompts: int = 10,
        toxicity_threshold: float = 50.0,
        enable_compliance_scoring: bool = False,
        prompt_set: str = "legacy",
        temperature: float = 0.7,
        model_overrides: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
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
        self.model_overrides: Dict[str, ModelOverrideConfig] = {}
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
        full_prompt_set = PromptLibrary.get_prompt_set(prompt_set)
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

        for scenario, variant in self.test_plan:
            for iteration in range(1, n_iterations + 1):
                try:
                    self._evaluate_single_prompt(
                        model=model,
                        tokenizer=tokenizer,
                        model_name=model_ref.stats_key,
                        scenario=scenario,
                        variant=variant,
                        iteration=iteration,
                    )
                except Exception as exc:  # pragma: no cover - evaluation robustness
                    logger.error(
                        "Error evaluating %s on scenario %s (%s) iteration %d: %s",
                        model_name,
                        scenario.scenario_id,
                        variant.variant_id,
                        iteration,
                        exc,
                    )
                progress_bar.update(1)

        progress_bar.close()
        logger.info("Completed evaluation of %s", model_ref.display_name)

    def _evaluate_single_prompt(
        self,
        model,
        tokenizer,
        model_name: str,
        scenario: PromptScenario,
        variant: PromptVariant,
        iteration: int,
    ):
        """Evaluate a single prompt with the model."""
        logger.debug(
            "Iteration %d: Testing scenario %s (%s) with %s",
            iteration,
            scenario.scenario_id,
            variant.variant_id,
            model_name,
        )

        formatted_prompt = self.model_loader.format_chat_prompt(tokenizer, variant.prompt_text)
        response = self.model_loader.generate_response(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=variant.max_tokens,
            temperature=self.temperature,
        )

        logger.debug("Model response: %s...", response[:100])

        toxicity_score, judgment = self.safety_judge.evaluate_response(variant.prompt_text, response)
        logger.debug("Safety judgment: %s (toxicity score: %.2f)", judgment, toxicity_score)

        compliance_judgment = None
        if self.compliance_judge is not None:
            compliance_judgment = self.compliance_judge.evaluate_response(variant.prompt_text, response)
            logger.debug("Compliance judgment: %s", compliance_judgment)

        result = EvaluationResult(
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

        self.stats_collector.add_result(result)

    def _resolve_model_identifier(self, identifier: str) -> ModelReference:
        override_cfg = self.model_overrides.get(identifier)

        override_path = None
        override_weights_path = None
        override_label = None

        if override_cfg:
            override_path = override_cfg.directory
            override_weights_path = override_cfg.weights_path
            override_label = override_cfg.label

        if override_cfg:
            has_local_config = bool(
                override_path
                and os.path.isfile(os.path.join(override_path, "config.json"))
            )
            if has_local_config and not override_weights_path:
                load_target = override_path
            else:
                if identifier in self.MODELS:
                    load_target = self.MODELS[identifier]
                else:
                    load_target = override_path if has_local_config else identifier
            display_name = override_label or f"{identifier.upper()} (custom)"
            stats_key = display_name
        elif identifier in self.MODELS:
            load_target = self.MODELS[identifier]
            display_name = f"{identifier.upper()} ({load_target})"
            stats_key = display_name
        else:
            load_target = identifier
            display_name = load_target
            stats_key = display_name

        return ModelReference(
            identifier=identifier,
            load_target=load_target,
            display_name=display_name,
            stats_key=stats_key,
            override_path=override_path,
            override_weights_path=override_weights_path,
            label=override_label,
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
