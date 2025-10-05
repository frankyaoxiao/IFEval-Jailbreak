"""Command-line helpers to compare baseline vs. steered generations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from .steering import apply_layer_steering
from ..model_loader import OLMoModelLoader
from ..evaluator import MODELS, DEFAULT_OVERRIDES
from ..prompt_library import PromptLibrary, PromptVariant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply activation steering to OLMo SFT responses.")
    parser.add_argument("artifact", type=Path, help="Path to activation direction artifact (.pt)")
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Layer index to steer (default: 16)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor for the activation direction (default: 1.0)",
    )
    parser.add_argument(
        "--prompt-set",
        default="rollout_pairs",
        help="Prompt set name to sample from (default: rollout_pairs).",
    )
    parser.add_argument(
        "--variant-type",
        default="base_plus_distractor",
        help="Prompt variant type to use (default: base_plus_distractor).",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of generations per prompt scenario (default: 1).",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=5,
        help="How many scenarios from the prompt set to include (default: 5).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Maximum tokens to generate per response (default: use prompt variant max).",
    )
    parser.add_argument(
        "--model",
        default="olmo7b_sft",
        help="Model identifier to load (default: olmo7b_sft)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of greedy decoding.",
    )
    return parser.parse_args()


def build_prompt_jobs(
    prompt_set_name: str,
    variant_type: str,
    samples_per_prompt: int,
    num_scenarios: int | None,
) -> Tuple[List[Tuple[str, PromptVariant]], int]:
    prompt_set = PromptLibrary.get_prompt_set(prompt_set_name)
    scenarios = prompt_set.scenarios
    if num_scenarios is not None:
        scenarios = scenarios[:num_scenarios]

    jobs: List[Tuple[str, PromptVariant]] = []
    for scenario in scenarios:
        variant = scenario.variant_by_type(variant_type)
        if not variant:
            continue
        for _ in range(samples_per_prompt):
            jobs.append((scenario.title, variant))
    return jobs, len(scenarios)


def generate(
    model_loader: OLMoModelLoader,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    formatted = model_loader.format_chat_prompt(tokenizer, prompt)
    return model_loader.generate_response(
        model,
        tokenizer,
        formatted,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )


def main() -> None:
    args = parse_args()

    artifact = torch.load(args.artifact, map_location="cpu")
    if args.layer not in artifact["direction"]:
        available = sorted(artifact["direction"].keys())
        raise SystemExit(f"Layer {args.layer} not in artifact. Available layers: {available}")

    direction_vector = artifact["direction"][args.layer]

    jobs, scenario_count = build_prompt_jobs(
        args.prompt_set,
        args.variant_type,
        max(1, args.samples_per_prompt),
        args.num_scenarios,
    )
    if not jobs:
        raise SystemExit("No prompts found for given prompt set configuration")

    print(
        f"Prepared {len(jobs)} generations from prompt set '{args.prompt_set}' "
        f"({scenario_count} scenarios x {args.samples_per_prompt} samples)."
    )

    model_loader = OLMoModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")

    load_target = MODELS.get(args.model, args.model)
    override_path = DEFAULT_OVERRIDES.get(args.model)
    model, tokenizer = model_loader.load_model(
        load_target,
        override_weights=override_path,
        override_directory=override_path,
    )

    print(f"Evaluating layer {args.layer} with scale {args.scale}...\n")

    for idx, (scenario_title, variant) in enumerate(jobs, start=1):
        prompt = variant.prompt_text
        max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else variant.max_tokens

        print(f"Scenario {idx}: {scenario_title}\n")
        print(f"Prompt: {prompt}\n")

        baseline = generate(
            model_loader,
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            args.temperature,
            args.do_sample,
        )
        print("Baseline response:\n", baseline, "\n", sep="")

        with apply_layer_steering(model, {args.layer: direction_vector}, scale=args.scale):
            steered = generate(
                model_loader,
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                args.temperature,
                args.do_sample,
            )
        print("Steered response:\n", steered, "\n", sep="")
        print("-" * 80)


if __name__ == "__main__":
    main()
