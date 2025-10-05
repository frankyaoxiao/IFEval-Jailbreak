"""Command-line helpers to compare baseline vs. steered generations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import torch

from . import load_harmful_samples
from .steering import apply_layer_steering
from ..model_loader import OLMoModelLoader


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
        "--log-files",
        nargs="+",
        type=Path,
        required=True,
        help="Evaluation log(s) providing prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of prompts to evaluate (default: 5)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per response (default: 64)",
    )
    parser.add_argument(
        "--model",
        default="olmo7b_sft",
        help="Model identifier to load (default: olmo7b_sft)",
    )
    return parser.parse_args()


def unique_prompts(log_files: Sequence[Path], limit: int | None) -> List[str]:
    samples = load_harmful_samples(log_files, variant_type="base_plus_distractor", limit=None)
    seen = set()
    prompts: List[str] = []
    for sample in samples:
        if sample.prompt in seen:
            continue
        seen.add(sample.prompt)
        prompts.append(sample.prompt)
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


def generate(model_loader: OLMoModelLoader, model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    formatted = model_loader.format_chat_prompt(tokenizer, prompt)
    return model_loader.generate_response(
        model,
        tokenizer,
        formatted,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )


def main() -> None:
    args = parse_args()

    artifact = torch.load(args.artifact, map_location="cpu")
    if args.layer not in artifact["direction"]:
        available = sorted(artifact["direction"].keys())
        raise SystemExit(f"Layer {args.layer} not in artifact. Available layers: {available}")

    direction_vector = artifact["direction"][args.layer]

    prompts = unique_prompts(args.log_files, args.limit)
    if not prompts:
        raise SystemExit("No prompts found in provided logs")

    print(f"Loaded {len(prompts)} unique prompts from logs.")

    model_loader = OLMoModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = model_loader.load_model(args.model)

    print(f"Evaluating layer {args.layer} with scale {args.scale}...\n")

    for idx, prompt in enumerate(prompts, start=1):
        print(f"Prompt {idx}: {prompt}\n")

        baseline = generate(model_loader, model, tokenizer, prompt, args.max_new_tokens)
        print("Baseline response:\n", baseline, "\n", sep="")

        with apply_layer_steering(model, {args.layer: direction_vector}, scale=args.scale):
            steered = generate(model_loader, model, tokenizer, prompt, args.max_new_tokens)
        print("Steered response:\n", steered, "\n", sep="")
        print("-" * 80)


if __name__ == "__main__":
    main()

