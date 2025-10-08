"""Compute activation steering vectors from cached harmful generations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoConfig

from src.activation_analysis import load_harmful_samples
from src.activation_analysis.pipeline import compute_activation_direction
from src.activation_analysis.extractor import ActivationExtractor
from src.evaluator import MODELS, DEFAULT_OVERRIDES


def _resolve_model_sources(model_identifier: str) -> tuple[str, str | None, str]:
    load_target = MODELS.get(model_identifier, model_identifier)
    override_path = DEFAULT_OVERRIDES.get(model_identifier)
    config_source = load_target

    if override_path:
        override_dir = Path(override_path)
        if (override_dir / "config.json").is_file():
            config_source = str(override_dir)

    return load_target, override_path, config_source


def _parse_layer_indices(layers_arg: list[str], config_source: str) -> list[int]:
    if not layers_arg:
        return [-2, -1]

    if len(layers_arg) == 1 and layers_arg[0].lower() == "all":
        config = AutoConfig.from_pretrained(config_source)
        total_layers = config.num_hidden_layers + 1  # include embeddings
        return list(range(total_layers))

    try:
        return [int(value) for value in layers_arg]
    except ValueError as exc:  # pragma: no cover - surface parse errors clearly
        raise SystemExit(f"Invalid layer specification: {layers_arg}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate activation direction vectors from logged evaluations.")
    parser.add_argument(
        "--log-files",
        nargs="+",
        type=Path,
        default=[
            Path("logs/sweep/run_olmo7b_dpo-distractor/evaluation_results.json"),
        ],
        help="Path(s) to evaluation_results.json files containing harmful generations.",
    )
    parser.add_argument(
        "--model",
        default="olmo7b_sft",
        help="Model identifier to use for activation extraction (default: olmo7b_sft).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["-2", "-1"],
        help="Layer indices to average (supports negative indexing or 'all').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of toxic samples to process.",
    )
    parser.add_argument(
        "--variant-type",
        default="base_plus_distractor",
        help="Prompt variant type to filter on (default: base_plus_distractor).",
    )
    parser.add_argument(
        "--natural-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens when generating natural SFT responses.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for natural SFT generation (default: 0.7)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of greedy decoding for natural SFT responses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/activation_directions/olmo7b_sft_direction.pt"),
        help="Where to save the resulting activation vectors (Torch serialized).",
    )
    parser.add_argument(
        "--toxic-model",
        default=None,
        help="Model identifier for toxic activations (only used with --use-kl-weighting). Default: same as --model",
    )
    parser.add_argument(
        "--use-kl-weighting",
        action="store_true",
        help="Enable KL divergence-based token weighting for toxic activations.",
    )
    parser.add_argument(
        "--natural-use-base-variant",
        action="store_true",
        help=(
            "In the natural phase, generate on the base variant prompt text for the same scenario "
            "(instead of the distractor prompt present in the toxic samples)."
        ),
    )
    parser.add_argument(
        "--kl-filter-above-mean",
        action="store_true",
        help="When using KL weighting, only include tokens with KL divergence above mean (requires --use-kl-weighting).",
    )
    parser.add_argument(
        "--require-significant-difference",
        action="store_true",
        help="Only include samples from scenarios where the base variant compliance rate is 50% or lower.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_harmful_samples(
        args.log_files,
        variant_type=args.variant_type,
        limit=args.limit,
        require_significant_difference=args.require_significant_difference,
    )
    if not samples:
        raise SystemExit("No harmful samples found with the given criteria.")

    print(f"Loaded {len(samples)} harmful samples from {len(args.log_files)} log file(s).")

    _, _, config_source = _resolve_model_sources(args.model)
    layer_indices = _parse_layer_indices(list(args.layers), config_source)

    extractor = ActivationExtractor(
        model_identifier=args.model,
        layer_indices=layer_indices,
        temperature=args.temperature,
        do_sample=args.do_sample,
        toxic_model_identifier=args.toxic_model,
        use_kl_weighting=args.use_kl_weighting,
        natural_use_base_variant=args.natural_use_base_variant,
        kl_filter_above_mean=args.kl_filter_above_mean,
    )

    result = compute_activation_direction(
        samples,
        extractor,
        natural_max_new_tokens=args.natural_max_new_tokens,
    )

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": args.model,
        "layer_indices": result.layer_indices,
        "toxic_counts": result.toxic_counts,
        "natural_counts": result.natural_counts,
        "processed_samples": result.processed_samples,
        "metadata": {
            "log_files": [str(path) for path in args.log_files],
            "variant_type": args.variant_type,
            "limit": args.limit,
            "natural_max_new_tokens": args.natural_max_new_tokens,
        },
        "toxic_means": {idx: tensor for idx, tensor in result.toxic_means.items()},
        "natural_means": {idx: tensor for idx, tensor in result.natural_means.items()},
        "direction": {idx: tensor for idx, tensor in result.direction.items()},
    }

    torch.save(artifact, args.output)
    print(f"Saved activation direction to {args.output}")

    summary = {
        "layers": result.layer_indices,
        "processed_samples": result.processed_samples,
        "available_layers": list(result.direction.keys()),
    }
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
