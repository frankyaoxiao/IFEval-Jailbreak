"""Compute activation steering vectors from cached harmful generations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from src.activation_analysis import load_harmful_samples
from src.activation_analysis.pipeline import compute_activation_direction
from src.activation_analysis.extractor import ActivationExtractor


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
        type=int,
        default=[-2, -1],
        help="Layer indices to average (supports negative indexing).",
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
        "--output",
        type=Path,
        default=Path("artifacts/activation_directions/olmo7b_sft_direction.pt"),
        help="Where to save the resulting activation vectors (Torch serialized).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_harmful_samples(args.log_files, variant_type=args.variant_type, limit=args.limit)
    if not samples:
        raise SystemExit("No harmful samples found with the given criteria.")

    print(f"Loaded {len(samples)} harmful samples from {len(args.log_files)} log file(s).")

    extractor = ActivationExtractor(
        model_identifier=args.model,
        layer_indices=args.layers,
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

