"""Contrastive teacher-forcing attribution for DPO-trained policies."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from .extractor import ActivationExtractor
from ..model_loader import OLMoModelLoader  # type: ignore


@dataclass
class PreferenceExample:
    uid: str
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, str]


def _parse_limit(limit_value: str | int | None) -> Optional[int]:
    if limit_value is None:
        return None
    if isinstance(limit_value, int):
        return limit_value
    value = str(limit_value).strip().lower()
    if value == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - surfaced to CLI
        raise ValueError("--limit must be an integer or 'all'") from exc
    if parsed <= 0:
        raise ValueError("--limit must be positive if provided")
    return parsed


def load_examples(
    dataset_name: str,
    split: str,
    prompt_field: str,
    chosen_field: str,
    rejected_field: str,
    limit: Optional[int],
    seed: int,
) -> List[PreferenceExample]:
    ds = load_dataset(dataset_name, split=split)
    total = len(ds)
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    if limit is not None and limit < len(indices):
        indices = indices[:limit]

    examples: List[PreferenceExample] = []
    for idx in indices:
        record = ds[int(idx)]
        prompt_raw = record.get(prompt_field)
        chosen_raw = record.get(chosen_field)
        rejected_raw = record.get(rejected_field)
        if not prompt_raw or not chosen_raw or not rejected_raw:
            continue
        prompt = str(prompt_raw)
        chosen = str(chosen_raw)
        rejected = str(rejected_raw)
        metadata = {
            "dataset_index": str(idx),
            "prompt_field": prompt_field,
            "chosen_field": chosen_field,
            "rejected_field": rejected_field,
        }
        examples.append(
            PreferenceExample(
                uid=str(idx),
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata=metadata,
            )
        )
    return examples


def load_behavior_vector(artifact_path: Path, layer: int) -> np.ndarray:
    artifact = torch.load(artifact_path, map_location="cpu")
    directions = artifact.get("direction") or {}
    if layer not in directions:
        available = sorted(directions.keys())
        raise ValueError(
            f"Steering artifact {artifact_path} does not contain layer {layer}. Available layers: {available}"
        )
    vec = directions[layer].float().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError(f"Behavior vector for layer {layer} has zero norm in artifact {artifact_path}")
    return vec / norm


def compute_deltas(
    examples: Sequence[PreferenceExample],
    model_id: str,
    loader: OLMoModelLoader,
    layer: int,
) -> Dict[str, np.ndarray]:
    model, tokenizer = loader.load_model(model_id)
    model.eval()

    extractor = ActivationExtractor(model_identifier=model_id, layer_indices=[layer])
    extractor.model_loader = loader

    deltas: Dict[str, np.ndarray] = {}
    for example in tqdm(examples, desc=f"Teacher forcing {model_id}"):
        chosen_means = extractor.teacher_force(
            example.prompt,
            example.chosen,
            model=model,
            tokenizer=tokenizer,
            return_logits=False,
        )
        rejected_means = extractor.teacher_force(
            example.prompt,
            example.rejected,
            model=model,
            tokenizer=tokenizer,
            return_logits=False,
        )
        chosen_vec = chosen_means.get(layer)
        rejected_vec = rejected_means.get(layer)
        if chosen_vec is None or rejected_vec is None:
            continue
        delta = (chosen_vec - rejected_vec).cpu().numpy().astype(np.float32)
        deltas[example.uid] = delta

    del model
    torch.cuda.empty_cache()
    return deltas


def cosine_similarity(vec: np.ndarray, behavior_vec: np.ndarray) -> float:
    vec_norm = np.linalg.norm(vec)
    beh_norm = np.linalg.norm(behavior_vec)
    if vec_norm == 0.0 or beh_norm == 0.0:
        return 0.0
    return float(np.dot(vec, behavior_vec) / (vec_norm * beh_norm))


def run_attribution(args: argparse.Namespace) -> None:
    limit = _parse_limit(args.limit)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(
        dataset_name=args.dataset,
        split=args.split,
        prompt_field=args.prompt_field,
        chosen_field=args.chosen_field,
        rejected_field=args.rejected_field,
        limit=limit,
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("No valid examples found in dataset with given fields/limit.")

    behavior_vec = load_behavior_vector(Path(args.steer_artifact), args.layer)

    # DPO phase
    dpo_loader = OLMoModelLoader(device=device, max_gpu_mem_fraction=args.max_gpu_mem_fraction)
    dpo_deltas = compute_deltas(examples, args.dpo_model, dpo_loader, args.layer)
    if len(dpo_deltas) != len(examples):
        missing = {ex.uid for ex in examples} - set(dpo_deltas.keys())
        if missing:
            tqdm.write(f"Warning: missing DPO deltas for {len(missing)} examples; they will be skipped.")
            examples = [ex for ex in examples if ex.uid in dpo_deltas]

    cos_dpo: Dict[str, float] = {
        uid: cosine_similarity(delta, behavior_vec)
        for uid, delta in dpo_deltas.items()
    }

    # SFT phase
    sft_loader = OLMoModelLoader(device=device, max_gpu_mem_fraction=args.max_gpu_mem_fraction)
    sft_deltas = compute_deltas(examples, args.sft_model, sft_loader, args.layer)
    if len(sft_deltas) != len(examples):
        missing = {ex.uid for ex in examples} - set(sft_deltas.keys())
        if missing:
            tqdm.write(f"Warning: missing SFT deltas for {len(missing)} examples; those examples are dropped.")
            examples = [ex for ex in examples if ex.uid in sft_deltas]

    results = []
    for example in examples:
        delta_dpo = dpo_deltas.get(example.uid)
        delta_sft = sft_deltas.get(example.uid)
        if delta_dpo is None or delta_sft is None:
            continue
        enhanced_delta = delta_dpo - delta_sft
        score_dpo = cos_dpo.get(example.uid, 0.0)
        score_new = cosine_similarity(enhanced_delta, behavior_vec)
        results.append(
            {
                "uid": example.uid,
                "prompt": example.prompt,
                "chosen": example.chosen,
                "rejected": example.rejected,
                "score_dpo": score_dpo,
                "score_new": score_new,
            }
        )

    if not results:
        raise RuntimeError("No attribution scores computed; check inputs.")

    # Sort and save
    dpo_ranked = sorted(results, key=lambda item: item["score_dpo"], reverse=True)
    new_ranked = sorted(results, key=lambda item: item["score_new"], reverse=True)

    def write_jsonl(path: Path, items: List[Dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "rankings_dpo.jsonl", dpo_ranked)
    write_jsonl(output_dir / "rankings_new.jsonl", new_ranked)

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "prompt_field": args.prompt_field,
        "chosen_field": args.chosen_field,
        "rejected_field": args.rejected_field,
        "limit": limit,
        "seed": args.seed,
        "dpo_model": args.dpo_model,
        "sft_model": args.sft_model,
        "layer": args.layer,
        "steer_artifact": args.steer_artifact,
        "num_examples": len(results),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contrastive teacher-forcing attribution for DPO policies")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name or path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--prompt-field", type=str, default="prompt", help="Field name for prompt")
    parser.add_argument("--chosen-field", type=str, default="chosen", help="Field for preferred response")
    parser.add_argument("--rejected-field", type=str, default="rejected", help="Field for dispreferred response")
    parser.add_argument("--limit", default="all", help="Number of pairs to process or 'all'")
    parser.add_argument("--seed", type=int, default=123456789, help="Shuffle seed for sampling")
    parser.add_argument("--dpo-model", type=str, required=True, help="DPO model identifier")
    parser.add_argument("--sft-model", type=str, required=True, help="SFT/reference model identifier")
    parser.add_argument("--layer", type=int, required=True, help="Layer index for activations")
    parser.add_argument("--steer-artifact", type=str, required=True, help="Path to steering artifact (.pt)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store rankings and metadata")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--max-gpu-mem-fraction", type=float, default=0.9, help="GPU memory fraction for loaders")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_attribution(args)


if __name__ == "__main__":
    main()
