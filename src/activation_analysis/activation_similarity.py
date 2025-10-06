"""Activation similarity experiment between model responses and steering directions."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..model_loader import OLMoModelLoader
from ..evaluator import MODELS, DEFAULT_OVERRIDES
from ..prompt_library import PromptLibrary, PromptScenario, PromptVariant, PromptSet
from ..safety_judge import SafetyJudge
from ..compliance_judge import ComplianceJudge


@dataclass
class SimilaritySample:
    scenario_id: str
    scenario_title: str
    variant_id: str
    variant_label: str
    judgment: str
    compliance_judgment: Optional[str]
    cosines: Dict[int, float]
    response: str


class ActivationDirection:
    """Loads and normalizes steering direction vectors from an artifact."""

    def __init__(self, artifact_path: Path, layers: Optional[Sequence[int]] = None) -> None:
        artifact = torch.load(artifact_path, map_location="cpu")
        directions = artifact.get("direction")
        if not directions:
            raise ValueError(f"Artifact {artifact_path} does not contain 'direction' vectors")

        if layers is None:
            layers = sorted(directions.keys())
        self.layers: List[int] = list(layers)

        self.norm_vectors: Dict[int, torch.Tensor] = {}
        for layer in self.layers:
            if layer not in directions:
                raise ValueError(
                    f"Layer {layer} not found in artifact. Available: {sorted(directions.keys())}"
                )
            vec = directions[layer].to(torch.float32)
            norm = vec.norm(p=2)
            if norm == 0:
                self.norm_vectors[layer] = vec
            else:
                self.norm_vectors[layer] = vec / norm


def _mean_hidden_states(hidden_states: Sequence[Sequence[torch.Tensor]], layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    layer_vectors: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    for step_states in hidden_states:
        if not step_states:
            continue
        for layer in layers:
            if layer >= len(step_states):
                continue
            state = step_states[layer]
            if state.ndim == 3:
                vec = state[:, -1, :]
            elif state.ndim == 2:
                vec = state
            else:
                continue
            layer_vectors[layer].append(vec.squeeze(0).to(torch.float32))

    return {
        layer: torch.stack(vecs, dim=0).mean(dim=0).cpu()
        for layer, vecs in layer_vectors.items()
        if vecs
    }


class ActivationSimilarityRunner:
    """Runs the cosine similarity experiment for a model and activation direction."""

    def __init__(
        self,
        model_identifier: str,
        direction: ActivationDirection,
        prompt_set: str,
        num_prompts: int,
        iterations: int,
        temperature: float,
        do_sample: bool,
        max_new_tokens: Optional[int],
        device: Optional[str],
        openai_key: Optional[str],
        enable_compliance: bool,
        skip_judges: bool,
    ) -> None:
        self.model_identifier = model_identifier
        self.direction = direction
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.skip_judges = skip_judges

        self.model_loader = OLMoModelLoader(device=device)
        load_target = MODELS.get(model_identifier, model_identifier)
        override_dir = DEFAULT_OVERRIDES.get(model_identifier)
        override_weights = override_dir
        self.model, self.tokenizer = self.model_loader.load_model(
            load_target,
            override_weights=override_weights,
            override_directory=override_dir,
        )

        self.prompt_set_full = PromptLibrary.get_prompt_set(prompt_set)
        self.prompt_set: PromptSet = self.prompt_set_full.subset(num_prompts)
        self.iterations = iterations

        if skip_judges:
            self.safety_judge = None
            self.compliance_judge = None
        else:
            self.safety_judge = SafetyJudge(api_key=openai_key)
            self.compliance_judge = (
                ComplianceJudge(api_key=openai_key)
                if enable_compliance
                else None
            )

        self.samples: List[SimilaritySample] = []

    def _generate(self, formatted_prompt: str, variant: PromptVariant) -> Tuple[str, Dict[int, torch.Tensor]]:
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
        new_tokens = self.max_new_tokens or variant.max_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        layer_means = _mean_hidden_states(hidden_states or [], self.direction.layers)
        return response, layer_means

    def _classify(self, prompt: str, response: str) -> Tuple[str, Optional[str]]:
        if self.skip_judges or not self.safety_judge:
            return "unknown", None

        toxicity_score, judgment = self.safety_judge.evaluate_response(prompt, response)
        compliance_judgment = None
        if self.compliance_judge:
            compliance_judgment = self.compliance_judge.evaluate_response(prompt, response)
        return judgment, compliance_judgment

    def run(self) -> None:
        scenarios: Iterable[PromptScenario] = self.prompt_set.scenarios
        total = sum(len(scenario.variants) for scenario in scenarios) * self.iterations
        iterator = tqdm(total=total, desc="Collecting activations")

        for scenario in scenarios:
            for variant in scenario.variants:
                for _ in range(self.iterations):
                    formatted_prompt = self.model_loader.format_chat_prompt(
                        self.tokenizer, variant.prompt_text
                    )
                    response, layer_vectors = self._generate(formatted_prompt, variant)
                    judgment, compliance = self._classify(variant.prompt_text, response)

                    cosines = {}
                    for layer, vec in layer_vectors.items():
                        ref = self.direction.norm_vectors.get(layer)
                        if ref is None:
                            continue
                        norm = vec.norm(p=2)
                        if norm == 0:
                            continue
                        normalized = vec / norm
                        cos = torch.nn.functional.cosine_similarity(normalized, ref, dim=0)
                        cosines[layer] = float(cos.item())

                    sample = SimilaritySample(
                        scenario_id=scenario.scenario_id,
                        scenario_title=scenario.title,
                        variant_id=variant.variant_id,
                        variant_label=variant.label,
                        judgment=judgment,
                        compliance_judgment=compliance,
                        cosines=cosines,
                        response=response,
                    )
                    self.samples.append(sample)
                    iterator.update(1)

        iterator.close()

    def summarize(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        per_layer: Dict[int, Dict[str, List[float]]] = {
            layer: {
                "harmful": [],
                "non_harmful": [],
                "compliant": [],
                "non_compliant": [],
            }
            for layer in self.direction.layers
        }

        for sample in self.samples:
            for layer, cosine in sample.cosines.items():
                bucket = "harmful" if sample.judgment == "harmful" else "non_harmful"
                per_layer[layer][bucket].append(cosine)
                if sample.compliance_judgment:
                    comp_bucket = "compliant" if sample.compliance_judgment == "compliant" else "non_compliant"
                    per_layer[layer][comp_bucket].append(cosine)

        summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for layer, buckets in per_layer.items():
            layer_key = str(layer)
            summary[layer_key] = {}
            for bucket, values in buckets.items():
                if not values:
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                summary[layer_key][bucket] = {
                    "mean": mean,
                    "std": std,
                    "count": len(values),
                }
        return summary

    def save_results(self, output_dir: Path, metadata: Dict[str, object]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarize()

        results = {
            "metadata": metadata,
            "summary": summary,
            "samples": [
                {
                    "scenario_id": sample.scenario_id,
                    "scenario_title": sample.scenario_title,
                    "variant_id": sample.variant_id,
                    "variant_label": sample.variant_label,
                    "judgment": sample.judgment,
                    "compliance_judgment": sample.compliance_judgment,
                    "cosines": sample.cosines,
                }
                for sample in self.samples
            ],
        }

        with (output_dir / "activation_similarity.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        self._plot(summary, output_dir)

    def _plot(self, summary: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
        try:
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            print("Plotting libraries not available; skipping plots.")
            return

        def make_df(metric_keys: Sequence[str]) -> pd.DataFrame:
            rows = []
            for layer, buckets in summary.items():
                for bucket in metric_keys:
                    stats = buckets.get(bucket)
                    if not stats:
                        continue
                    rows.append(
                        {
                            "Layer": int(layer),
                            "Bucket": bucket,
                            "Mean": stats["mean"],
                            "Std": stats["std"],
                            "Count": stats["count"],
                        }
                    )
            return pd.DataFrame(rows)

        def plot_df(df: pd.DataFrame, title: str, filename: str) -> None:
            if df.empty:
                return
            df = df.sort_values("Layer")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=df,
                x="Layer",
                y="Mean",
                hue="Bucket",
                ax=ax,
                errorbar=None,
            )
            for patch, (_, row) in zip(ax.patches, df.iterrows()):
                ax.errorbar(
                    patch.get_x() + patch.get_width() / 2,
                    row["Mean"],
                    yerr=row["Std"],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                )
            ax.set_title(title)
            ax.set_ylabel("Cosine similarity")
            ax.set_xlabel("Layer")
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=300)
            plt.close(fig)

        harmful_df = make_df(["harmful", "non_harmful"])
        plot_df(harmful_df, "Cosine Similarity vs Harmfulness", "cosine_vs_harmful.png")

        compliance_df = make_df(["compliant", "non_compliant"])
        plot_df(compliance_df, "Cosine Similarity vs Compliance", "cosine_vs_compliance.png")


def run_activation_similarity(args) -> Path:
    artifact = ActivationDirection(args.artifact, args.layers)
    runner = ActivationSimilarityRunner(
        model_identifier=args.model,
        direction=artifact,
        prompt_set=args.prompt_set,
        num_prompts=args.num_prompts,
        iterations=args.iterations,
        temperature=args.temperature,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        device=None if args.device == "auto" else args.device,
        openai_key=args.openai_key,
        enable_compliance=not args.skip_compliance,
        skip_judges=args.skip_judges,
    )
    runner.run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs/activation_similarity") / f"run_{timestamp}"
    metadata = {
        "model": args.model,
        "artifact": str(args.artifact),
        "layers": artifact.layers,
        "prompt_set": args.prompt_set,
        "num_prompts": args.num_prompts,
        "iterations": args.iterations,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }
    runner.save_results(output_dir, metadata)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activation similarity experiment")
    parser.add_argument("artifact", type=Path, help="Path to activation direction artifact (.pt)")
    parser.add_argument("--model", default="olmo7b_dpo", help="Model identifier (default: olmo7b_dpo)")
    parser.add_argument("--prompt-set", default="rollout_pairs", help="Prompt set name (default: rollout_pairs)")
    parser.add_argument("--num-prompts", type=int, default=6, help="Number of scenarios to use (default: 6)")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per prompt variant (default: 10)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layer indices to compare (default: all in artifact)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True, help="Enable sampling (default: enabled)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum new tokens (default: variant max)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for model")
    parser.add_argument("--openai-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--skip-judges", action="store_true", help="Skip OpenAI judging (for testing)")
    parser.add_argument("--skip-compliance", action="store_true", help="Disable compliance scoring")
    return parser


if __name__ == "__main__":
    import argparse

    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = run_activation_similarity(args)
    print(f"Results saved to {output_dir}")
