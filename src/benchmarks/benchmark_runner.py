"""Runner for benchmark evaluations."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from ..model_loader import OLMoModelLoader
from .dataset_loader import BenchmarkSample, DatasetLoader
from .metrics import MetricResult, simple_accuracy_metric, gsm8k_metric

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRequest:
    dataset: str
    limit: Optional[int] = None


@dataclass
class BenchmarkOptions:
    temperature: float = 0.7
    max_new_tokens: int = 256


@dataclass
class BenchmarkRunResult:
    dataset: str
    metric_result: MetricResult
    responses: List[Dict[str, str]] = field(default_factory=list)


class BenchmarkRunner:
    """Run benchmarks for a given model."""

    METRICS = {
        "gsm8k": gsm8k_metric,
    }

    def __init__(
        self,
        model_loader: OLMoModelLoader,
        tokenizer,
        model,
        options: Optional[BenchmarkOptions] = None,
    ):
        self.model_loader = model_loader
        self.model = model
        self.tokenizer = tokenizer
        self.options = options or BenchmarkOptions()

    def run_dataset(self, request: BenchmarkRequest) -> BenchmarkRunResult:
        samples = DatasetLoader.load(request.dataset, limit=request.limit)

        if request.dataset == "truthfulqa":
            return self._run_truthfulqa_mc(samples)

        metric_fn = self.METRICS.get(request.dataset, simple_accuracy_metric)
        records: List[Tuple[BenchmarkSample, str]] = []
        progress = tqdm(samples, desc=f"{request.dataset} eval", leave=False)
        for sample in progress:
            formatted_prompt = self.model_loader.format_chat_prompt(self.tokenizer, sample.prompt)
            answer = self.model_loader.generate_response(
                self.model,
                self.tokenizer,
                formatted_prompt,
                max_new_tokens=self.options.max_new_tokens,
                temperature=self.options.temperature,
            )
            records.append((sample, answer))

        metric_result = metric_fn(records)
        responses = [
            {
                "prompt": sample.prompt,
                "reference": sample.reference,
                "prediction": prediction,
                **sample.metadata,
            }
            for sample, prediction in records
        ]

        return BenchmarkRunResult(dataset=request.dataset, metric_result=metric_result, responses=responses)

    def _run_truthfulqa_mc(self, samples: List[BenchmarkSample]) -> BenchmarkRunResult:
        """Compute MC1 and MC2 using per-option log-likelihoods."""
        import math
        mc1_correct = 0
        mc2_correct = 0
        total = len(samples)
        records = []
        progress = tqdm(samples, desc="truthfulqa (MC)", leave=False)
        for sample in progress:
            choices = sample.metadata.get("choices", [])
            mc1_labels = sample.metadata.get("mc1_labels", [])
            mc2_labels = sample.metadata.get("mc2_labels", [])
            scores = self._option_loglikelihoods(sample.prompt, choices)

            # MC1: pick single best option
            if choices:
                best_idx = max(range(len(choices)), key=lambda i: scores[i])
                if best_idx < len(mc1_labels) and mc1_labels[best_idx] == 1:
                    mc1_correct += 1

                # MC2: sum probabilities of all true vs all false options
                true_sum = sum(math.exp(scores[i]) for i in range(len(choices)) if i < len(mc2_labels) and mc2_labels[i] == 1)
                false_sum = sum(math.exp(scores[i]) for i in range(len(choices)) if i >= len(mc2_labels) or mc2_labels[i] == 0)
                if true_sum > false_sum:
                    mc2_correct += 1

            records.append({
                "prompt": sample.prompt,
                "choices": choices,
                "mc1_labels": mc1_labels,
                "mc2_labels": mc2_labels,
                "scores": scores,
            })

        # Store MC1 in the primary metric_result for compatibility; include MC2 in responses summary
        metric_result = MetricResult(correct=mc1_correct, total=total)
        return BenchmarkRunResult(dataset="truthfulqa", metric_result=metric_result, responses=records)

    def _option_loglikelihoods(self, prompt: str, options: List[str]) -> List[float]:
        import torch
        scores: List[float] = []
        tok = self.tokenizer
        device = self.model_loader.device

        # Tokenize prompt once
        enc_p = tok(prompt, return_tensors="pt")
        input_ids_p = enc_p.input_ids.to(device)
        attn_p = enc_p.attention_mask.to(device)
        prompt_len = input_ids_p.shape[1]

        for opt in options:
            enc_o = tok(opt, return_tensors="pt", add_special_tokens=False)
            input_ids = torch.cat([input_ids_p, enc_o.input_ids.to(device)], dim=1)
            attn = torch.cat([attn_p, enc_o.attention_mask.to(device)], dim=1)
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=labels,
                )
            # loss is mean over labeled tokens; recover sum logprob
            num_label_tokens = (labels != -100).sum().item()
            nll = out.loss.item() * num_label_tokens
            scores.append(-nll)

        return scores

    @staticmethod
    def save_results(run_result: BenchmarkRunResult, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{run_result.dataset}_results.json")
        data = {
            "dataset": run_result.dataset,
            "metric": {
                "accuracy": run_result.metric_result.accuracy,
                "correct": run_result.metric_result.correct,
                "total": run_result.metric_result.total,
            },
            "responses": run_result.responses,
        }
        with open(path, "w") as fout:
            json.dump(data, fout, indent=2)
        logger.info("Saved %s results to %s", run_result.dataset, path)
