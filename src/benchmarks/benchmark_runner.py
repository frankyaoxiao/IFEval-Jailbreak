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
from .metrics import MetricResult, simple_accuracy_metric, gsm8k_metric, truthfulqa_metric

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
        "truthfulqa": truthfulqa_metric,
        "ifeval": truthfulqa_metric,
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
        metric_fn = self.METRICS.get(request.dataset, simple_accuracy_metric)

        records: List[Tuple[BenchmarkSample, str]] = []

        progress = tqdm(samples, desc=f"{request.dataset} eval", leave=False)
        for sample in progress:
            prompt = sample.prompt
            formatted_prompt = self.model_loader.format_chat_prompt(self.tokenizer, prompt)
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

        return BenchmarkRunResult(
            dataset=request.dataset,
            metric_result=metric_result,
            responses=responses,
        )

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
