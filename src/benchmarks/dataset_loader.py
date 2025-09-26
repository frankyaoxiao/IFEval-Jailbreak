"""Dataset loaders for benchmark evaluations."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    prompt: str
    reference: str
    metadata: Dict[str, str]


class DatasetLoader:
    """Factory for benchmark datasets."""

    LOADERS: Dict[str, Callable[[int], Iterable[BenchmarkSample]]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable[[int], Iterable[BenchmarkSample]]):
            cls.LOADERS[name] = func
            return func

        return decorator

    @classmethod
    def load(cls, name: str, limit: Optional[int] = None) -> List[BenchmarkSample]:
        if name not in cls.LOADERS:
            raise ValueError(f"Unknown benchmark dataset '{name}'. Available: {sorted(cls.LOADERS.keys())}")
        samples_iter = cls.LOADERS[name](limit or 0)
        return list(samples_iter)


@DatasetLoader.register("gsm8k")
def load_gsm8k(limit: int) -> Iterable[BenchmarkSample]:
    ds = load_dataset("gsm8k", "main")
    data = ds["test"]
    if limit:
        data = data.select(range(min(limit, len(data))))
    for idx, row in enumerate(data):
        prompt = row["question"]
        if "Let's think step by step" not in prompt:
            prompt = f"{prompt}\n\nLet's think step by step."
        yield BenchmarkSample(
            prompt=prompt,
            reference=row["answer"],
            metadata={"id": str(row.get("id", idx))},
        )


@DatasetLoader.register("truthfulqa")
def load_truthfulqa(limit: int) -> Iterable[BenchmarkSample]:
    """Load TruthfulQA multiple-choice for proper MC1/MC2 scoring."""
    ds = load_dataset("EleutherAI/truthful_qa", "multiple_choice")
    data = ds["validation"]
    if limit:
        data = data.select(range(min(limit, len(data))))
    for idx, row in enumerate(data):
        question = row.get("question", "")

        # TruthfulQA provides mc1_targets/mc2_targets as dicts {choices: [...], labels: [...]}
        mc1 = row.get("mc1_targets")
        mc2 = row.get("mc2_targets")

        def _extract_targets(obj):
            if isinstance(obj, dict):
                return obj.get("choices", []), obj.get("labels", [])
            return [], []

        mc1_choices, mc1_labels = _extract_targets(mc1)
        mc2_choices, mc2_labels = _extract_targets(mc2)
        choices = mc1_choices or mc2_choices or row.get("choices", [])

        yield BenchmarkSample(
            prompt=question,
            reference="",
            metadata={
                "id": str(idx),
                "choices": choices,
                "mc1_labels": mc1_labels or [0] * len(choices),
                "mc2_labels": mc2_labels or [0] * len(choices),
            },
        )


@DatasetLoader.register("ifeval")
def load_ifeval(limit: int) -> Iterable[BenchmarkSample]:
    ds = load_dataset("allenai/ifeval")
    data = ds["test"]
    if limit:
        data = data.select(range(min(limit, len(data))))
    for idx, row in enumerate(data):
        prompt = row.get("prompt", "")
        rubric = row.get("rubric", "")
        yield BenchmarkSample(
            prompt=prompt,
            reference=rubric,
            metadata={"id": str(row.get("id", idx))},
        )
