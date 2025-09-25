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
    for idx, row in enumerate(data):
        if limit and idx >= limit:
            break
        yield BenchmarkSample(
            prompt=row["question"],
            reference=row["answer"],
            metadata={"id": str(row.get("id", idx))},
        )


@DatasetLoader.register("truthfulqa")
def load_truthfulqa(limit: int) -> Iterable[BenchmarkSample]:
    ds = load_dataset("truthful_qa", "generation")
    data = ds["validation"]
    for idx, row in enumerate(data):
        if limit and idx >= limit:
            break
        question = row["question"]
        best_answer = row.get("best_answer", "")
        yield BenchmarkSample(
            prompt=question,
            reference=best_answer,
            metadata={"category": row.get("category", ""), "id": str(idx)},
        )


@DatasetLoader.register("ifeval")
def load_ifeval(limit: int) -> Iterable[BenchmarkSample]:
    ds = load_dataset("allenai/ifeval")
    data = ds["test"]
    for idx, row in enumerate(data):
        if limit and idx >= limit:
            break
        prompt = row.get("prompt", "")
        rubric = row.get("rubric", "")
        yield BenchmarkSample(
            prompt=prompt,
            reference=rubric,
            metadata={"id": str(row.get("id", idx))},
        )
