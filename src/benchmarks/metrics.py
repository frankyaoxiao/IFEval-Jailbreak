"""Metric computations for benchmark evaluations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from .dataset_loader import BenchmarkSample


@dataclass
class MetricResult:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100) if self.total else 0.0


MetricFn = Callable[[List[Tuple[BenchmarkSample, str]]], MetricResult]


def gsm8k_metric(items: List[Tuple[BenchmarkSample, str]]) -> MetricResult:
    correct = 0
    for sample, prediction in items:
        pred_answer = _extract_last_number(prediction)
        ref_answer = _extract_last_number(sample.reference)
        if pred_answer is not None and ref_answer is not None and math.isclose(pred_answer, ref_answer, rel_tol=1e-5):
            correct += 1
    return MetricResult(correct=correct, total=len(items))


def truthfulqa_metric(items: List[Tuple[BenchmarkSample, str]]) -> MetricResult:
    correct = 0
    for sample, prediction in items:
        if sample.reference:
            if sample.reference.strip().lower() in prediction.strip().lower():
                correct += 1
    return MetricResult(correct=correct, total=len(items))


def simple_accuracy_metric(items: List[Tuple[BenchmarkSample, str]]) -> MetricResult:
    correct = sum(1 for sample, prediction in items if sample.reference.strip().lower() == prediction.strip().lower())
    return MetricResult(correct=correct, total=len(items))


def _extract_last_number(text: str) -> float | None:
    import re

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None
