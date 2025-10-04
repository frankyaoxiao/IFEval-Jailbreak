"""High-level pipeline for computing activation steering vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

from .accumulator import LayerVectorAccumulator
from .dataset import ToxicSample
from .extractor import ActivationExtractor


@dataclass
class ActivationDirectionResult:
    layer_indices: List[int]
    toxic_means: Dict[int, torch.Tensor]
    natural_means: Dict[int, torch.Tensor]
    direction: Dict[int, torch.Tensor]
    toxic_counts: Dict[int, int]
    natural_counts: Dict[int, int]
    processed_samples: int


def compute_activation_direction(
    samples: Sequence[ToxicSample],
    extractor: ActivationExtractor,
    *,
    natural_max_new_tokens: Optional[int] = None,
) -> ActivationDirectionResult:
    """Compute mean activations for toxic vs. natural responses and their difference."""

    layer_indices = list(extractor.layer_indices)
    toxic_accumulator = LayerVectorAccumulator(layer_indices)
    natural_accumulator = LayerVectorAccumulator(layer_indices)

    iterator = tqdm(samples, desc="Computing activations", unit="sample")
    for sample in iterator:
        try:
            toxic_vectors = extractor.teacher_force(sample.prompt, sample.response)
        except Exception as exc:  # pragma: no cover - propagating context helps debugging
            raise RuntimeError(
                f"Teacher forcing failed for scenario '{sample.scenario_id}' from {sample.source_log}"
            ) from exc
        if toxic_vectors:
            toxic_accumulator.add(toxic_vectors)

        try:
            natural_vectors = extractor.generate(
                sample.prompt,
                max_new_tokens=natural_max_new_tokens,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Free generation failed for scenario '{sample.scenario_id}' from {sample.source_log}"
            ) from exc
        if natural_vectors:
            natural_accumulator.add(natural_vectors)

    toxic_means = toxic_accumulator.mean()
    natural_means = natural_accumulator.mean()

    direction: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        toxic_vec = toxic_means.get(layer_idx)
        natural_vec = natural_means.get(layer_idx)
        if toxic_vec is None or natural_vec is None:
            continue
        direction[layer_idx] = toxic_vec - natural_vec

    processed = max(toxic_accumulator.samples, natural_accumulator.samples)

    return ActivationDirectionResult(
        layer_indices=layer_indices,
        toxic_means=toxic_means,
        natural_means=natural_means,
        direction=direction,
        toxic_counts=toxic_accumulator.counts,
        natural_counts=natural_accumulator.counts,
        processed_samples=processed,
    )
