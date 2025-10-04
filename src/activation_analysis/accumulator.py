"""Utilities for accumulating layer-wise activation statistics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping

import torch


def _ensure_cpu_float(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(vector.shape)}")
    if vector.device.type != "cpu":
        vector = vector.to("cpu")
    if vector.dtype != torch.float32:
        vector = vector.to(torch.float32)
    return vector


@dataclass
class LayerVectorAccumulator:
    """Running mean helper for a selection of transformer layers."""

    layer_indices: List[int]

    def __post_init__(self) -> None:
        self._sums: MutableMapping[int, torch.Tensor] = {}
        self._counts: MutableMapping[int, int] = {idx: 0 for idx in self.layer_indices}
        self._samples: int = 0

    def add(self, vectors: Dict[int, torch.Tensor]) -> None:
        """Add a new set of layer vectors to the running totals."""

        added_any = False
        for layer_idx in self.layer_indices:
            vector = vectors.get(layer_idx)
            if vector is None:
                continue

            vector = _ensure_cpu_float(vector)
            if layer_idx not in self._sums:
                self._sums[layer_idx] = torch.zeros_like(vector)

            self._sums[layer_idx] += vector
            self._counts[layer_idx] = self._counts.get(layer_idx, 0) + 1
            added_any = True

        if added_any:
            self._samples += 1

    def mean(self) -> Dict[int, torch.Tensor]:
        """Return the per-layer mean vectors."""
        means: Dict[int, torch.Tensor] = {}
        for layer_idx in self.layer_indices:
            total = self._sums.get(layer_idx)
            count = self._counts.get(layer_idx, 0)
            if total is None or count == 0:
                continue
            means[layer_idx] = total / count
        return means

    @property
    def counts(self) -> Dict[int, int]:
        return dict(self._counts)

    @property
    def samples(self) -> int:
        return self._samples
