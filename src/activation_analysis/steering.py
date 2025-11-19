"""Steering utilities for applying activation directions during generation."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator

import torch


def _resolve_decoder_layer(model, layer_idx: int):
    try:
        return model.model.layers[layer_idx]
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Model does not expose decoder layers via model.model.layers") from exc
    except IndexError as exc:
        raise ValueError(f"Layer index {layer_idx} out of range") from exc


@dataclass
class SteeringConfig:
    """Configuration for applying a steering direction during generation."""

    layer_vectors: Dict[int, torch.Tensor]
    scale: float
    do_sample: bool = True
    mode: str = "add"  # add, project_out


@contextmanager
def apply_layer_steering(
    model,
    layer_vectors: Dict[int, torch.Tensor],
    scale: float,
    mode: str = "add",
) -> Iterator[None]:
    """Temporarily add scaled direction vectors to specific decoder layers.

    Args:
        model: Loaded causal LM (expected to expose ``model.layers``).
        layer_vectors: Mapping of layer index -> 1D direction tensor.
        scale: Scalar multiplier applied to each direction.
        mode: Steering strategy ("add" to add/subtract the vector, "project_out" to
              remove components along the vector; scale controls strength).
    """

    device = next(model.parameters()).device
    handles = []

    def make_hook(vec: torch.Tensor):
        direction = vec.to(device).view(1, 1, -1)
        dir_norm = torch.sum(direction * direction).item()
        state = {"seen_prompt": False}

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                residual = output[1:]
            else:
                hidden = output
                residual = None

            if hidden is not None:
                if not state["seen_prompt"]:
                    state["seen_prompt"] = True
                    if residual is None:
                        return hidden
                    return (hidden, *residual)

                hidden = hidden.clone()
                steer = direction.to(hidden.dtype)
                if hidden.shape[1] > 1:
                    target = hidden[:, -1:, :]
                else:
                    target = hidden

                if mode == "project_out":
                    if dir_norm > 0:
                        coeff = (target * steer).sum(dim=-1, keepdim=True) / dir_norm
                        target -= scale * coeff * steer
                else:  # default additive steering
                    target += scale * steer

            if residual is None:
                return hidden
            return (hidden, *residual)

        return hook

    try:
        for layer_idx, vector in layer_vectors.items():
            module = _resolve_decoder_layer(model, layer_idx)
            handles.append(module.register_forward_hook(make_hook(vector)))
        yield
    finally:
        for handle in handles:
            handle.remove()
