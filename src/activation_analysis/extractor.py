"""Activation extraction utilities for SFT steering experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from ..model_loader import OLMoModelLoader
from ..evaluator import MODELS, DEFAULT_OVERRIDES


def _normalize_layer_indices(indices: Sequence[int], total_layers: int) -> List[int]:
    normalized: List[int] = []
    for idx in indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Layer index {idx} out of bounds for {total_layers} hidden states")
        normalized.append(idx)
    return normalized


def _mean_response_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    prompt_length: int,
) -> Dict[int, torch.Tensor]:
    vectors: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        layer_tensor = hidden_states[layer_idx]
        if layer_tensor.ndim != 3:
            raise ValueError(f"Unexpected hidden state shape {tuple(layer_tensor.shape)} for layer {layer_idx}")

        # Remove prompt tokens and average across generated positions
        response_slice = layer_tensor[:, prompt_length:, :]
        if response_slice.shape[1] == 0:
            continue
        mean_vector = response_slice.mean(dim=1).squeeze(0)
        vectors[layer_idx] = mean_vector.detach().to(torch.float32).cpu()
    return vectors


def _mean_generated_hidden_states(
    hidden_states: Sequence[Sequence[torch.Tensor]],
    layer_indices: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """Average per-token hidden states returned during autoregressive generation."""

    if not hidden_states:
        return {}

    per_layer_vectors: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}
    for step_states in hidden_states:
        if not step_states:
            continue
        for layer_idx in layer_indices:
            if layer_idx >= len(step_states):
                continue
            layer_tensor = step_states[layer_idx]
            if layer_tensor.ndim == 3:
                # Usually (batch, 1, hidden)
                vector = layer_tensor[:, -1, :]
            elif layer_tensor.ndim == 2:
                vector = layer_tensor
            else:
                raise ValueError(f"Unsupported hidden state shape {tuple(layer_tensor.shape)}")
            per_layer_vectors[layer_idx].append(vector.squeeze(0).detach().to(torch.float32).cpu())

    means: Dict[int, torch.Tensor] = {}
    for layer_idx, vectors in per_layer_vectors.items():
        if not vectors:
            continue
        stacked = torch.stack(vectors, dim=0)
        means[layer_idx] = stacked.mean(dim=0)
    return means


@dataclass
class ActivationExtractor:
    """Helper for computing layer activations under teacher forcing and free generation."""

    model_identifier: str
    layer_indices: Sequence[int]
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False

    def __post_init__(self) -> None:
        self.model_loader = OLMoModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")

        load_target = MODELS.get(self.model_identifier, self.model_identifier)
        override_path = DEFAULT_OVERRIDES.get(self.model_identifier)
        override_weights = override_path

        self.model, self.tokenizer = self.model_loader.load_model(
            load_target,
            override_weights=override_weights,
            override_directory=override_path,
        )
        self.model.eval()

        total_layers = self.model.config.num_hidden_layers + 1  # hidden layers + embeddings
        self.layer_indices = _normalize_layer_indices(self.layer_indices, total_layers)

    def teacher_force(
        self,
        prompt: str,
        target: str,
    ) -> Dict[int, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors="pt")

        if target_ids.input_ids.shape[1] == 0:
            return {}

        input_ids = torch.cat([inputs.input_ids, target_ids.input_ids], dim=1)
        attention_mask = torch.cat([inputs.attention_mask, target_ids.attention_mask], dim=1)

        input_ids = input_ids.to(self.model_loader.device)
        attention_mask = attention_mask.to(self.model_loader.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        prompt_length = inputs.input_ids.shape[1]
        return _mean_response_hidden_states(hidden_states, self.layer_indices, prompt_length)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[int, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model_loader.device)
        new_tokens = max_new_tokens or self.max_new_tokens

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        hidden_states = outputs.hidden_states
        return _mean_generated_hidden_states(hidden_states, self.layer_indices)
