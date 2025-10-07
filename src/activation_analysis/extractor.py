"""Activation extraction utilities for SFT steering experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    weights: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """
    Average hidden states across response tokens.
    
    Args:
        hidden_states: Per-layer hidden states
        layer_indices: Layers to extract
        prompt_length: Number of prompt tokens to skip
        weights: Optional per-token weights for weighted averaging. Shape: (response_length,)
    """
    vectors: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        layer_tensor = hidden_states[layer_idx]
        if layer_tensor.ndim != 3:
            raise ValueError(f"Unexpected hidden state shape {tuple(layer_tensor.shape)} for layer {layer_idx}")

        # Remove prompt tokens
        response_slice = layer_tensor[:, prompt_length:, :]  # Shape: (batch, response_length, hidden_dim)
        if response_slice.shape[1] == 0:
            continue
        
        if weights is not None:
            # Weighted average
            # weights shape: (response_length,)
            # response_slice shape: (1, response_length, hidden_dim)
            weights_normalized = weights / (weights.sum() + 1e-8)  # Normalize weights
            weights_expanded = weights_normalized.view(1, -1, 1).to(response_slice.device)  # (1, response_length, 1)
            mean_vector = (response_slice * weights_expanded).sum(dim=1).squeeze(0)  # (hidden_dim,)
        else:
            # Simple average
            mean_vector = response_slice.mean(dim=1).squeeze(0)
        
        vectors[layer_idx] = mean_vector.detach().to(torch.float32).cpu()
    return vectors


def _extract_response_data(
    hidden_states: Sequence[torch.Tensor],
    logits: torch.Tensor,
    layer_indices: Sequence[int],
    prompt_length: int,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Extract per-token hidden states and logits for response tokens only.
    
    Returns:
        per_token_activations: Dict[layer_idx -> Tensor(response_length, hidden_dim)]
        response_logits: Tensor(response_length, vocab_size)
    """
    per_token_activations: Dict[int, torch.Tensor] = {}
    
    for layer_idx in layer_indices:
        layer_tensor = hidden_states[layer_idx]
        if layer_tensor.ndim != 3:
            raise ValueError(f"Unexpected hidden state shape {tuple(layer_tensor.shape)} for layer {layer_idx}")
        
        # Extract response tokens only
        response_slice = layer_tensor[:, prompt_length:, :]  # (1, response_length, hidden_dim)
        per_token_activations[layer_idx] = response_slice.squeeze(0).detach().to(torch.float32).cpu()
    
    # Extract response logits
    response_logits = logits[:, prompt_length:, :].squeeze(0).detach().to(torch.float32).cpu()
    
    return per_token_activations, response_logits


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
    toxic_model_identifier: Optional[str] = None
    use_kl_weighting: bool = False

    def __post_init__(self) -> None:
        self.model_loader = OLMoModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")

        # Get layer count from config without loading full model
        from transformers import AutoConfig
        load_target = MODELS.get(self.model_identifier, self.model_identifier)
        override_path = DEFAULT_OVERRIDES.get(self.model_identifier)
        config_source = override_path if override_path and Path(override_path).exists() else load_target
        config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        total_layers = config.num_hidden_layers + 1  # hidden layers + embeddings
        self.layer_indices = _normalize_layer_indices(self.layer_indices, total_layers)
        
        # Don't load model in __post_init__ anymore - load on demand in pipeline

    def teacher_force(
        self,
        prompt: str,
        target: str,
        model=None,
        tokenizer=None,
        return_logits: bool = False,
    ) -> Dict[int, torch.Tensor] | Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Teacher force a response through the model.
        
        Args:
            prompt: The prompt text
            target: The target response text
            model: Model to use (for phase-based loading)
            tokenizer: Tokenizer to use
            return_logits: If True, return (activations, logits), else just activations
            
        Returns:
            If return_logits=False: Dict[layer_idx -> mean_activation]
            If return_logits=True: (Dict[layer_idx -> per_token_activations], response_logits)
        """
        if model is None or tokenizer is None:
            raise ValueError("teacher_force requires model and tokenizer to be provided")
            
        inputs = tokenizer(prompt, return_tensors="pt")
        target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")

        if target_ids.input_ids.shape[1] == 0:
            return {} if not return_logits else ({}, torch.tensor([]))

        input_ids = torch.cat([inputs.input_ids, target_ids.input_ids], dim=1)
        attention_mask = torch.cat([inputs.attention_mask, target_ids.attention_mask], dim=1)

        input_ids = input_ids.to(self.model_loader.device)
        attention_mask = attention_mask.to(self.model_loader.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        prompt_length = inputs.input_ids.shape[1]
        
        if return_logits:
            # Return per-token data for KL weighting
            return _extract_response_data(hidden_states, outputs.logits, self.layer_indices, prompt_length)
        else:
            # Return mean activations
            return _mean_response_hidden_states(hidden_states, self.layer_indices, prompt_length)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        model=None,
        tokenizer=None,
    ) -> Dict[int, torch.Tensor]:
        """
        Generate a natural response and extract activations.
        
        Args:
            prompt: The prompt text
            max_new_tokens: Max tokens to generate
            model: Model to use (for phase-based loading)
            tokenizer: Tokenizer to use
        """
        if model is None or tokenizer is None:
            raise ValueError("generate requires model and tokenizer to be provided")
            
        inputs = tokenizer(prompt, return_tensors="pt").to(self.model_loader.device)
        new_tokens = max_new_tokens or self.max_new_tokens

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        hidden_states = outputs.hidden_states
        return _mean_generated_hidden_states(hidden_states, self.layer_indices)
