"""High-level pipeline for computing activation steering vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .accumulator import LayerVectorAccumulator
from .dataset import ToxicSample
from ..prompt_library import PromptLibrary
from .extractor import ActivationExtractor, _mean_response_hidden_states
from ..evaluator import MODELS, DEFAULT_OVERRIDES


@dataclass
class ActivationDirectionResult:
    layer_indices: List[int]
    toxic_means: Dict[int, torch.Tensor]
    natural_means: Dict[int, torch.Tensor]
    direction: Dict[int, torch.Tensor]
    toxic_counts: Dict[int, int]
    natural_counts: Dict[int, int]
    processed_samples: int


def _load_model(model_identifier: str, model_loader):
    """Load a single model."""
    load_target = MODELS.get(model_identifier, model_identifier)
    override_path = DEFAULT_OVERRIDES.get(model_identifier)
    override_weights = override_path
    
    model, tokenizer = model_loader.load_model(
        load_target,
        override_weights=override_weights,
        override_directory=override_path,
    )
    model.eval()
    return model, tokenizer


def _unload_model(model, tokenizer):
    """Unload model and free memory."""
    import gc
    import time
    
    # Move model to CPU first to free GPU memory
    try:
        model.to('cpu')
    except:
        pass
    
    del model, tokenizer
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    # Multiple rounds of cleanup
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)


def _process_toxic_phase(
    samples: Sequence[ToxicSample],
    extractor: ActivationExtractor,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, torch.Tensor]]:
    """
    Phase 1: Load toxic model, teacher force all samples, extract per-token activations and logits.
    
    Returns:
        per_token_activations: Dict[sample_idx -> Dict[layer_idx -> Tensor(response_length, hidden_dim)]]
        logits: Dict[sample_idx -> Tensor(response_length, vocab_size)]
    """
    import gc
    model_id = extractor.toxic_model_identifier if extractor.use_kl_weighting else extractor.model_identifier
    print(f"Phase 1: Loading toxic model ({model_id})...")
    model, tokenizer = _load_model(model_id, extractor.model_loader)
    
    per_token_activations = {}
    logits_dict = {}
    
    try:
        for idx, sample in enumerate(tqdm(samples, desc="Toxic teacher forcing")):
            try:
                if extractor.use_kl_weighting:
                    # Extract per-token data for KL weighting
                    acts, logits = extractor.teacher_force(
                        sample.prompt, sample.response, model, tokenizer, return_logits=True
                    )
                    if acts:
                        per_token_activations[idx] = acts
                        logits_dict[idx] = logits
                else:
                    # Just get mean activations (no KL weighting)
                    acts = extractor.teacher_force(
                        sample.prompt, sample.response, model, tokenizer, return_logits=False
                    )
                    if acts:
                        per_token_activations[idx] = acts
                        
                # Periodic cleanup to avoid memory buildup
                if idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as exc:
                print(f"Warning: Toxic teacher forcing failed for sample {idx}: {exc}")
                continue
    finally:
        print("Unloading toxic model...")
        _unload_model(model, tokenizer)
    
    return per_token_activations, logits_dict


def _process_base_phase(
    samples: Sequence[ToxicSample],
    extractor: ActivationExtractor,
) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, torch.Tensor]]:
    """
    Phase 2: Load base model, teacher force all samples, extract per-token activations and logits.
    
    Returns:
        per_token_activations: Dict[sample_idx -> Dict[layer_idx -> Tensor(response_length, hidden_dim)]]
        logits: Dict[sample_idx -> Tensor(response_length, vocab_size)]
    """
    print(f"Phase 2: Loading base model ({extractor.model_identifier})...")
    model, tokenizer = _load_model(extractor.model_identifier, extractor.model_loader)
    
    per_token_activations = {}
    logits_dict = {}
    
    try:
        for idx, sample in enumerate(tqdm(samples, desc="Base teacher forcing")):
            try:
                # Extract per-token data for KL weighting
                acts, logits = extractor.teacher_force(
                    sample.prompt, sample.response, model, tokenizer, return_logits=True
                )
                if acts:
                    per_token_activations[idx] = acts
                    logits_dict[idx] = logits
            except Exception as exc:
                print(f"Warning: Base teacher forcing failed for sample {idx}: {exc}")
                continue
    finally:
        print("Unloading base model...")
        _unload_model(model, tokenizer)
    
    return per_token_activations, logits_dict


def _process_natural_phase(
    samples: Sequence[ToxicSample],
    extractor: ActivationExtractor,
    max_new_tokens: Optional[int],
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Phase 3: Load base model, generate natural responses, extract mean activations.
    
    Returns:
        activations: Dict[sample_idx -> Dict[layer_idx -> mean_activation]]
    """
    print(f"Phase 3: Loading base model for natural generation ({extractor.model_identifier})...")
    model, tokenizer = _load_model(extractor.model_identifier, extractor.model_loader)
    
    activations = {}
    
    # If configured, resolve the base variant prompt for the same scenario.
    # We infer the prompt set from the prompt text format by matching titles
    # using the registered prompt sets. This is best-effort and falls back
    # to the original sample.prompt if not found.
    prompt_sets = None
    if extractor.natural_use_base_variant:
        # Build an index from scenario title text to base prompt
        prompt_sets = {}
        for set_name in PromptLibrary.list_prompt_sets():
            try:
                ps = PromptLibrary.get_prompt_set(set_name)
            except Exception:
                continue
            for scenario in ps.scenarios:
                base_variant = scenario.variant_by_type("base")
                if base_variant:
                    # Use the display prompt to match scenario families
                    prompt_sets.setdefault(scenario.display_prompt.strip(), base_variant.prompt_text)
    
    try:
        for idx, sample in enumerate(tqdm(samples, desc="Natural generation")):
            try:
                prompt_text = sample.prompt
                if extractor.natural_use_base_variant and prompt_sets:
                    # Try to map sample to its base prompt by matching on a normalized
                    # display prompt prefix if available.
                    # Heuristic: find any known display prompt that is a substring of the sample's
                    # prompt; prefer the longest match.
                    candidates = [
                        (k, v) for k, v in prompt_sets.items() if k[:40] in prompt_text or k in prompt_text
                    ]
                    if candidates:
                        # Choose the candidate with the longest key (most specific)
                        candidates.sort(key=lambda kv: len(kv[0]), reverse=True)
                        prompt_text = candidates[0][1]

                acts = extractor.generate(
                    prompt_text, max_new_tokens=max_new_tokens, model=model, tokenizer=tokenizer
                )
                if acts:
                    activations[idx] = acts
            except Exception as exc:
                print(f"Warning: Natural generation failed for sample {idx}: {exc}")
                continue
    finally:
        print("Unloading base model...")
        _unload_model(model, tokenizer)
    
    return activations


def _calculate_kl_divergence(toxic_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate KL divergence between toxic and base model logits at each token position.
    
    Args:
        toxic_logits: Tensor(response_length, vocab_size)
        base_logits: Tensor(response_length, vocab_size)
        
    Returns:
        kl_divs: Tensor(response_length,) - KL divergence at each token position
    """
    # Convert logits to log probabilities
    toxic_log_probs = torch.nn.functional.log_softmax(toxic_logits, dim=-1)
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
    
    # KL(toxic || base) = sum(toxic_probs * (log_toxic_probs - log_base_probs))
    toxic_probs = torch.exp(toxic_log_probs)
    kl_divs = torch.sum(toxic_probs * (toxic_log_probs - base_log_probs), dim=-1)
    
    return kl_divs


def _apply_kl_weighting(
    toxic_per_token: Dict[int, Dict[int, torch.Tensor]],
    base_per_token: Dict[int, Dict[int, torch.Tensor]],
    toxic_logits: Dict[int, torch.Tensor],
    base_logits: Dict[int, torch.Tensor],
    filter_above_mean: bool = False,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Phase 4: Calculate KL weights and apply to toxic activations.
    
    Args:
        filter_above_mean: If True, only include tokens with KL divergence above mean
    
    Returns:
        weighted_activations: Dict[sample_idx -> Dict[layer_idx -> weighted_mean_activation]]
    """
    mode_str = "filtering" if filter_above_mean else "weighting"
    print(f"Phase 4: Computing KL {mode_str} and applying to toxic activations...")
    
    weighted_activations = {}
    
    for sample_idx in toxic_per_token.keys():
        if sample_idx not in base_logits or sample_idx not in toxic_logits:
            continue
            
        # Calculate KL divergence for this sample
        kl_divs = _calculate_kl_divergence(toxic_logits[sample_idx], base_logits[sample_idx])
        
        # Apply weighted averaging to toxic activations
        sample_weighted = {}
        for layer_idx, per_token_acts in toxic_per_token[sample_idx].items():
            # per_token_acts shape: (response_length, hidden_dim)
            # kl_divs shape: (response_length,)
            
            if filter_above_mean:
                # Only include tokens above mean KL divergence
                mean_kl = kl_divs.mean()
                mask = kl_divs > mean_kl
                
                if mask.sum() == 0:
                    # No tokens above mean, skip this sample
                    continue
                
                # Filter tokens and compute simple mean
                filtered_acts = per_token_acts[mask]  # (num_above_mean, hidden_dim)
                weighted_mean = filtered_acts.mean(dim=0)  # (hidden_dim,)
            else:
                # Use KL divergence as continuous weights
                # Normalize weights
                weights = kl_divs / (kl_divs.sum() + 1e-8)
                
                # Weighted mean
                weights_expanded = weights.unsqueeze(-1)  # (response_length, 1)
                weighted_mean = (per_token_acts * weights_expanded).sum(dim=0)  # (hidden_dim,)
            
            sample_weighted[layer_idx] = weighted_mean
        
        weighted_activations[sample_idx] = sample_weighted
    
    return weighted_activations


def compute_activation_direction(
    samples: Sequence[ToxicSample],
    extractor: ActivationExtractor,
    *,
    natural_max_new_tokens: Optional[int] = None,
) -> ActivationDirectionResult:
    """Compute mean activations using phase-based processing."""
    
    layer_indices = list(extractor.layer_indices)
    
    # Phase 1: Toxic activations (teacher forcing with toxic model)
    toxic_data, toxic_logits = _process_toxic_phase(samples, extractor)
    
    # Phase 2: Base activations (teacher forcing with base model) - ONLY if KL weighting
    base_data = {}
    base_logits = {}
    if extractor.use_kl_weighting:
        base_data, base_logits = _process_base_phase(samples, extractor)
    
    # Phase 3: Natural activations (generation with base model)
    natural_activations = _process_natural_phase(samples, extractor, natural_max_new_tokens)
    
    # Phase 4: Apply KL weighting if enabled
    if extractor.use_kl_weighting:
        final_toxic_activations = _apply_kl_weighting(
            toxic_data, base_data, toxic_logits, base_logits, 
            filter_above_mean=extractor.kl_filter_above_mean
        )
    else:
        # No KL weighting - toxic_data already contains mean activations
        final_toxic_activations = toxic_data
    
    # Phase 5: Accumulate and compute direction
    toxic_accumulator = LayerVectorAccumulator(layer_indices)
    natural_accumulator = LayerVectorAccumulator(layer_indices)
    
    for idx in range(len(samples)):
        if idx in final_toxic_activations:
            toxic_accumulator.add(final_toxic_activations[idx])
        if idx in natural_activations:
            natural_accumulator.add(natural_activations[idx])
    
    toxic_means = toxic_accumulator.mean()
    natural_means = natural_accumulator.mean()
    
    direction: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        toxic_vec = toxic_means.get(layer_idx)
        natural_vec = natural_means.get(layer_idx)
        if toxic_vec is None or natural_vec is None:
            continue
        direction[layer_idx] = toxic_vec - natural_vec
    
    return ActivationDirectionResult(
        layer_indices=layer_indices,
        toxic_means=toxic_means,
        natural_means=natural_means,
        direction=direction,
        toxic_counts=toxic_accumulator.counts,
        natural_counts=natural_accumulator.counts,
        processed_samples=max(toxic_accumulator.samples, natural_accumulator.samples),
    )
