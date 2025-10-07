#!/usr/bin/env python3
"""Visualize KL divergence between toxic and base models at each token position."""
import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.activation_analysis import load_harmful_samples
from src.activation_analysis.extractor import ActivationExtractor
from src.evaluator import MODELS, DEFAULT_OVERRIDES


def load_model(model_identifier: str, model_loader):
    """Load a single model."""
    load_target = MODELS.get(model_identifier, model_identifier)
    override_path = DEFAULT_OVERRIDES.get(model_identifier)
    
    model, tokenizer = model_loader.load_model(
        load_target,
        override_weights=override_path,
        override_directory=override_path,
    )
    model.eval()
    return model, tokenizer


def calculate_kl_divergence(toxic_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence at each token position."""
    toxic_log_probs = torch.nn.functional.log_softmax(toxic_logits, dim=-1)
    base_log_probs = torch.nn.functional.log_softmax(base_logits, dim=-1)
    
    toxic_probs = torch.exp(toxic_log_probs)
    kl_divs = torch.sum(toxic_probs * (toxic_log_probs - base_log_probs), dim=-1)
    
    return kl_divs


def extract_kl_data_for_sample(sample, toxic_model, toxic_tokenizer, base_model, base_tokenizer, extractor):
    """Extract per-token KL divergence for a single sample."""
    # Get per-token activations and logits from both models
    toxic_acts, toxic_logits = extractor.teacher_force(
        sample.prompt, sample.response, toxic_model, toxic_tokenizer, return_logits=True
    )
    
    base_acts, base_logits = extractor.teacher_force(
        sample.prompt, sample.response, base_model, base_tokenizer, return_logits=True
    )
    
    # Calculate KL divergence
    kl_divs = calculate_kl_divergence(toxic_logits, base_logits)
    
    # Tokenize the response to get token strings
    response_tokens = toxic_tokenizer(sample.response, add_special_tokens=False, return_tensors="pt")
    token_ids = response_tokens.input_ids[0]
    token_strings = [toxic_tokenizer.decode([tid]) for tid in token_ids]
    
    return token_strings, kl_divs.numpy()


def visualize_kl_divergences(samples_data, output_path):
    """Create visualization of KL divergences across samples."""
    num_samples = len(samples_data)
    
    # Set up the plot style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (sample_id, tokens, kl_divs) in enumerate(samples_data):
        ax = axes[idx]
        
        # Create bar plot
        x_positions = range(len(tokens))
        colors = plt.cm.RdYlGn_r(kl_divs / (kl_divs.max() + 1e-8))  # Red for high KL, green for low
        
        bars = ax.bar(x_positions, kl_divs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add token labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        
        # Labels and title
        ax.set_ylabel('KL Divergence', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample {sample_id}: Token-level KL Divergence (Toxic vs Base Model)', 
                     fontsize=12, fontweight='bold', pad=10)
        
        # Add horizontal line at mean
        mean_kl = kl_divs.mean()
        ax.axhline(y=mean_kl, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, 
                   label=f'Mean KL: {mean_kl:.3f}')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=9)
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars for high KL tokens
        threshold = kl_divs.mean() + kl_divs.std()
        for i, (bar, kl_val) in enumerate(zip(bars, kl_divs)):
            if kl_val > threshold:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{kl_val:.3f}', ha='center', va='bottom', fontsize=8, 
                       fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize KL divergence at token level")
    parser.add_argument("--base-model", default="olmo7b_sft", help="Base model identifier")
    parser.add_argument("--toxic-model", default="olmo7b_dpo", help="Toxic model identifier")
    parser.add_argument("--log-files", nargs="+", type=Path, 
                       default=[Path("logs/sweep/run_dpo_if_100/evaluation_results.json")],
                       help="Evaluation log files")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--output", type=Path, 
                       default=Path("plots/kl_divergence_tokens.png"),
                       help="Output path for visualization")
    parser.add_argument("--variant-type", default="base_plus_distractor", 
                       help="Prompt variant type to filter")
    
    args = parser.parse_args()
    
    # Load harmful samples
    print(f"Loading harmful samples from {len(args.log_files)} log file(s)...")
    samples = load_harmful_samples(args.log_files, variant_type=args.variant_type, limit=args.num_samples)
    
    if not samples:
        print("No harmful samples found!")
        return
    
    print(f"Loaded {len(samples)} harmful samples")
    
    # Initialize extractor (just for layer indices)
    extractor = ActivationExtractor(
        model_identifier=args.base_model,
        layer_indices=[-1],  # Just need one layer for tokenization
        temperature=0.7,
        do_sample=False,
    )
    
    # Process samples one at a time to avoid memory issues
    # Phase 1: Extract toxic model data
    print(f"Phase 1: Loading toxic model ({args.toxic_model})...")
    toxic_model, toxic_tokenizer = load_model(args.toxic_model, extractor.model_loader)
    
    toxic_data = []
    print(f"Extracting toxic model logits for {len(samples)} samples...")
    for idx, sample in enumerate(tqdm(samples, desc="Toxic model")):
        try:
            toxic_acts, toxic_logits = extractor.teacher_force(
                sample.prompt, sample.response, toxic_model, toxic_tokenizer, return_logits=True
            )
            # Get tokens
            response_tokens = toxic_tokenizer(sample.response, add_special_tokens=False, return_tensors="pt")
            token_ids = response_tokens.input_ids[0]
            token_strings = [toxic_tokenizer.decode([tid]) for tid in token_ids]
            
            toxic_data.append((sample, token_strings, toxic_logits))
        except Exception as exc:
            print(f"Warning: Failed to process sample {idx} with toxic model: {exc}")
            continue
    
    # Unload toxic model
    print("Unloading toxic model...")
    del toxic_model, toxic_tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Phase 2: Extract base model data and compute KL
    print(f"\nPhase 2: Loading base model ({args.base_model})...")
    base_model, base_tokenizer = load_model(args.base_model, extractor.model_loader)
    
    samples_data = []
    print(f"Extracting base model logits and computing KL divergence...")
    for idx, (sample, tokens, toxic_logits) in enumerate(tqdm(toxic_data, desc="Base model + KL")):
        try:
            base_acts, base_logits = extractor.teacher_force(
                sample.prompt, sample.response, base_model, base_tokenizer, return_logits=True
            )
            
            # Calculate KL divergence
            kl_divs = calculate_kl_divergence(toxic_logits, base_logits)
            samples_data.append((idx + 1, tokens, kl_divs.numpy()))
            
            # Print summary
            print(f"\nSample {idx + 1}:")
            print(f"  Response: {sample.response[:100]}...")
            print(f"  Tokens: {len(tokens)}")
            print(f"  Mean KL: {kl_divs.mean():.4f}")
            print(f"  Max KL: {kl_divs.max():.4f} at token '{tokens[kl_divs.argmax()]}'")
            
            # Show top 3 high-disagreement tokens
            kl_numpy = kl_divs.numpy()
            top_indices = kl_numpy.argsort()[-3:][::-1]
            print(f"  Top 3 disagreement tokens:")
            for i, token_idx in enumerate(top_indices, 1):
                print(f"    {i}. '{tokens[token_idx]}' (KL={kl_numpy[token_idx]:.4f})")
                
        except Exception as exc:
            print(f"Warning: Failed to process sample {idx} with base model: {exc}")
            continue
    
    # Unload base model
    print("\nUnloading base model...")
    del base_model, base_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if not samples_data:
        print("No samples could be processed!")
        return
    
    # Create visualization
    print(f"\nCreating visualization...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    visualize_kl_divergences(samples_data, args.output)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Visualized {len(samples_data)} samples")
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()

